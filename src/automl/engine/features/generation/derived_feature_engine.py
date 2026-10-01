"""Robust, sandboxed derived feature engine with AST formula evaluation and safe Python execution."""
from __future__ import annotations

import ast
import re
from typing import Any, Callable

import numpy as np
import pandas as pd

from automl.domain.features.derived_feature import (
    DerivedFeatureDefinition,
    DerivedFeatureType,
    FeatureEvaluationResult,
)
from automl.domain.features.feature_set import FeatureSet


# ---------------------------------------------------------------------------
# Safe Math Library for Formula Calculator
# ---------------------------------------------------------------------------


def _safe_log(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.log(np.maximum(1e-9, arr))


def _safe_log1p(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.log1p(np.maximum(0.0, arr))


def _safe_exp(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.exp(np.clip(arr, -50.0, 50.0))


def _safe_sqrt(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.sqrt(np.maximum(0.0, arr))


def _safe_zscore(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - mean) / (std + 1e-7)


def _safe_clip(x: Any, low: Any, high: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.clip(arr, float(low), float(high))


def _safe_where(cond: Any, x: Any, y: Any) -> np.ndarray:
    cond_arr = np.asarray(cond, dtype=bool)
    return np.where(cond_arr, x, y)


def _safe_fillna(x: Any, val: Any = 0.0) -> pd.Series:
    return pd.Series(x).fillna(val)


SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "log": _safe_log,
    "log1p": _safe_log1p,
    "exp": _safe_exp,
    "sqrt": _safe_sqrt,
    "abs": np.abs,
    "clip": _safe_clip,
    "round": lambda x, d=0: np.round(x, int(d)),
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "power": lambda x, p: np.power(x, np.clip(p, -50.0, 50.0)),
    "zscore": _safe_zscore,
    "min": np.minimum,
    "max": np.maximum,
    "where": _safe_where,
    "if_else": _safe_where,
    "fillna": _safe_fillna,
    "mean": np.nanmean,
    "std": np.nanstd,
}

SAFE_CONSTANTS: dict[str, Any] = {
    "pi": np.pi,
    "e": np.e,
    "nan": np.nan,
    "inf": np.inf,
}


# ---------------------------------------------------------------------------
# AST-Based Formula Calculator
# ---------------------------------------------------------------------------


class SafeFormulaCalculator:
    """Evaluates mathematical expressions over pandas columns using a strict AST whitelist."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.zero_division_occurred = False
        self.infinite_values_handled = 0

    def evaluate(self, expression: str) -> tuple[pd.Series, list[str]]:
        """Parses and computes the formula expression, returning the resulting Series and warnings."""
        if not expression or not expression.strip():
            raise ValueError("Formula expression cannot be empty.")

        self.zero_division_occurred = False
        self.infinite_values_handled = 0
        warnings: list[str] = []

        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as err:
            raise ValueError(f"Syntax error in formula '{expression}': {err}") from err

        self._validate_ast(tree.body)
        raw_result = self._eval_node(tree.body)

        # Convert scalar or array to pandas Series aligned with dataframe index
        if np.isscalar(raw_result):
            series = pd.Series(raw_result, index=self.df.index)
        elif isinstance(raw_result, (list, tuple, np.ndarray)):
            series = pd.Series(raw_result, index=self.df.index)
        elif isinstance(raw_result, pd.Series):
            series = raw_result.reindex(self.df.index)
        else:
            raise TypeError(f"Formula evaluation produced unsupported type: {type(raw_result).__name__}")

        # Post-process infinities and NaNs
        if pd.api.types.is_numeric_dtype(series):
            inf_mask = np.isinf(series.to_numpy(dtype=float, na_value=0.0))
            inf_count = int(np.sum(inf_mask))
            if inf_count > 0:
                self.infinite_values_handled += inf_count
                warnings.append(f"Clamped {inf_count} infinite values to safe numerical bounds.")
                clean_arr = np.nan_to_num(series.to_numpy(dtype=float), nan=np.nan, posinf=1e9, neginf=-1e9)
                series = pd.Series(clean_arr, index=self.df.index)

        if self.zero_division_occurred:
            warnings.append("Division by zero occurred and was safely insulated with zero-fill / epsilon bounds.")

        return series, warnings

    def _validate_ast(self, node: ast.AST) -> None:
        """Walks the AST ensuring no dangerous nodes or attribute access exist."""
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.BoolOp,
            ast.Call,
            ast.Name,
            ast.Constant,
            ast.IfExp,
            ast.keyword,
            ast.operator,
            ast.unaryop,
            ast.cmpop,
            ast.boolop,
            ast.expr_context,
        )

        for subnode in ast.walk(node):
            if not isinstance(subnode, allowed_nodes):
                raise ValueError(
                    f"Forbidden operation '{type(subnode).__name__}' in formula. Only safe arithmetic and mathematical functions are permitted."
                )

            # Check function calls
            if isinstance(subnode, ast.Call):
                if not isinstance(subnode.func, ast.Name):
                    raise ValueError("Indirect or chained function calls are forbidden.")
                if subnode.func.id not in SAFE_FUNCTIONS:
                    raise ValueError(
                        f"Function '{subnode.func.id}' is not in the allowed functions whitelist: {sorted(SAFE_FUNCTIONS.keys())}"
                    )

            # Check variable / column names
            if isinstance(subnode, ast.Name):
                name = subnode.id
                if (
                    name not in self.df.columns
                    and name not in SAFE_FUNCTIONS
                    and name not in SAFE_CONSTANTS
                ):
                    raise KeyError(f"Column or variable '{name}' not found in dataset columns.")

    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluates the AST node using vector operations."""
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in self.df.columns:
                return self.df[node.id]
            if node.id in SAFE_CONSTANTS:
                return SAFE_CONSTANTS[node.id]
            raise KeyError(f"Identifier '{node.id}' not found.")

        if isinstance(node, ast.UnaryOp):
            val = self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return ~val
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._eval_binary_op(node.op, left, right)

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            result = None
            curr_left = left
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                cmp_res = self._eval_comparison(op, curr_left, right)
                result = cmp_res if result is None else (result & cmp_res)
                curr_left = right
            return result

        if isinstance(node, ast.BoolOp):
            vals = [self._eval_node(val) for val in node.values]
            res = vals[0]
            for v in vals[1:]:
                if isinstance(node.op, ast.And):
                    res = res & v
                elif isinstance(node.op, ast.Or):
                    res = res | v
            return res

        if isinstance(node, ast.IfExp):
            cond = self._eval_node(node.test)
            body = self._eval_node(node.body)
            orelse = self._eval_node(node.orelse)
            return np.where(cond, body, orelse)

        if isinstance(node, ast.Call):
            func_name = node.func.id  # type: ignore[attr-defined]
            fn = SAFE_FUNCTIONS[func_name]
            args = [self._eval_node(a) for a in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value) for kw in node.keywords if kw.arg}
            return fn(*args, **kwargs)

        raise ValueError(f"Evaluation of node '{type(node).__name__}' is unsupported.")

    def _eval_binary_op(self, op: ast.operator, left: Any, right: Any) -> Any:
        """Safely executes binary operations with zero-division insulation."""
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            # Safe division
            right_arr = np.asarray(right, dtype=float)
            zero_mask = right_arr == 0.0
            if np.any(zero_mask):
                self.zero_division_occurred = True

            safe_denom = np.where(zero_mask, 1.0, right_arr)
            res = left / safe_denom
            # Where denom was zero, assign 0.0
            if isinstance(res, pd.Series):
                res = res.mask(pd.Series(zero_mask, index=res.index), 0.0)
            else:
                res = np.where(zero_mask, 0.0, res)

            if isinstance(op, ast.FloorDiv):
                return np.floor(res)
            return res

        if isinstance(op, ast.Mod):
            right_arr = np.asarray(right, dtype=float)
            zero_mask = right_arr == 0.0
            if np.any(zero_mask):
                self.zero_division_occurred = True
            safe_denom = np.where(zero_mask, 1.0, right_arr)
            res = left % safe_denom
            return np.where(zero_mask, 0.0, res)

        if isinstance(op, ast.Pow):
            return np.power(left, np.clip(right, -50.0, 50.0))

        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.BitXor):
            return left ^ right

        raise ValueError(f"Binary operator '{type(op).__name__}' is unsupported.")

    def _eval_comparison(self, op: ast.cmpop, left: Any, right: Any) -> Any:
        """Evaluates comparison operators."""
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        raise ValueError(f"Comparison operator '{type(op).__name__}' is unsupported.")


# ---------------------------------------------------------------------------
# Sandboxed Python Code Evaluator
# ---------------------------------------------------------------------------


class SafePythonEvaluator:
    """Executes sandboxed Python feature transformations with restricted globals and AST validation."""

    FORBIDDEN_NAMES = {
        "open", "exec", "eval", "compile", "getattr", "setattr", "delattr",
        "__import__", "globals", "locals", "vars", "breakpoint", "exit", "quit",
        "input", "os", "sys", "subprocess", "requests", "urllib", "socket",
        "shutil", "tempfile", "posix", "pty",
    }

    ALLOWED_BUILTINS = {
        "range": range, "len": len, "int": int, "float": float, "str": str,
        "bool": bool, "list": list, "dict": dict, "tuple": tuple, "enumerate": enumerate,
        "zip": zip, "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
        "isinstance": isinstance, "print": lambda *args: None,
        "ZeroDivisionError": ZeroDivisionError, "ValueError": ValueError,
        "TypeError": TypeError, "KeyError": KeyError, "IndexError": IndexError,
        "Exception": Exception,
    }

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def evaluate(self, code: str) -> tuple[pd.Series, list[str]]:
        """Validates AST security and executes sandboxed Python code."""
        if not code or not code.strip():
            raise ValueError("Python feature code cannot be empty.")

        warnings: list[str] = []
        try:
            tree = ast.parse(code.strip())
        except SyntaxError as err:
            raise ValueError(f"Syntax error in Python feature code: {err}") from err

        self._validate_ast(tree)

        # Prepare sandboxed namespace
        sandboxed_globals: dict[str, Any] = {
            "__builtins__": self.ALLOWED_BUILTINS,
            "np": np,
            "pd": pd,
            "df": self.df.copy(),
        }
        sandboxed_locals: dict[str, Any] = {}

        # Execute in sandboxed environment
        try:
            exec(compile(tree, filename="<derived_feature>", mode="exec"), sandboxed_globals, sandboxed_locals)
        except Exception as exc:
            raise RuntimeError(f"Error compiling sandboxed Python code: {exc}") from exc

        # Resolve callable function or result variable
        compute_fn = None
        for candidate_name in ("compute_feature", "transform", "calculate", "derive"):
            if candidate_name in sandboxed_locals and callable(sandboxed_locals[candidate_name]):
                compute_fn = sandboxed_locals[candidate_name]
                break

        if compute_fn is None:
            # Check if code assigned 'result' directly or defined a single lambda
            if "result" in sandboxed_locals:
                raw_result = sandboxed_locals["result"]
            else:
                callables = [v for v in sandboxed_locals.values() if callable(v)]
                if len(callables) == 1:
                    compute_fn = callables[0]
                else:
                    raise ValueError(
                        "Code must define a callable function (e.g. `def compute_feature(df):`) or assign to `result`."
                    )

        if compute_fn is not None:
            try:
                # Pass a protected copy of dataframe to prevent mutation
                df_copy = self.df.copy()
                raw_result = compute_fn(df_copy)
            except ZeroDivisionError:
                raw_result = pd.Series(0.0, index=self.df.index)
                warnings.append("ZeroDivisionError trapped in Python execution; filled with zeros.")
            except Exception as exc:
                raise RuntimeError(f"Runtime error in feature function: {exc}") from exc

        # Format and validate resulting Series
        if np.isscalar(raw_result):
            series = pd.Series(raw_result, index=self.df.index)
        elif isinstance(raw_result, (list, tuple, np.ndarray)):
            if len(raw_result) != len(self.df):
                raise ValueError(
                    f"Result length ({len(raw_result)}) does not match dataset row count ({len(self.df)})."
                )
            series = pd.Series(raw_result, index=self.df.index)
        elif isinstance(raw_result, pd.Series):
            if len(raw_result) != len(self.df):
                raise ValueError(
                    f"Result series length ({len(raw_result)}) does not match dataset row count ({len(self.df)})."
                )
            series = raw_result.reindex(self.df.index)
        else:
            raise TypeError(f"Unsupported return type: {type(raw_result).__name__}")

        # Post-process infinities
        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype=float, na_value=np.nan)
            inf_count = int(np.sum(np.isinf(arr)))
            if inf_count > 0:
                clean_arr = np.nan_to_num(arr, nan=np.nan, posinf=1e9, neginf=-1e9)
                series = pd.Series(clean_arr, index=self.df.index)
                warnings.append(f"Sanitized {inf_count} infinite values to safe bounds.")

        return series, warnings

    def _validate_ast(self, tree: ast.AST) -> None:
        """Strictly inspects AST to forbid system imports, dunder methods, and dangerous constructs."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Importing modules in feature code is strictly forbidden.")

            if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.ClassDef)):
                raise ValueError(f"AST node '{type(node).__name__}' is disallowed in feature code.")

            if isinstance(node, ast.Name):
                if node.id in self.FORBIDDEN_NAMES or node.id.startswith("__"):
                    raise ValueError(f"Access to forbidden identifier '{node.id}' is prohibited.")

            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    raise ValueError(f"Access to dunder attribute '{node.attr}' is prohibited.")


# ---------------------------------------------------------------------------
# Derived Feature Engine (Orchestrator)
# ---------------------------------------------------------------------------


class DerivedFeatureEngine:
    """Main service orchestrating validation, evaluation, and application of derived features."""

    def evaluate(
        self,
        df: pd.DataFrame,
        definition: DerivedFeatureDefinition,
    ) -> tuple[FeatureEvaluationResult, pd.Series | None]:
        """Evaluates and validates a derived feature definition against dataframe without modifying it."""
        feature_name = definition.name.strip()
        if not feature_name:
            return (
                FeatureEvaluationResult(
                    is_valid=False,
                    feature_name="",
                    error_message="Feature name cannot be empty.",
                ),
                None,
            )

        # Check name validity (alphanumeric and underscores)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", feature_name):
            return (
                FeatureEvaluationResult(
                    is_valid=False,
                    feature_name=feature_name,
                    error_message=f"Invalid feature name '{feature_name}'. Use alphanumeric characters and underscores.",
                ),
                None,
            )

        try:
            if definition.expression_type == DerivedFeatureType.FORMULA:
                calc = SafeFormulaCalculator(df)
                series, warnings = calc.evaluate(definition.expression)
                zero_div = calc.zero_division_occurred
                inf_handled = calc.infinite_values_handled
            elif definition.expression_type == DerivedFeatureType.PYTHON_CODE:
                py_eval = SafePythonEvaluator(df)
                series, warnings = py_eval.evaluate(definition.expression)
                zero_div = False
                inf_handled = 0
            else:
                return (
                    FeatureEvaluationResult(
                        is_valid=False,
                        feature_name=feature_name,
                        error_message=f"Unsupported expression type '{definition.expression_type}'.",
                    ),
                    None,
                )

            # Analyze output series
            row_count = len(series)
            null_count = int(series.isna().sum())
            null_pct = float(null_count / row_count) if row_count > 0 else 0.0

            if null_pct > 0.50:
                warnings.append(f"High null rate ({null_pct:.1%}) in generated feature.")

            # Constant column check
            is_constant = False
            nunique = series.nunique(dropna=True)
            if nunique <= 1:
                is_constant = True
                warnings.append("Feature is constant (zero variance across all observations).")

            # Summary stats for numeric series
            summary_stats: dict[str, float] = {}
            if pd.api.types.is_numeric_dtype(series):
                clean_s = series.dropna()
                if len(clean_s) > 0:
                    summary_stats = {
                        "min": float(clean_s.min()),
                        "max": float(clean_s.max()),
                        "mean": float(clean_s.mean()),
                        "std": float(clean_s.std(ddof=0)),
                        "median": float(clean_s.median()),
                    }

            # Sample values (up to 5 preview rows)
            sample_vals = [
                None if pd.isna(v) else (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                for v in series.head(5).tolist()
            ]

            result = FeatureEvaluationResult(
                is_valid=True,
                feature_name=feature_name,
                sample_values=sample_vals,
                dtype=str(series.dtype),
                row_count=row_count,
                null_count=null_count,
                null_percentage=round(null_pct, 4),
                is_constant=is_constant,
                zero_division_occurred=zero_div,
                infinite_values_handled=inf_handled,
                warnings=warnings,
                error_message=None,
                summary_stats=summary_stats,
            )
            return result, series

        except Exception as exc:
            return (
                FeatureEvaluationResult(
                    is_valid=False,
                    feature_name=feature_name,
                    error_message=str(exc),
                ),
                None,
            )

    def apply(
        self,
        df: pd.DataFrame,
        definition: DerivedFeatureDefinition,
    ) -> tuple[pd.DataFrame, FeatureEvaluationResult]:
        """Applies the derived feature, returning a new DataFrame containing the added column."""
        result, series = self.evaluate(df, definition)
        if not result.is_valid or series is None:
            raise ValueError(f"Failed to apply derived feature: {result.error_message}")

        out_df = df.copy()
        out_df[definition.name] = series
        return out_df, result

    def propose_candidate_feature_set(
        self,
        base_features: list[str],
        derived_feature_names: list[str],
        dataset_id: str,
        name: str = "derived_features",
        lineage: str = "",
    ) -> FeatureSet:
        """Packages derived features into an explicit FeatureSet candidate for experimentation."""
        import uuid

        fs_id = f"fs_derived_{uuid.uuid4().hex[:6]}"
        return FeatureSet(
            id=fs_id,
            dataset_id=dataset_id,
            name=name,
            feature_names=base_features + derived_feature_names,
            created_by="derived_feature_engine",
            lineage=lineage or f"Base ({len(base_features)}) + {len(derived_feature_names)} derived features",
        )
