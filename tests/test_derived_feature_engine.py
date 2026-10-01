from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.agents.specialists.feature_advisor import FeatureAdvisor
from automl.domain.features.derived_feature import (
    DerivedFeatureDefinition,
    DerivedFeatureType,
    FeatureEvaluationResult,
)
from automl.engine.features.generation.derived_feature_engine import (
    DerivedFeatureEngine,
    SafeFormulaCalculator,
    SafePythonEvaluator,
)
from automl.interfaces.web.server import AutoMLWebHandler


# ---------------------------------------------------------------------------
# 1. Domain Entities Tests
# ---------------------------------------------------------------------------


def test_derived_feature_definition_serialization():
    defn = DerivedFeatureDefinition(
        name="debt_to_income",
        expression_type=DerivedFeatureType.FORMULA,
        expression="debt / (income + 1e-5)",
        description="Leverage ratio",
        source_columns=["debt", "income"],
        created_by="llm_agent",
        metadata={"domain": "finance"},
    )
    data = defn.to_dict()
    assert data["name"] == "debt_to_income"
    assert data["expression_type"] == "formula"
    assert data["expression"] == "debt / (income + 1e-5)"
    assert data["created_by"] == "llm_agent"

    restored = DerivedFeatureDefinition.from_dict(data)
    assert restored.name == defn.name
    assert restored.expression_type == DerivedFeatureType.FORMULA
    assert restored.source_columns == ["debt", "income"]
    assert restored.metadata["domain"] == "finance"


def test_feature_evaluation_result_serialization():
    res = FeatureEvaluationResult(
        is_valid=True,
        feature_name="log_sales",
        sample_values=[1.0, 2.5, 3.8],
        dtype="float64",
        row_count=100,
        null_count=2,
        null_percentage=0.02,
        is_constant=False,
        zero_division_occurred=True,
        infinite_values_handled=1,
        warnings=["Division by zero insulated"],
        summary_stats={"min": 0.0, "max": 10.0, "mean": 5.0},
    )
    data = res.to_dict()
    assert data["is_valid"] is True
    assert data["feature_name"] == "log_sales"
    assert data["zero_division_occurred"] is True

    restored = FeatureEvaluationResult.from_dict(data)
    assert restored.is_valid is True
    assert restored.null_percentage == 0.02
    assert restored.summary_stats["mean"] == 5.0


# ---------------------------------------------------------------------------
# 2. SafeFormulaCalculator Tests (Math, AST, Zero-Div, Sandboxing)
# ---------------------------------------------------------------------------


def test_formula_calculator_basic_arithmetic():
    df = pd.DataFrame({
        "a": [10.0, 20.0, 30.0],
        "b": [2.0, 5.0, 10.0],
    })
    calc = SafeFormulaCalculator(df)

    res, warnings = calc.evaluate("a + b")
    assert np.allclose(res, [12.0, 25.0, 40.0])

    res, _ = calc.evaluate("a * b - 5")
    assert np.allclose(res, [15.0, 95.0, 295.0])

    res, _ = calc.evaluate("a / b")
    assert np.allclose(res, [5.0, 4.0, 3.0])


def test_formula_calculator_zero_division_protection():
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0],
        "denom": [2.0, 0.0, 5.0],
    })
    calc = SafeFormulaCalculator(df)

    res, warnings = calc.evaluate("num / denom")
    # At index 1, denom is 0.0 -> must be protected and set to 0.0
    assert res.iloc[1] == 0.0
    assert res.iloc[0] == 5.0
    assert res.iloc[2] == 6.0
    assert calc.zero_division_occurred is True
    assert any("division by zero" in w.lower() for w in warnings)


def test_formula_calculator_math_functions():
    df = pd.DataFrame({
        "x": [0.0, 1.0, 10.0, -5.0],
        "y": [1.0, 4.0, 9.0, 16.0],
    })
    calc = SafeFormulaCalculator(df)

    # Safe sqrt: negative values clamped to 0.0
    res, _ = calc.evaluate("sqrt(x)")
    assert res.iloc[0] == 0.0
    assert res.iloc[1] == 1.0
    assert res.iloc[3] == 0.0

    # Safe log1p
    res, _ = calc.evaluate("log1p(x)")
    assert res.iloc[0] == 0.0
    assert np.isclose(res.iloc[1], np.log1p(1.0))

    # Safe clip
    res, _ = calc.evaluate("clip(x, 0, 5)")
    assert res.iloc[0] == 0.0
    assert res.iloc[2] == 5.0
    assert res.iloc[3] == 0.0

    # Safe zscore
    res, _ = calc.evaluate("zscore(y)")
    assert np.isclose(np.mean(res), 0.0, atol=1e-5)

    # Ternary / conditional where
    res, _ = calc.evaluate("where(x > 0, 1, 0)")
    assert res.tolist() == [0, 1, 1, 0]


def test_formula_calculator_comparisons_and_boolean_ops():
    df = pd.DataFrame({
        "a": [1, 5, 10],
        "b": [2, 5, 8],
    })
    calc = SafeFormulaCalculator(df)

    res, _ = calc.evaluate("(a >= 5) & (b <= 5)")
    assert res.tolist() == [False, True, False]

    res, _ = calc.evaluate("if_else(a == b, 100, 0)")
    assert res.tolist() == [0, 100, 0]


def test_formula_calculator_security_sandbox():
    df = pd.DataFrame({"x": [1, 2, 3]})
    calc = SafeFormulaCalculator(df)

    # Disallowed system calls
    with pytest.raises(ValueError, match="not in the allowed functions whitelist"):
        calc.evaluate("open('/etc/passwd')")

    with pytest.raises(ValueError, match="Forbidden|forbidden|not found"):
        calc.evaluate("__import__('os').system('ls')")

    with pytest.raises(ValueError, match="whitelist|Forbidden|forbidden"):
        calc.evaluate("eval('1 + 1')")

    with pytest.raises(ValueError, match="Forbidden operation"):
        calc.evaluate("[x for x in [1, 2]]")

    # Missing column
    with pytest.raises(KeyError, match="not found in dataset columns"):
        calc.evaluate("x + non_existent_col")

    # Empty formula
    with pytest.raises(ValueError, match="cannot be empty"):
        calc.evaluate("   ")


# ---------------------------------------------------------------------------
# 3. SafePythonEvaluator Tests (Sandboxed Python Code)
# ---------------------------------------------------------------------------


def test_python_evaluator_function_mode():
    df = pd.DataFrame({
        "val_a": [10.0, 20.0, 30.0],
        "val_b": [2.0, 4.0, 5.0],
    })
    py_eval = SafePythonEvaluator(df)

    code = """
def compute_feature(df):
    ratio = df['val_a'] / (df['val_b'] + 1e-5)
    return np.log1p(ratio)
"""
    series, warnings = py_eval.evaluate(code)
    assert len(series) == 3
    assert np.isclose(series.iloc[0], np.log1p(10.0 / 2.00001))


def test_python_evaluator_result_assignment_mode():
    df = pd.DataFrame({"x": [1, 2, 3]})
    py_eval = SafePythonEvaluator(df)

    code = "result = df['x'] * 10 + 5"
    series, _ = py_eval.evaluate(code)
    assert series.tolist() == [15, 25, 35]


def test_python_evaluator_dimension_mismatch_error():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    py_eval = SafePythonEvaluator(df)

    code = """
def compute_feature(df):
    return [1, 2]  # Wrong length!
"""
    with pytest.raises(ValueError, match="does not match dataset row count"):
        py_eval.evaluate(code)


def test_python_evaluator_security_sandbox():
    df = pd.DataFrame({"x": [1, 2, 3]})
    py_eval = SafePythonEvaluator(df)

    # Disallow imports
    with pytest.raises(ValueError, match="Importing modules.*strictly forbidden"):
        py_eval.evaluate("import os\nresult = df['x']")

    with pytest.raises(ValueError, match="Importing modules.*strictly forbidden"):
        py_eval.evaluate("from math import sqrt\nresult = df['x']")

    # Disallow forbidden names
    with pytest.raises(ValueError, match="forbidden identifier"):
        py_eval.evaluate("result = open('/etc/hosts')")

    with pytest.raises(ValueError, match="forbidden identifier"):
        py_eval.evaluate("result = eval('1 + 1')")

    # Disallow dunder attributes
    with pytest.raises(ValueError, match="dunder"):
        py_eval.evaluate("result = df['x'].__class__")


# ---------------------------------------------------------------------------
# 4. DerivedFeatureEngine Tests (Orchestration & Validation)
# ---------------------------------------------------------------------------


def test_engine_evaluate_valid_formula():
    df = pd.DataFrame({
        "income": [50000.0, 75000.0, 120000.0],
        "debt": [10000.0, 25000.0, 40000.0],
    })
    engine = DerivedFeatureEngine()
    defn = DerivedFeatureDefinition(
        name="dti",
        expression_type=DerivedFeatureType.FORMULA,
        expression="debt / income",
    )

    res, series = engine.evaluate(df, defn)
    assert res.is_valid is True
    assert res.feature_name == "dti"
    assert res.null_count == 0
    assert not res.is_constant
    assert len(res.sample_values) == 3
    assert res.summary_stats["mean"] > 0.0
    assert series is not None


def test_engine_evaluate_constant_detection():
    df = pd.DataFrame({
        "a": [5, 5, 5, 5],
        "b": [5, 5, 5, 5],
    })
    engine = DerivedFeatureEngine()
    defn = DerivedFeatureDefinition(
        name="const_diff",
        expression_type=DerivedFeatureType.FORMULA,
        expression="a - b",
    )

    res, series = engine.evaluate(df, defn)
    assert res.is_valid is True
    assert res.is_constant is True
    assert any("constant" in w.lower() for w in res.warnings)


def test_engine_evaluate_invalid_name():
    df = pd.DataFrame({"x": [1, 2]})
    engine = DerivedFeatureEngine()
    defn = DerivedFeatureDefinition(
        name="123_invalid-name!",
        expression_type=DerivedFeatureType.FORMULA,
        expression="x * 2",
    )

    res, series = engine.evaluate(df, defn)
    assert res.is_valid is False
    assert "Invalid feature name" in res.error_message
    assert series is None


def test_engine_apply_transformation():
    df = pd.DataFrame({
        "tenure": [1.0, 12.0, 24.0],
        "charges": [20.0, 50.0, 70.0],
    })
    engine = DerivedFeatureEngine()
    defn = DerivedFeatureDefinition(
        name="clv_proxy",
        expression_type=DerivedFeatureType.FORMULA,
        expression="tenure * charges",
    )

    out_df, res = engine.apply(df, defn)
    assert "clv_proxy" in out_df.columns
    assert "clv_proxy" not in df.columns  # Original df remains untouched!
    assert out_df["clv_proxy"].tolist() == [20.0, 600.0, 1680.0]

    # Propose FeatureSet
    fs = engine.propose_candidate_feature_set(["tenure", "charges"], ["clv_proxy"], "ds_1")
    assert fs.name == "derived_features"
    assert "clv_proxy" in fs.feature_names


# ---------------------------------------------------------------------------
# 5. FeatureAdvisor Tests (Proposals, LLM & Validation)
# ---------------------------------------------------------------------------


def test_feature_advisor_heuristics():
    advisor = FeatureAdvisor()
    proposals = advisor.propose_features(
        column_names=["tenure", "monthly_charges", "churn"],
        numeric_columns=["tenure", "monthly_charges"],
        target_column="churn",
        domain_hint="customer_churn",
        skewed_columns=["monthly_charges"],
    )
    assert len(proposals) >= 2
    names = [p.name for p in proposals]
    assert any("log1p" in n for n in names)
    assert any("clv" in n or "charge" in n for n in names)

    # Validate against sample dataframe
    df = pd.DataFrame({
        "tenure": [1.0, 10.0, 20.0, 30.0],
        "monthly_charges": [20.0, 50.0, 70.0, 90.0],
        "churn": [1, 0, 0, 0],
    })
    valid_defs, results = advisor.validate_and_filter(df, proposals)
    assert len(valid_defs) >= 2
    for r in results:
        assert r.is_valid is True


def test_feature_advisor_llm_and_fallback():
    class FakeLLMResponse:
        def __init__(self, data: list):
            self.parsed = data
            self.content = json.dumps(data)

    class FakeLLMProvider:
        def __init__(self, succeed: bool = True):
            self.succeed = succeed

        def generate(self, prompt, system_prompt, response_schema):
            if not self.succeed:
                raise RuntimeError("LLM failure")
            return FakeLLMResponse([
                {
                    "name": "custom_ratio",
                    "expression_type": "formula",
                    "expression": "a / (b + 1e-5)",
                    "description": "Custom LLM ratio",
                    "source_columns": ["a", "b"],
                }
            ])

    adv_llm = FeatureAdvisor(llm_provider=FakeLLMProvider(succeed=True))
    props = adv_llm.propose_features(
        column_names=["a", "b", "y"],
        numeric_columns=["a", "b"],
        target_column="y",
    )
    assert len(props) == 1
    assert props[0].name == "custom_ratio"

    # Failing LLM falls back to heuristics
    adv_fail = FeatureAdvisor(llm_provider=FakeLLMProvider(succeed=False))
    props_fallback = adv_fail.propose_features(
        column_names=["tenure", "charges", "churn"],
        numeric_columns=["tenure", "charges"],
        target_column="churn",
        domain_hint="customer_churn",
    )
    assert len(props_fallback) >= 1


# ---------------------------------------------------------------------------
# 6. AutoMLWorkspace Integration Tests
# ---------------------------------------------------------------------------


def test_workspace_derived_feature_workflow(tmp_path: Path):
    ws, _, _ = build_application(root_dir=str(tmp_path / "ws"))

    csv_file = tmp_path / "finance.csv"
    pd.DataFrame({
        "income": [50000.0, 80000.0, 110000.0],
        "debt": [15000.0, 20000.0, 10000.0],
        "default": [0, 0, 1],
    }).to_csv(csv_file, index=False)

    ds = ws.register_dataset(
        name="finance_data",
        path=str(csv_file),
        target="default",
        task_type="binary_classification",
    )

    # 1. Validate derived feature preview
    val_res = ws.validate_derived_feature(
        dataset_id=ds.id,
        name="debt_ratio",
        expression="debt / income",
        expression_type="formula",
    )
    assert val_res.is_valid is True
    assert val_res.feature_name == "debt_ratio"
    assert len(val_res.sample_values) == 3

    # 2. Apply derived feature
    apply_res, profile = ws.apply_derived_feature(
        dataset_id=ds.id,
        name="debt_ratio",
        expression="debt / income",
        expression_type="formula",
        description="Debt to income ratio",
    )
    assert apply_res.is_valid is True
    col_names = [c.name for c in profile.columns]
    assert "debt_ratio" in col_names

    # 3. Suggest derived features
    suggestions = ws.suggest_derived_features(dataset_id=ds.id)
    assert isinstance(suggestions, list)


# ---------------------------------------------------------------------------
# 7. Web API Endpoints Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def running_derived_server(tmp_path: Path):
    workspace_dir = str(tmp_path / "web_derived_ws")
    ws, _, _ = build_application(root_dir=workspace_dir)

    csv_file = tmp_path / "retail.csv"
    pd.DataFrame({
        "sales": [100.0, 250.0, 500.0, 0.0],
        "cost": [50.0, 120.0, 200.0, 0.0],
        "target": [1, 0, 1, 0],
    }).to_csv(csv_file, index=False)

    ds = ws.register_dataset(
        name="retail_sales",
        path=str(csv_file),
        target="target",
        task_type="binary_classification",
    )

    AutoMLWebHandler.workspace_dir = workspace_dir
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    yield {"base_url": base_url, "dataset_id": ds.id}

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_web_api_derived_feature_endpoints(running_derived_server):
    base = running_derived_server["base_url"]
    dataset_id = running_derived_server["dataset_id"]

    # 1. POST /api/features/calculate (Formula with zero-division protection)
    calc_payload = {
        "dataset_id": dataset_id,
        "name": "profit_margin",
        "expression": "(sales - cost) / sales",
        "expression_type": "formula",
    }
    req = Request(
        f"{base}/api/features/calculate",
        data=json.dumps(calc_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        res = data["result"]
        assert res["is_valid"] is True
        assert res["feature_name"] == "profit_margin"
        assert res["zero_division_occurred"] is True  # Row 4 has sales=0.0

    # 2. POST /api/features/calculate with Python code
    py_payload = {
        "dataset_id": dataset_id,
        "name": "cost_squared",
        "expression": "def compute_feature(df):\n    return df['cost'] ** 2",
        "expression_type": "python_code",
    }
    req_py = Request(
        f"{base}/api/features/calculate",
        data=json.dumps(py_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req_py) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["result"]["is_valid"] is True
        assert data["result"]["sample_values"][0] == 2500.0

    # 3. POST /api/features/apply
    apply_payload = {
        "dataset_id": dataset_id,
        "name": "profit_margin",
        "expression": "(sales - cost) / (sales + 1e-6)",
        "expression_type": "formula",
    }
    req_apply = Request(
        f"{base}/api/features/apply",
        data=json.dumps(apply_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req_apply) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        assert data["feature_count"] == 4  # sales, cost, target + profit_margin

    # 4. POST /api/features/suggest
    req_sugg = Request(
        f"{base}/api/features/suggest",
        data=json.dumps({"dataset_id": dataset_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req_sugg) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "success"
        assert isinstance(data["suggestions"], list)


def test_formula_calculator_all_safe_math_functions():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
        "c": [-2.0, 0.0, 2.0],
        "d": [np.nan, 2.0, np.nan],
    })
    calc = SafeFormulaCalculator(df)

    res, _ = calc.evaluate("sin(a) + cos(b) + tan(c)")
    assert len(res) == 3

    res, _ = calc.evaluate("exp(c) + abs(c) + round(a, 1)")
    assert len(res) == 3

    res, _ = calc.evaluate("min(a, b) + max(a, b) + power(a, 2)")
    assert len(res) == 3

    res, _ = calc.evaluate("mean(a) + std(b) + fillna(d, 0.0)")
    assert len(res) == 3

    # Mod and floor div with zero-denom
    res, w = calc.evaluate("b % c")
    assert calc.zero_division_occurred is True

    res, _ = calc.evaluate("b // c")
    assert len(res) == 3

    # Python-style ternary `x if cond else y`
    res, _ = calc.evaluate("a if c > 0 else b")
    assert res.tolist() == [4.0, 5.0, 3.0]

    # Unary ops
    res, _ = calc.evaluate("+a - (-b)")
    assert res.tolist() == [5.0, 7.0, 9.0]


def test_python_evaluator_error_trapping_and_edge_cases():
    df = pd.DataFrame({"x": [10.0, 20.0]})
    py_eval = SafePythonEvaluator(df)

    # Alternative function names: transform
    code_transform = "def transform(df):\n    return df['x'] * 3"
    s, _ = py_eval.evaluate(code_transform)
    assert s.tolist() == [30.0, 60.0]

    # Alternative function names: calculate
    code_calc = "def calculate(df):\n    return df['x'] + 1"
    s, _ = py_eval.evaluate(code_calc)
    assert s.tolist() == [11.0, 21.0]

    # Alternative: single lambda
    code_lambda = "derive = lambda df: df['x'] - 5"
    s, _ = py_eval.evaluate(code_lambda)
    assert s.tolist() == [5.0, 15.0]

    # ZeroDivisionError trapped
    code_zdiv = "def compute_feature(df):\n    raise ZeroDivisionError()"
    s, w = py_eval.evaluate(code_zdiv)
    assert s.tolist() == [0.0, 0.0]
    assert any("zerodivisionerror" in item.lower() for item in w)

    # Runtime error
    code_err = "def compute_feature(df):\n    raise KeyError('col_not_found')"
    with pytest.raises(RuntimeError, match="Runtime error in feature function"):
        py_eval.evaluate(code_err)

    # Empty code
    with pytest.raises(ValueError, match="cannot be empty"):
        py_eval.evaluate("   ")

    # Missing function or result
    with pytest.raises(ValueError, match="must define a callable"):
        py_eval.evaluate("x = 10\ny = 20")


def test_engine_high_null_warning_and_text_series():
    df = pd.DataFrame({
        "s": ["cat", "dog", "bird"],
        "n": [np.nan, np.nan, 5.0],
    })
    engine = DerivedFeatureEngine()

    defn = DerivedFeatureDefinition(
        name="null_feat",
        expression_type=DerivedFeatureType.FORMULA,
        expression="n",
    )
    res, _ = engine.evaluate(df, defn)
    assert res.is_valid is True
    assert any("high null rate" in w.lower() for w in res.warnings)
