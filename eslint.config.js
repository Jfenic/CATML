import globals from "globals";
import js from "@eslint/js";

/** Clean Arch PR5 — gradual ESLint for Workbench static JS. */
export default [
  {
    ignores: [
      "src/automl/interfaces/web/static/vendor/**",
      "website/**",
      "node_modules/**",
      "dist/**",
    ],
  },
  js.configs.recommended,
  {
    files: ["src/automl/interfaces/web/static/js/**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        ...globals.browser,
        Chart: "readonly",
        lucide: "readonly",
      },
    },
    rules: {
      // Gradual: keep noise low while the module graph stabilizes.
      "no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      "no-empty": ["error", { allowEmptyCatch: true }],
      "no-constant-condition": ["error", { checkLoops: false }],
    },
  },
];
