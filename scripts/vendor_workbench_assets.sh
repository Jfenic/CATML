#!/usr/bin/env bash
# Download pinned Workbench vendor JS/CSS/fonts and build offline Tailwind CSS.
# Run from repository root. Requires network + npm (Node 18+).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATIC="$ROOT/src/automl/interfaces/web/static"
VENDOR="$STATIC/vendor"
FONTS="$VENDOR/fonts"
BUILD="$ROOT/.vendor-build"

CHART_VER="4.4.7"
LUCIDE_VER="0.469.0"
TAILWIND_VER="3.4.17"

mkdir -p "$VENDOR" "$FONTS" "$BUILD"

echo "==> Chart.js v${CHART_VER}"
curl -fsSL "https://cdn.jsdelivr.net/npm/chart.js@${CHART_VER}/dist/chart.umd.min.js" \
  -o "$VENDOR/chart.umd.min.js"

echo "==> Lucide UMD v${LUCIDE_VER}"
curl -fsSL "https://cdn.jsdelivr.net/npm/lucide@${LUCIDE_VER}/dist/umd/lucide.min.js" \
  -o "$VENDOR/lucide.min.js"

echo "==> Fonts (IBM Plex Mono + Geist via npm fontsource/geist)"
cd "$BUILD"
cat > package.json <<EOF
{
  "name": "catml-workbench-vendor-build",
  "private": true,
  "devDependencies": {
    "tailwindcss": "${TAILWIND_VER}",
    "@fontsource/ibm-plex-mono": "5.2.5",
    "geist": "1.3.1"
  }
}
EOF
npm install --silent --no-fund --no-audit

# IBM Plex Mono woff2 (latin 400/500/600/700)
IBM_SRC="$BUILD/node_modules/@fontsource/ibm-plex-mono/files"
for w in ibm-plex-mono-latin-400-normal.woff2 \
         ibm-plex-mono-latin-500-normal.woff2 \
         ibm-plex-mono-latin-600-normal.woff2 \
         ibm-plex-mono-latin-700-normal.woff2; do
  cp "$IBM_SRC/$w" "$FONTS/$w"
done

# Geist / Geist Mono from vercel geist package (woff2 under dist/fonts)
GEIST_DIR="$BUILD/node_modules/geist/dist/fonts"
if [[ -d "$GEIST_DIR/geist-sans" ]]; then
  cp "$GEIST_DIR/geist-sans/Geist-Regular.woff2" "$FONTS/Geist-Regular.woff2" 2>/dev/null || true
  cp "$GEIST_DIR/geist-sans/Geist-Medium.woff2" "$FONTS/Geist-Medium.woff2" 2>/dev/null || true
  cp "$GEIST_DIR/geist-sans/Geist-SemiBold.woff2" "$FONTS/Geist-SemiBold.woff2" 2>/dev/null || true
  cp "$GEIST_DIR/geist-sans/Geist-Bold.woff2" "$FONTS/Geist-Bold.woff2" 2>/dev/null || true
fi
if [[ -d "$GEIST_DIR/geist-mono" ]]; then
  cp "$GEIST_DIR/geist-mono/GeistMono-Regular.woff2" "$FONTS/GeistMono-Regular.woff2" 2>/dev/null || true
  cp "$GEIST_DIR/geist-mono/GeistMono-Medium.woff2" "$FONTS/GeistMono-Medium.woff2" 2>/dev/null || true
  cp "$GEIST_DIR/geist-mono/GeistMono-SemiBold.woff2" "$FONTS/GeistMono-SemiBold.woff2" 2>/dev/null || true
  cp "$GEIST_DIR/geist-mono/GeistMono-Bold.woff2" "$FONTS/GeistMono-Bold.woff2" 2>/dev/null || true
fi

# Fallback naming discovery if vercel package layout differs
find "$BUILD/node_modules/geist" -name '*.woff2' 2>/dev/null | head -40 || true

echo "==> Tailwind CSS v${TAILWIND_VER} (offline build)"
cat > "$BUILD/tailwind.config.js" <<'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "../src/automl/interfaces/web/static/**/*.{html,js}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Geist"', "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ['"Geist Mono"', '"IBM Plex Mono"', "ui-monospace", "Menlo", "monospace"],
      },
      colors: {
        catml: {
          black: "#080A0F",
          bar: "#0D1017",
          panel: "#11151E",
          card: "#161B26",
          border: "#242A36",
          text: "#F7F8FA",
          muted: "#8B95A7",
          blue: "#4F67FF",
          indigo: "#6956E8",
          cyan: "#53C8FF",
          success: "#22C55E",
          warning: "#F59E0B",
          danger: "#EF4444",
        },
      },
    },
  },
  plugins: [],
};
EOF

cat > "$BUILD/input.css" <<'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF

npx --no-install tailwindcss -c "$BUILD/tailwind.config.js" -i "$BUILD/input.css" \
  -o "$STATIC/css/tailwind.built.css" --minify

# Versions manifest for audits / SRI regeneration
cat > "$VENDOR/VERSIONS.txt" <<EOF
chart.js=${CHART_VER}
lucide=${LUCIDE_VER}
tailwindcss=${TAILWIND_VER} (built offline to css/tailwind.built.css)
ibm-plex-mono=@fontsource/ibm-plex-mono@5.2.5
geist=geist@1.3.1 (woff2 copied when present)
generated=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "==> Done. Vendor tree:"
find "$VENDOR" "$STATIC/css/tailwind.built.css" -type f 2>/dev/null | sort
