# CATML public website

A single light homepage built with React, TypeScript and Vite. The site is independent of the Python package and dark Workbench. All runtime fonts and images are served locally; no analytics, form service or backend is required.

## Develop

Use Node.js 22.12+ (Node.js 24 recommended).

```bash
cd website
npm ci
npm run dev
```

The normal development URL is http://127.0.0.1:5173. Compile with `npm run build` and inspect the production bundle with `npm run preview`.

## Browser verification

```bash
npx playwright install --with-deps chromium
WEBSITE_URL=http://127.0.0.1:5173 npm run check
```

The checks cover 320, 390, 768, 1024 and 1440 px, image loading, local anchors, WCAG A/AA via axe, mobile navigation, clipboard, keyboard entry and reduced motion. Screenshots go to `/tmp/catml-landing-checks`; override with `CHECK_OUTPUT`.

## Vercel

Import the repository, select **website** as Root Directory and **Vite** as framework. Build command: `npm run build`. Output: `dist`. The included `vercel.json` adds basic response headers. No environment variables or backend are required. Connect the intended domain in Vercel before adding an absolute canonical URL and absolute `og:url`/`og:image` URLs in `index.html`.

Setup references: [Vite guide](https://vite.dev/guide/) and [Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite).

The project is ready for deployment; creating a production deployment or connecting a domain is a separate operation.

## Content and evidence

- `src/App.tsx` owns copy, links, Python examples and dated foundation statistics.
- MIT reflects the actual root LICENSE. The supplied Apache 2.0 copy was corrected.
- Installation is from the GitHub source checkout, without assuming a `catml` PyPI release.
- The sample leaderboard is labeled **Example output**, not presented as benchmark evidence.
- 461 tests / 87.53% coverage are a historical snapshot from the October 2, 2026 Workbench refinement, linked to commit `e4a2286`. They are not live CI status; update copy only with attributable validation evidence.
- CATML Platform is explicitly planned. Its capabilities describe future intent, not a shipped service. There is no inactive waitlist or pricing page.
- Inference needs a compatible Python environment with CATML and the relevant estimator dependencies, but no original workspace/database.
- The seven-step agent diagram is illustrative. Reasoning is supplied by an external compatible agent; CATML supplies tools, approvals, budgets and cooperative cancellation.

## Real Workbench screenshot

`public/workbench.png` is captured directly from the committed Workbench at `e4a2286`, without redesigning its UI or altering its DOM. A disposable synthetic customer-churn dataset is used; the screenshot is labeled as demo data. No private or competition data is included.

Start the existing Workbench against a disposable populated workspace, then run:

```bash
WORKBENCH_URL=http://127.0.0.1:8092 node scripts/capture-workbench.mjs
```

The script captures Dataset Inspector at 1440 × 1000. Keep source provenance current when replacing the image. Refresh only the image while Workbench development continues on its own branch.

`public/og-image.png` is a code-rendered social card. Regenerate against a running website with:

```bash
WEBSITE_URL=http://127.0.0.1:5173 node scripts/generate-og.mjs
```
