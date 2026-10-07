import { useEffect, useRef, useState, type ReactNode } from 'react';

const repo = 'https://github.com/Jfenic/CATML';
const docs = `${repo}/blob/main/docs/README.md`;
const quickstart = `from catml import AutoML
import pandas as pd

df = pd.read_csv("customers.csv")
automl = AutoML(task="classification")

result = automl.fit(df, target="churn")
print(result.leaderboard())

test_df = pd.read_csv("customers_test.csv")
preds = result.predict(test_df)
result.best_model.save("model.pkl")`;
const artifact = `result.best_model.save("model.pkl")

from catml import ModelArtifact

model = ModelArtifact.load("model.pkl")
predictions = model.predict(new_data)`;
const installation = `git clone https://github.com/Jfenic/CATML.git
cd CATML
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[mcp]"`;

function Icon({ name, className = '' }: { name: 'arrow' | 'github' | 'code' | 'window' | 'agent' | 'check' | 'copy' | 'database' | 'box' | 'shield' | 'menu' | 'close'; className?: string }) {
  const paths: Record<typeof name, ReactNode> = {
    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
    github: <><path d="M9 19c-4 1-4-2-6-2m12 5v-4c0-1 .1-2-.5-2.5 3-.4 6-1.5 6-6 0-1.3-.5-2.5-1.3-3.4.1-.4.6-1.8-.1-3.6 0 0-1.1-.4-3.6 1.3a12 12 0 0 0-6.6 0C6.4 2.1 5.3 2.5 5.3 2.5c-.7 1.8-.2 3.2-.1 3.6C4.4 7 4 8.2 4 9.5c0 4.5 2.9 5.6 5.9 6-.5.5-.9 1.3-.9 2.5v4" /></>,
    code: <><path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-13-2 16" /></>,
    window: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9h18M9 9v11m-3-14h.01M9 6h.01" /></>,
    agent: <><rect x="8" y="8" width="8" height="8" rx="2" /><path d="M10 3v5m4-5v5m-4 8v5m4-5v5M3 10h5m-5 4h5m8-4h5m-5 4h5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V4H4v12h4" /></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0" /></>,
    box: <><path d="m12 3 9 5v9l-9 5-9-5V8l9-5Zm0 9v10M3 8l9 4 9-4M7 5.8l9 5v5" /></>,
    shield: <><path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z" /><path d="m8 12 3 3 5-6" /></>,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
  };
  return <svg className={`icon ${className}`} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function Logo() {
  return <span className="logo"><svg width="25" height="25" viewBox="0 0 30 30" fill="none" aria-hidden="true"><path d="M21 6H11l-6 9 6 9h10M11 15h14" stroke="currentColor" strokeWidth="2.5" /><circle cx="25" cy="15" r="3" fill="#4F67FF" /></svg>CATML<span className="logo-period">.</span></span>;
}

function ButtonLink({ children, href, secondary = false, github = false }: { children: ReactNode; href: string; secondary?: boolean; github?: boolean }) {
  return <a className={`button ${secondary ? 'button-secondary' : 'button-primary'}`} href={href}>{github && <Icon name="github" />}{children}{!github && <Icon name="arrow" />}</a>;
}

function CodeBlock({ code, title, compact = false }: { code: string; title: string; compact?: boolean }) {
  const [status, setStatus] = useState('Copy code');
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  async function copy() {
    try { await navigator.clipboard.writeText(code); setStatus('Copied'); }
    catch { setStatus('Select code to copy'); }
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setStatus('Copy code'), 2500);
  }
  const parts = code.split(/("[^"\n]*"|\b(?:from|import|as|print)\b)/g);
  return <div className={`code-panel ${compact ? 'code-compact' : ''}`}>
    <div className="code-toolbar"><span className="terminal-dots" aria-hidden="true"><i /><i /><i /></span><span>{title}</span><button onClick={copy} className="copy-button" aria-label={`Copy ${title} code`}><Icon name="copy" /><span aria-live="polite">{status}</span></button></div>
    <pre tabIndex={0} aria-label={`${title} code example`}><code>{parts.map((part, i) => <span key={i} className={part.startsWith('"') ? 'syntax-string' : /^(from|import|as|print)$/.test(part) ? 'syntax-keyword' : undefined}>{part}</span>)}</code></pre>
  </div>;
}

function SectionHeading({ label, title, description, centered = false }: { label: string; title: ReactNode; description?: string; centered?: boolean }) {
  return <div className={`section-heading ${centered ? 'centered' : ''}`}><p className="eyebrow">{label}</p><h2>{title}</h2>{description && <p className="section-description">{description}</p>}</div>;
}

function PipelineVisual() {
  return <div className="pipeline-visual" role="img" aria-label="A dataset flows through CATML to models and a portable artifact. An AI agent connects to CATML through MCP.">
    <div className="diagram-topline"><span>THE CATML ENGINE</span><span className="diagram-status"><i /> LOCAL</span></div>
    <div className="diagram-grid" aria-hidden="true" />
    <div className="agent-node"><Icon name="agent" /><span>AI Agent</span><span className="small-tag">MCP</span></div>
    <div className="agent-wire" />
    <div className="engine-layer layer-back" /><div className="engine-layer layer-middle" />
    <div className="engine-node"><Logo /><span>Train. Compare. Export.</span><div className="engine-bars"><i /><i /><i /><i /><i /></div></div>
    <div className="data-node"><Icon name="database" /><span>Dataset</span><small>YOUR DATA</small></div>
    <div className="artifact-node"><Icon name="box" /><span>model.pkl</span><small>YOUR MODEL</small></div>
    <div className="wire wire-left" /><div className="wire wire-right" />
    <div className="models-node"><span>MODELS</span><span>Compare & evaluate <Icon name="check" /></span></div>
    <div className="diagram-bottomline"><span>01 / DATA → MODEL</span><span>Built to keep you in control</span></div>
  </div>;
}

function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    function escape(event: KeyboardEvent) { if (event.key === 'Escape' && menuOpen) { setMenuOpen(false); menuButton.current?.focus(); } }
    document.addEventListener('keydown', escape);
    return () => document.removeEventListener('keydown', escape);
  }, [menuOpen]);
  return <>
    <a className="skip-link" href="#main">Skip to content</a>
    <header className="site-header"><div className="container navbar">
      <a className="brand-link" href="#" aria-label="CATML home"><Logo /></a>
      <button ref={menuButton} className="menu-toggle" aria-expanded={menuOpen} aria-controls="primary-navigation" aria-label={menuOpen ? 'Close navigation' : 'Open navigation'} onClick={() => setMenuOpen(!menuOpen)}><Icon name={menuOpen ? 'close' : 'menu'} /></button>
      <nav id="primary-navigation" aria-label="Main navigation" className={menuOpen ? 'nav-links is-open' : 'nav-links'} onClick={() => setMenuOpen(false)}>
        <a href="#product">Product</a><a href={docs}>Docs</a><a href={repo}>GitHub</a><a href="#community">Community</a><a href="#platform">Platform <span className="nav-planned">PLANNED</span></a><a className="nav-cta" href="#get-started">Get Started <Icon name="arrow" /></a>
      </nav>
    </div></header>
    <main id="main">
      <section className="container hero" aria-labelledby="hero-title">
        <div className="hero-copy"><p className="hero-badge"><span />OPEN SOURCE <i>·</i> LOCAL-FIRST <i>·</i> AGENT-NATIVE</p>
          <h1 id="hero-title">Agent-native AutoML<br />for tabular data<span className="headline-period">.</span></h1>
          <p className="hero-description">Train, compare and export models locally.<br className="desktop-break" /> Let AI agents operate ML experiments through MCP.</p>
          <div className="button-row"><ButtonLink href="#get-started">Get Started</ButtonLink><ButtonLink href={repo} secondary github>View on GitHub</ButtonLink></div>
          <a className="hero-product-link" href="#workbench"><Icon name="window" /> Take a look inside the Workbench <span>↗</span></a>
        </div><PipelineVisual />
      </section>
      <div className="container principles-strip"><span>Built for your workflow.</span><div><span><Icon name="database" /> Local by default</span><span><Icon name="shield" /> Governed experiments</span><span><Icon name="box" /> Portable artifacts</span></div><a href={`${repo}/blob/main/LICENSE`}>Apache-2.0 licensed <span>↗</span></a></div>

      <section id="product" className="container section">
        <SectionHeading label="YOUR WORKFLOW, YOUR CHOICE" title={<>One engine.<br />Three ways to work.</>} description="Start in Python. Explore visually. Connect an agent. Every interface works with the same experiment engine." />
        <div className="entry-cards">
          <article className="entry-card"><div className="card-top"><Icon name="code" /><span>01 / PYTHON</span></div><h3>Python API</h3><p>A simple high-level API for data scientists and ML engineers.</p><div className="entry-preview python-preview"><span className="syntax-keyword">from</span> catml <span className="syntax-keyword">import</span> AutoML<br /><br /><span className="muted-code"># Your next model starts here.</span><br />result = automl.fit(df, target=<span className="preview-string">"churn"</span>)</div><a className="text-link" href="#quickstart">Explore the API <Icon name="arrow" /></a></article>
          <article className="entry-card"><div className="card-top"><Icon name="window" /><span>02 / WORKBENCH</span></div><h3>Visual Workbench</h3><p>Understand datasets, review recommendations and launch experiments visually.</p><div className="entry-preview image-preview"><img src="/workbench.png" width="1440" height="1000" alt="CATML Workbench displaying a real dataset profile" loading="lazy" /></div><a className="text-link" href="#workbench">See the Workbench <Icon name="arrow" /></a></article>
          <article className="entry-card"><div className="card-top"><Icon name="agent" /><span>03 / AGENTS</span></div><h3>AI Agents</h3><p>Let compatible AI agents inspect, plan and operate CATML through MCP.</p><div className="entry-preview agent-preview"><span><Icon name="agent" /> Agent</span><span className="mcp-connection">MCP</span><span className="mini-catml">CATML</span></div><a className="text-link" href="#agents">Meet the agent workflow <Icon name="arrow" /></a></article>
        </div>
      </section>

      <section id="quickstart" className="container section quickstart-section">
        <SectionHeading label="LESS SETUP. MORE EXPERIMENTS." title={<>From dataset to model<br />in a few lines.</>} description="Fit candidate models, inspect the leaderboard and save the winner. A familiar API, with the experiment history behind it." />
        <div className="quickstart-grid"><CodeBlock code={quickstart} title="train.py" /><div className="result-panel"><div className="result-top"><span className="eyebrow">EXPERIMENT RESULTS</span><span className="example-badge">Example output</span></div><h3>Let the results speak.</h3><table><caption className="sr-only">Illustrative leaderboard, not benchmark results</caption><thead><tr><th scope="col">Model</th><th scope="col">ROC-AUC</th></tr></thead><tbody>{[['LightGBM', '0.914'], ['XGBoost', '0.907'], ['CatBoost', '0.901']].map(([model, score], i) => <tr key={model} className={i === 0 ? 'winning-row' : ''}><td>{model}{i === 0 && <span className="best-badge">BEST</span>}</td><td>{score}</td></tr>)}</tbody></table><div className="artifact-result"><span className="artifact-icon"><Icon name="box" /></span><div><small>MODEL ARTIFACT</small><strong>model.pkl</strong></div><span className="ready-label"><Icon name="check" /> Portable</span></div><p className="result-note">Illustrative scores. Your results depend on your data, configuration and installed model backends.</p></div></div>
      </section>

      <section className="container section artifact-section" id="artifacts"><div><SectionHeading label="THE MODEL IS YOURS" title={<>Train here.<br />Run anywhere.</>} description="Save the winning model as a standalone artifact. Load it in a compatible Python environment, without the original workspace." /><ul className="check-list"><li><Icon name="check" /> Full preprocessing pipeline included</li><li><Icon name="check" /> Original target labels preserved</li><li><Icon name="check" /> No CATML workspace required for inference</li></ul><a className="text-link" href={`${repo}/blob/main/src/automl/artifacts/model_artifact.py`}>Inside the model artifact <Icon name="arrow" /></a></div><CodeBlock code={artifact} title="predict.py" compact /></section>

      <section id="workbench" className="workbench-section"><div className="container"><SectionHeading centered label="SEE WHAT YOUR DATA IS SAYING" title={<>Understand your data<br />before you train.</>} description="Profile datasets, inspect statistical recommendations, select features and launch experiments from CATML Workbench." /><figure className="workbench-figure"><div className="screenshot-chrome"><span className="terminal-dots" aria-hidden="true"><i /><i /><i /></span><span><Icon name="shield" /> localhost · CATML Workbench</span><span className="chrome-label">LOCAL WORKSPACE</span></div><a href="/workbench.png" aria-label="Open the full CATML Workbench screenshot"><img src="/workbench.png" width="1440" height="1000" alt="Actual CATML Workbench Dataset Inspector showing statistical profiling, feature recommendations and selected columns for a synthetic churn dataset" loading="lazy" /></a><figcaption>Actual CATML Workbench · Synthetic demo dataset · Captured from the committed product, not a mockup.</figcaption></figure><div className="workbench-details"><span><Icon name="database" /> Dataset profiling</span><span><Icon name="code" /> Feature selection</span><span><Icon name="window" /> Experiment management</span><a className="text-link" href={`${repo}#interactive-web-workbench-visual-ml-lab`}>Launch locally <Icon name="arrow" /></a></div></div></section>

      <section id="agents" className="container section agents-section"><div className="split-heading"><SectionHeading label="AGENT-NATIVE, BY DESIGN" title={<>Built for humans.<br />Built for agents.</>} /><p>CATML exposes a governed workflow for compatible agents to inspect, plan, execute and evaluate ML experiments under explicit controls.</p></div><ol className="agent-workflow" aria-label="Illustrative agent workflow">{['Prompt', 'Explore', 'Read', 'Reason', 'Modify', 'Test', 'Evaluate'].map((step, i) => <li key={step}><span className="step-number">0{i + 1}</span><span>{step}</span>{i < 6 && <Icon name="arrow" />}</li>)}</ol><div className="agent-footnote"><p>An external agent supplies the reasoning. CATML supplies the tools and controls.</p><span>PROPOSE ≠ ACCEPT</span></div><div className="governance-grid">{[['MCP', 'Connect through stdio or streamable HTTP.'], ['Approvals', 'Gate governed mutations with explicit approval.'], ['Budgets', 'Bound operations with persistent reservations.'], ['Cancellation', 'Stop cooperatively between execution steps.']].map(([title, text]) => <div key={title}><Icon name="check" /><h3>{title}</h3><p>{text}</p></div>)}</div><a className="text-link" href={`${repo}/blob/main/docs/features/agentic-system/README.md`}>Read the agent integration guide <Icon name="arrow" /></a></section>

      <section id="local-first" className="local-section"><div className="container local-grid"><div><SectionHeading label="LOCAL-FIRST. ALWAYS YOUR CALL." title={<>Your data.<br />Your machine.<br />Your control.</>} description="CATML is designed to run locally. Your datasets and experiments do not need to leave your environment." /><p className="local-note">Connecting an external AI agent is optional. What that agent shares with its provider depends on your integration.</p></div><div className="local-panel"><div className="local-panel-title"><Icon name="shield" /><span>YOUR ENVIRONMENT</span><span className="local-dot" /></div><div className="local-flow"><span><Icon name="database" />Data</span><Icon name="arrow" /><span className="local-engine">CATML</span><Icon name="arrow" /><span><Icon name="box" />Local models</span></div><div className="local-features">{['Local execution', 'SQLite persistence', 'Local workers', 'Portable model artifacts'].map(text => <span key={text}><Icon name="check" />{text}</span>)}</div><div className="local-panel-footer"><span>No cloud account required.</span><span>localhost</span></div></div></div></section>

      <section id="community" className="container section community-section"><SectionHeading centered label="OPEN SOURCE AT THE CORE" title={<>Open locally.<br />Scale when you need it.</>} description="A complete local toolkit today. A commercial platform on the roadmap." /><div className="edition-grid"><article className="edition-card"><div className="edition-top"><Icon name="code" /><span className="edition-badge">AVAILABLE NOW · OPEN SOURCE</span></div><h3>CATML Community</h3><p>Your local AutoML toolkit.<br />Free and open source under the Apache-2.0 License.</p><ul className="check-list">{['Core AutoML · Python API · CLI', 'Visual Workbench', 'Local execution & persistent workers', 'MCP integration & local governance', 'Portable model artifacts', 'Experiment management'].map(text => <li key={text}><Icon name="check" />{text}</li>)}</ul><ButtonLink href="#get-started">Get CATML</ButtonLink></article><article id="platform" className="edition-card platform-card"><div className="edition-top"><Icon name="window" /><span className="edition-badge planned-badge">PLANNED · COMMERCIAL</span></div><h3>CATML Platform</h3><p>A future path for shared infrastructure.<br />Planned capabilities, not available today.</p><ul className="check-list planned-list">{['Managed compute & cloud execution', 'Teams & collaboration', 'Managed deployments', 'Monitoring', 'Central governance', 'RBAC / SSO & enterprise capabilities'].map(text => <li key={text}><span className="planned-mark">+</span>{text}</li>)}</ul><ButtonLink href={`${repo}/blob/main/TASKS.md`} secondary>Explore the roadmap</ButtonLink></article></div></section>

      <section className="container foundation-section"><div><p className="eyebrow">BUILT ON A TESTED FOUNDATION</p><a href={`${repo}/actions`}>Inspect the source. See the checks. <span>↗</span></a></div><div className="foundation-stats"><div><strong>461</strong><span>tests passing¹</span></div><div><strong>87.53<span>%</span></strong><span>code coverage¹</span></div><div><strong>Apache-2.0</strong><span>open-source license</span></div><div><strong>Local</strong><span>execution by default</span></div></div><p className="foundation-source">¹ Recorded validation on October 2, 2026, for the Workbench refinement. <a href={`${repo}/blob/e4a2286/TASKS.md`}>View the recorded result</a>. Counts are a dated snapshot, not live CI status.</p><div className="technology-list"><span>scikit-learn</span><span>LightGBM</span><span>XGBoost</span><span>CatBoost</span><span>Optuna</span><span>SQLite</span><span>MCP</span></div></section>

      <section id="get-started" className="get-started-section"><div className="container get-started-grid"><div><p className="eyebrow">START WITH YOUR NEXT DATASET</p><h2>Build your first<br />CATML model.</h2><p>Open-source AutoML for tabular data.<br />From Python, Workbench or AI agents.</p><div className="button-row"><ButtonLink href={`${repo}#installation`}>Installation guide</ButtonLink><ButtonLink href={repo} secondary github>View on GitHub</ButtonLink></div></div><div><CodeBlock code={installation} title="Install from source" compact /><p className="install-note">Python 3.10+ · macOS / Linux shell · <a href={`${repo}/blob/main/examples/quickstart.py`}>Run the quickstart example ↗</a></p></div></div></section>
    </main>
    <footer className="container site-footer"><div className="footer-brand"><a href="#" aria-label="CATML home"><Logo /></a><p>Train better models.<br />Keep control.</p></div><div className="footer-column"><h2>Product</h2><a href="#workbench">Workbench</a><a href="#agents">Agents</a><a href="#community">Community</a><a href="#platform">Platform <small>Planned</small></a></div><div className="footer-column"><h2>Developers</h2><a href={docs}>Documentation</a><a href={repo}>GitHub</a><a href={`${repo}/tree/main/examples`}>Examples</a></div><div className="footer-column"><h2>Project</h2><a href={`${repo}/blob/main/TASKS.md`}>Roadmap</a><a href={`${repo}/blob/main/CONTRIBUTING.md`}>Contributing</a><a href={`${repo}/blob/main/LICENSE`}>License</a></div><div className="footer-bottom"><span>© {new Date().getFullYear()} CATML Contributors · Apache-2.0 License</span><span>Local-first. Agent-native. Open-source.</span></div></footer>
  </>;
}
export default App;
