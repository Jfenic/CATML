import { chromium } from '@playwright/test';
import { fileURLToPath } from 'node:url';
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
  await page.goto(process.env.WEBSITE_URL || 'http://127.0.0.1:5174', { waitUntil: 'networkidle' });
  await page.evaluate(() => {
    document.body.innerHTML = `<div style="height:630px;padding:64px 72px;position:relative;background:#fff;color:#0b0d12"><div style="font-size:30px;font-weight:750;letter-spacing:-1px">CATML<span style="color:#4f67ff">.</span></div><div style="margin-top:62px;color:#667085;font-size:13px;letter-spacing:2px">OPEN SOURCE · LOCAL-FIRST · AGENT-NATIVE</div><h1 style="font-size:76px;line-height:1.04;letter-spacing:-4px;margin-top:24px">Agent-native AutoML<br>for tabular data<span style="color:#4f67ff">.</span></h1><p style="font-size:23px;color:#667085;margin-top:28px">Train better models. Keep control.</p><div style="position:absolute;bottom:48px;left:72px;right:72px;border-top:1px solid #e5e7eb;padding-top:24px;display:flex;justify-content:space-between;font-size:14px;color:#667085"><span>Python API &nbsp; / &nbsp; Visual Workbench &nbsp; / &nbsp; AI Agents</span><span>CATML Community · MIT</span></div></div>`;
  });
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: fileURLToPath(new URL('../public/og-image.png', import.meta.url)) });
  console.log('OpenGraph image generated');
} finally { await browser.close(); }
