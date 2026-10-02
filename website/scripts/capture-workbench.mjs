import { chromium } from '@playwright/test';
import { fileURLToPath } from 'node:url';

const url = process.env.WORKBENCH_URL || 'http://127.0.0.1:8092';
const output = fileURLToPath(new URL('../public/workbench.png', import.meta.url));
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  page.on('pageerror', error => console.error('Workbench:', error.message));
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.locator('[data-nav="datasets"]').click();
  await page.waitForFunction(() => document.querySelector('#mainContent')?.textContent.includes('tenure_months'));
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: output, animations: 'disabled' });
  console.log(`Captured real Workbench: ${output}`);
} finally { await browser.close(); }
