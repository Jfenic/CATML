import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const baseURL = process.env.WEBSITE_URL || 'http://127.0.0.1:5174';
const outputDir = process.env.CHECK_OUTPUT || '/tmp/catml-landing-checks';
await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch();
try {
  const context = await browser.newContext({ permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const width of [1440, 1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto(baseURL, { waitUntil: 'networkidle' });
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator('h1').count(), 1);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(overflow, false, `Horizontal overflow at ${width}px`);
    const imageErrors = await page.locator('img').evaluateAll(images => images.filter(image => !image.complete || image.naturalWidth === 0).map(image => image.src));
    assert.deepEqual(imageErrors, [], 'Images must load');
    const brokenAnchors = await page.locator('a[href^="#"]').evaluateAll(links => links.map(link => link.getAttribute('href')).filter(href => href !== '#' && !document.getElementById(href.slice(1))));
    assert.deepEqual(brokenAnchors, []);
    if (width === 1440) await page.screenshot({ path: `${outputDir}/landing-hero.png`, animations: 'disabled' });
    if (width === 1440 || width === 390) await page.screenshot({ path: `${outputDir}/landing-${width}.png`, fullPage: true, animations: 'disabled' });
    const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
    assert.deepEqual(results.violations.map(({ id, nodes }) => ({ id, targets: nodes.map(node => node.target) })), [], `Accessibility at ${width}px`);
    console.log(`PASS ${width}px: layout, images, anchors, WCAG A/AA`);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  const menu = page.getByRole('button', { name: 'Open navigation' });
  await menu.click();
  assert.equal(await page.getByRole('button', { name: 'Close navigation' }).getAttribute('aria-expanded'), 'true');
  await page.keyboard.press('Escape');
  assert.equal(await menu.getAttribute('aria-expanded'), 'false');
  await menu.click();
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('link', { name: 'Product', exact: true }).click();
  assert.equal(await menu.getAttribute('aria-expanded'), 'false');
  await page.getByRole('button', { name: 'Copy train.py code', exact: true }).click();
  assert.ok((await page.evaluate(() => navigator.clipboard.readText())).includes('from catml import AutoML'));
  const keyboardPage = await context.newPage();
  await keyboardPage.goto(baseURL, { waitUntil: 'networkidle' });
  await keyboardPage.keyboard.press('Tab');
  assert.equal(await keyboardPage.evaluate(() => document.activeElement.textContent), 'Skip to content');
  await keyboardPage.close();
  await page.emulateMedia({ reducedMotion: 'reduce' });
  assert.equal(await page.locator('html').evaluate(el => getComputedStyle(el).scrollBehavior), 'auto');
  const animations = await page.locator('.hero-copy').evaluate(el => getComputedStyle(el).animationName);
  assert.equal(animations, 'none');
  assert.deepEqual(errors, [], 'No browser exceptions');
  console.log('PASS mobile menu, Escape, anchor navigation, clipboard, keyboard entry and reduced motion');
} finally { await browser.close(); }
