import { readFile } from "node:fs/promises";
const cache = new Map();
export async function moduleUrl(url) {
  if (cache.has(url.href)) return cache.get(url.href);
  let source = await readFile(url, "utf8");
  const imports = [...source.matchAll(/from\s+["'](\.[^"']+)["']/g)];
  for (const match of imports) {
    const dependency = await moduleUrl(new URL(match[1], url));
    source = source.replace(match[0], `from ${JSON.stringify(dependency)}`);
  }
  const data = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
  cache.set(url.href, data);
  return data;
}
export const load = async path => import(await moduleUrl(new URL(`../../src/automl/interfaces/web/static/js/${path}`, import.meta.url)));
