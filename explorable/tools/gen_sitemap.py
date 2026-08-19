#!/usr/bin/env python3
"""Generate sitemap.xml from the site tree. Directory index.html files are
listed as trailing-slash URLs; tools/ is excluded. Run from anywhere."""
from pathlib import Path
import datetime

BASE = "https://lattice.project89.org"
ROOT = Path(__file__).resolve().parent.parent
TODAY = datetime.date.today().isoformat()

urls = []
for p in sorted(ROOT.rglob("*.html")):
    rel = p.relative_to(ROOT)
    if rel.parts[0] == "tools":
        continue
    if rel.name == "index.html":
        loc = f"{BASE}/" if len(rel.parts) == 1 else f"{BASE}/{'/'.join(rel.parts[:-1])}/"
        prio = "1.0" if len(rel.parts) == 1 else "0.8"
    else:
        loc = f"{BASE}/{rel}"
        prio = "0.6"
    urls.append((loc, prio))

lines = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
for loc, prio in urls:
    lines += ["  <url>", f"    <loc>{loc}</loc>", f"    <lastmod>{TODAY}</lastmod>",
              "    <changefreq>weekly</changefreq>", f"    <priority>{prio}</priority>", "  </url>"]
lines.append("</urlset>")
(ROOT / "sitemap.xml").write_text("\n".join(lines) + "\n")
print(f"sitemap.xml: {len(urls)} urls")
