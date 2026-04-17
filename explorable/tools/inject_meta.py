#!/usr/bin/env python3
"""
Inject OG + Twitter + favicon + analytics meta tags into every HTML page.

Idempotent: re-running replaces the previous block rather than stacking.

Edit BASE_URL below to match the deployment target before deploying.
"""
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these before deploying
# ---------------------------------------------------------------------------
BASE_URL    = "https://lattice.project89.org"
SITE_NAME   = "The Living Lattice"
SITE_AUTHOR = "Michael Sharpe"
TWITTER     = ""   # e.g. "@your_handle" — leave blank to omit

# ---------------------------------------------------------------------------
# Analytics — PostHog
# ---------------------------------------------------------------------------
# Paste your PostHog project API key into POSTHOG_KEY below (it's a public
# client-side key — safe to commit). Set POSTHOG_HOST to your PostHog region
# (US: https://us.i.posthog.com  ·  EU: https://eu.i.posthog.com  ·  self-host
# your own URL). Leave POSTHOG_KEY empty to skip injecting analytics.
POSTHOG_KEY  = ""
POSTHOG_HOST = "https://us.i.posthog.com"

def _posthog_snippet(key: str, host: str) -> str:
    if not key:
        return ""
    return (
        "<!-- PostHog -->\n"
        "<script>\n"
        "!function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){"
        "function g(t,e){var o=e.split(\".\");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){"
        "t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement(\"script\"))"
        ".type=\"text/javascript\",p.async=!0,p.src=s.api_host+\"/static/array.js\","
        "(r=t.getElementsByTagName(\"script\")[0]).parentNode.insertBefore(p,r);var u=e;"
        "for(void 0!==a?u=e[a]=[]:a=\"posthog\",u.people=u.people||[],u.toString=function(t){"
        "var e=\"posthog\";return\"posthog\"!==a&&(e+=\".\"+a),t||(e+=\" (stub)\"),e},"
        "u.people.toString=function(){return u.toString(1)+\".people (stub)\"},"
        "o=\"init capture identify alias people.set people.set_once set_config register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing reset isFeatureEnabled onFeatureFlags getFeatureFlag getFeatureFlagPayload reloadFeatureFlags group updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures getActiveMatchingSurveys getSurveys onSessionId\".split(\" \"),"
        "n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);\n"
        f"posthog.init('{key}', {{api_host: '{host}', person_profiles: 'identified_only'}});\n"
        "</script>"
    )

ANALYTICS_SNIPPET = _posthog_snippet(POSTHOG_KEY, POSTHOG_HOST)

# ---------------------------------------------------------------------------
# Per-page metadata
# ---------------------------------------------------------------------------
DEFAULT_DESCRIPTION = (
    "The fine structure constant derived from first principles on the diamond lattice. "
    "An interactive essay with zero free parameters."
)

PAGES = {
    "index.html": {
        "title": "The Living Lattice — α from first principles",
        "desc":  DEFAULT_DESCRIPTION,
        "path":  "/",
    },
    "sections/01-prelude.html": {
        "title": "Prelude — The Living Lattice",
        "desc":  "One oscillator to two to coupled to living K. The intuition for the Coherence Learning Rule — where the essay begins.",
        "path":  "/sections/01-prelude.html",
    },
    "sections/02-oscillators.html": {
        "title": "Oscillators on a graph — The Living Lattice",
        "desc":  "Phases, couplings, and the alignment integral. The von Mises distribution and R₀(K).",
        "path":  "/sections/02-oscillators.html",
    },
    "sections/03-coherence.html": {
        "title": "Coherence capital — The Living Lattice",
        "desc":  "C = I_phase · ρ. Neither a seizure nor silence — the universe's compromise between synchrony and richness.",
        "path":  "/sections/03-coherence.html",
    },
    "sections/04-clr.html": {
        "title": "The Coherence Learning Rule — The Living Lattice",
        "desc":  "The full CLR, the potential landscape, the death threshold, and the Coherence Theorem.",
        "path":  "/sections/04-clr.html",
    },
    "sections/05-binary-field.html": {
        "title": "How a binary field emerges — The Living Lattice",
        "desc":  "Alive and dead bonds crystallise from CLR dynamics. The K-field as a learned lattice connectivity.",
        "path":  "/sections/05-binary-field.html",
    },
    "sections/06-plm-npd.html": {
        "title": "Phase-locked modes — The Living Lattice",
        "desc":  "Phase-locked modes as the atoms of form. Geometry and memory in the K-field.",
        "path":  "/sections/06-plm-npd.html",
    },
    "sections/07-vortices.html": {
        "title": "Spontaneous vortices — The Living Lattice",
        "desc":  "Topological defects nucleate from random phases. Winding as a conserved charge; the electron begins here.",
        "path":  "/sections/07-vortices.html",
    },
    "sections/08-diamond.html": {
        "title": "Why the diamond lattice — The Living Lattice",
        "desc":  "Five structural filters uniquely select diamond among all 3D lattices. z = 4, bipartite, tetrahedral.",
        "path":  "/sections/08-diamond.html",
    },
    "sections/09-electron.html": {
        "title": "The electron — The Living Lattice",
        "desc":  "A vortex with integer charge, directed current, and chirality. The electron as topology.",
        "path":  "/sections/09-electron.html",
    },
    "sections/10-fermion.html": {
        "title": "How the lattice makes it a fermion — The Living Lattice",
        "desc":  "Dirac dispersion, spin-½, and the vacuum's response. g = 2 drops out of bipartite symmetry.",
        "path":  "/sections/10-fermion.html",
    },
    "sections/11-bkt-wall.html": {
        "title": "The BKT wall — The Living Lattice",
        "desc":  "Where the CLR's push meets a topological ceiling. K = 2/π, and where α operationally lives.",
        "path":  "/sections/11-bkt-wall.html",
    },
    "sections/12-living-vs-static.html": {
        "title": "Living versus static — The Living Lattice",
        "desc":  "How 1/α goes from 143 to 137 through a single choice about where to evaluate the Debye-Waller factor.",
        "path":  "/sections/12-living-vs-static.html",
    },
    "sections/13-alpha-formula.html": {
        "title": "The α formula, piece by piece — The Living Lattice",
        "desc":  "α = R₀(2/π)⁴ × (π/4)^(1/√e + α/2π). Every factor clickable; every factor on the lattice.",
        "path":  "/sections/13-alpha-formula.html",
    },
    "sections/14-dimension.html": {
        "title": "Why three dimensions — The Living Lattice",
        "desc":  "Only d = 3 produces a physical α. Watch atoms collapse or drift at other dimensions.",
        "path":  "/sections/14-dimension.html",
    },
    "sections/15-lce.html": {
        "title": "Closing the gap with linked clusters — The Living Lattice",
        "desc":  "A linked-cluster expansion over diamond subgraphs closes the gap to 1.5 parts per billion.",
        "path":  "/sections/15-lce.html",
    },
    "sections/16-g-factor.html": {
        "title": "From α to g — The Living Lattice",
        "desc":  "Plug the lattice α into the standard QED series. Eleven matching digits with the most precise measurement in physics.",
        "path":  "/sections/16-g-factor.html",
    },
    "sections/17-coda.html": {
        "title": "Coda: what just happened — The Living Lattice",
        "desc":  "The arrow of intelligence, entropy-coherence duality, and an invitation beyond physics.",
        "path":  "/sections/17-coda.html",
    },
}

# Start / end markers so the block is replaceable on re-run.
START_MARK = "<!-- BEGIN AUTO-INJECTED META -->"
END_MARK   = "<!-- END AUTO-INJECTED META -->"
STYLESHEET_TAG = '<link rel="stylesheet" href="'  # we always inject BEFORE this

def rel_root(html_rel_path: str) -> str:
    """Return the relative path to the site root from this HTML file."""
    parts = Path(html_rel_path).parts
    if len(parts) <= 1:
        return "./"
    return "../" * (len(parts) - 1)

def build_block(page_rel: str, meta: dict) -> str:
    url   = BASE_URL + meta["path"]
    desc  = meta["desc"]
    title = meta["title"]
    root  = rel_root(page_rel)
    og_image = BASE_URL + "/og-image.png"
    lines = [
        START_MARK,
        f'<meta name="description" content="{desc}">',
        f'<meta name="author" content="{SITE_AUTHOR}">',
        f'<link rel="canonical" href="{url}">',
        # Favicons
        f'<link rel="icon" type="image/svg+xml" href="{root}favicon.svg">',
        f'<link rel="apple-touch-icon" href="{root}apple-touch-icon.png">',
        # Open Graph
        f'<meta property="og:title" content="{title}">',
        f'<meta property="og:description" content="{desc}">',
        f'<meta property="og:url" content="{url}">',
        f'<meta property="og:site_name" content="{SITE_NAME}">',
        '<meta property="og:type" content="article">',
        f'<meta property="og:image" content="{og_image}">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        '<meta property="og:image:alt" content="The Living Lattice — α from first principles">',
        # Twitter Card
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{title}">',
        f'<meta name="twitter:description" content="{desc}">',
        f'<meta name="twitter:image" content="{og_image}">',
    ]
    if TWITTER:
        lines.append(f'<meta name="twitter:site" content="{TWITTER}">')
    if ANALYTICS_SNIPPET.strip():
        lines.append(ANALYTICS_SNIPPET.strip())
    lines.append(END_MARK)
    return "\n".join(lines)

def inject(html_path: Path, rel_path: str, block: str) -> bool:
    txt = html_path.read_text()
    original = txt
    # 1. If a previous auto-injected block exists, replace it.
    pattern = re.compile(
        re.escape(START_MARK) + r".*?" + re.escape(END_MARK),
        re.DOTALL,
    )
    if pattern.search(txt):
        txt = pattern.sub(block, txt)
    else:
        # 2. Otherwise insert BEFORE the first <link rel="stylesheet">, or
        #    before </head> as a fallback.
        link_idx = txt.find(STYLESHEET_TAG)
        head_idx = txt.find("</head>")
        target = link_idx if link_idx != -1 else head_idx
        if target == -1:
            print(f"  SKIP {rel_path}: no <link rel=\"stylesheet\"> or </head>")
            return False
        # Find the start of the line containing `target` so indentation is preserved.
        line_start = txt.rfind("\n", 0, target) + 1
        indent = ""  # heads are usually at column 0
        txt = txt[:line_start] + block + "\n" + txt[line_start:]
    if txt != original:
        html_path.write_text(txt)
        return True
    return False

def main():
    root = Path(__file__).resolve().parent.parent
    for rel, meta in PAGES.items():
        p = root / rel
        if not p.exists():
            print(f"  MISS {rel}")
            continue
        block = build_block(rel, meta)
        changed = inject(p, rel, block)
        print(f"  {'OK  ' if changed else 'skip'} {rel}")

if __name__ == "__main__":
    main()
