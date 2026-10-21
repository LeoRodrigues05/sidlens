#!/usr/bin/env python
"""Build the review page for every mechanistic experiment: AR, diffusion, SID maps, and a storyline tab.

    python docs/results/mechanisms_review/build_report.py [--out <file>]

Writes `sidlens_mechanisms_review.html` beside this file: one self-contained page
(no network) with five tabs (storyline, final important experiments, AR, diffusion, SID maps). Each experiment shows what was done, its results with
plots, and its takeaways. Figures are either the experiment's own PNG figures
(embedded) or SVG charts drawn here from its summary CSVs. Every input is read from
`$SIDLENS_WORK/derived` (`sidlens.paths.DERIVED`) or from this repository; the page
lists the source of each experiment.

The text is in `content_*.py` beside this file. Wording follows `docs/paper/WRITING.md`.
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from sidlens import paths                                                # noqa: E402
from sidlens.viz import diffusion_report as DR                           # noqa: E402

DERIVED = paths.DERIVED

PAGES = (("story", "Storyline and recommendations"), ("final", "Final important experiments"), ("ar", "AR experiments"),
         ("dm", "Diffusion experiments"), ("maps", "SID maps (both models)"))

PROVISIONAL = """<div class="banner"><strong>Provisional results.</strong> Every number on this page comes from the
checkpoints we have now: two AR next-item models, one AR next-two model, 12 retrained AR models and 20 DiffGRM models.
The paper's final numbers will come from rerunning the full analysis on the intermediate checkpoints of the final
training run (with our co-author). We expect the trends to hold, but we have not checked that yet, so treat the values
here as a preview of the write-up, not as its final results.</div>"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def png(path: Path, caption: str, width: str = "100%") -> str:
    """An experiment's own PNG figure, embedded. Refuses a missing file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"figure {path} is missing")
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure class="png"><figcaption>{esc(caption)}</figcaption>'
            f'<img src="data:image/png;base64,{data}" alt="{esc(caption)}" style="max-width:{width}" loading="lazy"></figure>')


def figs(*items: str) -> str:
    return '<div class="figs">' + "".join(items) + "</div>"


def numbers(rows: list[tuple[str, str]]) -> str:
    """A short two-column table of key numbers."""
    body = "".join(f"<tr><th>{esc(a)}</th><td>{b}</td></tr>" for a, b in rows)
    return f'<div class="table-wrap"><table class="kv"><tbody>{body}</tbody></table></div>'


def experiment(e: dict) -> str:
    meta = " · ".join(esc(x) for x in (e.get("models"), e.get("data"), e.get("status")) if x)
    results = e["results"]() if callable(e["results"]) else e["results"]
    take = "".join(f"<li>{t}</li>" for t in e["takeaways"])
    src = f'<p class="src">Source: <code>{esc(e["source"])}</code></p>' if e.get("source") else ""
    return (f'<section class="exp" id="{e["id"]}"><header><span class="code">{esc(e["code"])}</span>'
            f'<h2>{esc(e["title"])}</h2><p class="meta">{meta}</p></header>'
            f'<h3>What we did</h3>{e["did"]}'
            f'<h3>Results</h3>{results}'
            f'<h3>Takeaways</h3><ul class="take">{take}</ul>{src}</section>')


CSS_EXTRA = """
.tabs{position:sticky;top:0;z-index:5;background:var(--page);border-bottom:1px solid var(--border);display:flex;gap:4px;
flex-wrap:wrap;padding:10px 16px 0;max-width:1240px;margin:0 auto}
.tab{background:none;border:1px solid transparent;border-bottom:none;border-radius:8px 8px 0 0;padding:8px 14px;
font:inherit;font-size:14px;color:var(--ink-2);cursor:pointer}
.tab[aria-selected="true"]{background:var(--surface);border-color:var(--border);color:var(--ink);font-weight:600}
.page[hidden]{display:none}
.banner{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--q-rqvae);border-radius:8px;
padding:10px 14px;margin:14px 0 18px;font-size:14px}
section.exp{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 22px;margin:22px 0}
section.exp header h2{margin:2px 0 4px;padding:0;border:none;font-size:21px}
section.exp .code{font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:.04em}
section.exp .meta{color:var(--ink-2);font-size:13px;margin:0}
section.exp h3{font-size:15px;margin:18px 0 6px;color:var(--ink)}
ul.take li{margin:5px 0}
p.src{font-size:12px;color:var(--muted);margin-top:12px}
figure.png{background:#fff;border-radius:10px;padding:10px 12px;grid-column:1/-1}
figure.png figcaption{color:#0b0b0b}
figure.png img{width:100%;height:auto;display:block}
table.kv th{width:46%;font-weight:500;color:var(--ink-2)}
.claim{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin:18px 0}
.claim h3{margin:0 0 6px;font-size:17px}
.claim .rq{font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:.04em}
.claim dl{display:grid;grid-template-columns:150px 1fr;gap:6px 14px;margin:10px 0 0;font-size:14px}
.claim dt{color:var(--ink-2);font-weight:600}
.claim dd{margin:0}
.pill{display:inline-block;font-size:12px;border:1px solid var(--border);border-radius:999px;padding:1px 8px;margin:1px 3px 1px 0;
text-decoration:none;color:var(--ink)}
.pill:hover{border-color:var(--accent)}
ol.order li{margin:6px 0}
h1.pagetitle{font-size:26px;margin:4px 0 8px}
.role{display:inline-block;font-size:12px;font-weight:600;border-radius:999px;padding:1px 9px;margin-left:6px;vertical-align:1px;
border:1px solid var(--border);color:var(--ink-2)}
.role-main{background:var(--accent);border-color:var(--accent);color:#fff}
.role-shared{border-style:dashed}
section.fx{padding:14px 20px}
dl.fxd{display:grid;grid-template-columns:130px minmax(0,1fr);gap:6px 14px;margin:8px 0 0;font-size:14px}
dl.fxd dt{color:var(--ink-2);font-weight:600}
dl.fxd dd{margin:0;max-width:86ch}
dl.terms{grid-template-columns:190px minmax(0,1fr)}
table.ov{border-collapse:collapse;font-size:13.5px;width:100%}
table.ov th,table.ov td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid var(--border)}
table.ov th{color:var(--ink-2);font-weight:600}
table.ov td:first-child{white-space:nowrap}
table.cov td:nth-child(1){width:34%;white-space:normal} table.cov td:nth-child(2){width:36%}
@media (max-width:860px){.claim dl,dl.fxd,dl.terms{grid-template-columns:1fr}}
"""

JS_TABS = """
(function(){
var pages=['story','final','ar','dm','maps'];
function pageOf(h){h=(h||'').replace('#','');var p=h.split('-')[0];return pages.indexOf(p)>=0?p:'story';}
function show(h){var p=pageOf(h);document.querySelectorAll('.page').forEach(function(el){el.hidden=(el.id!=='page-'+p);});
document.querySelectorAll('.tab').forEach(function(t){t.setAttribute('aria-selected',t.dataset.page===p?'true':'false');});
var id=(h||'').replace('#','');var el=id&&document.getElementById(id);if(el&&id.indexOf('-')>0){el.scrollIntoView();}else{window.scrollTo(0,0);}}
document.querySelectorAll('.tab').forEach(function(t){t.addEventListener('click',function(){location.hash=t.dataset.page;});});
window.addEventListener('hashchange',function(){show(location.hash);});
show(location.hash);
})();
"""


def page(page_id: str, intro: str, entries: list[dict]) -> str:
    toc = "".join(f'<a href="#{e["id"]}">{esc(e["title"])}</a>' for e in entries)
    body = intro + "".join(experiment(e) for e in entries)
    return (f'<div class="page" id="page-{page_id}"><div class="layout"><nav class="toc"><strong>On this page</strong>{toc}</nav>'
            f'<main>{PROVISIONAL}{body}</main></div></div>')


def build() -> str:
    import content_ar
    import content_diffusion
    import content_final
    import content_maps
    import content_story

    ar = content_ar.experiments()
    dm = content_diffusion.experiments()
    mp = content_maps.experiments()
    index = {e["id"]: e for e in ar + dm + mp}
    story = content_story.page(index)
    tabs = "".join(f'<button class="tab" type="button" data-page="{p}" aria-selected="false">{esc(t)}</button>' for p, t in PAGES)
    pages = (f'<div class="page" id="page-story"><div class="layout"><nav class="toc"><strong>On this page</strong>'
             f'{content_story.TOC}</nav><main>{PROVISIONAL}{story}</main></div></div>'
             f'<div class="page" id="page-final"><div class="layout"><nav class="toc"><strong>On this page</strong>'
             f'{content_final.TOC}</nav><main>{PROVISIONAL}{content_final.page(index)}</main></div></div>'
             + page("ar", content_ar.INTRO, ar) + page("dm", content_diffusion.INTRO, dm) + page("maps", content_maps.INTRO, mp))
    head = (f'<header class="top"><h1>SidLens: mechanistic analysis of SID recommenders</h1>'
            f'<p class="meta">Qwen-AR and DiffGRM on Amazon Industrial &amp; Scientific (3,105 items) · review copy built '
            f'{date.today().isoformat()} · {len(ar)} AR, {len(dm)} diffusion and {len(mp)} SID-map experiments</p></header>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>SidLens Mechanisms Review</title><style>{DR.CSS}{CSS_EXTRA}'
            f'header.top{{max-width:1240px;margin:0 auto;padding:22px 16px 4px}} header.top h1{{font-size:28px;margin:0 0 4px}}</style></head><body>'
            f'<button class="theme-toggle" type="button">Light / dark</button>{head}<nav class="tabs" aria-label="Pages">{tabs}</nav>'
            f'{pages}<div id="tip" role="tooltip"></div><script>{DR.JS}{JS_TABS}</script></body></html>')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "sidlens_mechanisms_review.html")
    args = ap.parse_args(argv)
    doc = build()
    args.out.write_text(doc)
    print(f"wrote {args.out} ({len(doc) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
