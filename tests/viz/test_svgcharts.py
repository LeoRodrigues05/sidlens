"""Inline-SVG report charts: well-formed, escaped, and zero kept in view (no data needed)."""

import xml.etree.ElementTree as ET

from sidlens.viz import svgcharts as SC


def _svg(html: str) -> ET.Element:
    start, end = html.index("<svg"), html.index("</svg>") + len("</svg>")
    return ET.fromstring(html[start:end])


def test_dumbbell_is_valid_svg_with_escaped_labels_and_a_zero_line():
    rows = [{"quantizer": "rqkmeans", "depth": 3, "w128": (0.2, 0.1, 0.3), "w512": (-0.1, -0.2, 0.0)},
            {"quantizer": "MQ", "depth": 5, "w128": None, "w512": (0.5, float("nan"), float("nan"))}]
    html = SC.dumbbell(rows, title="a <b> & c", x_label="x")
    root = _svg(html)
    assert "a &lt;b&gt; &amp; c" in html
    assert any(el.get("class") == "zero" for el in root.iter())
    marks = [el for el in root.iter() if el.get("class") == "mark"]
    assert len(marks) == 3 and all(el.get("data-tip") for el in marks)


def test_profiles_hbars_and_table_parse():
    panels = [{"title": "depth 3", "series": [{"quantizer": "rqvae", "label": "RQ-VAE 3×128", "dashed": False,
                                                "points": [(0, 0.1, 0.0, 0.2), (1, 0.4, 0.3, 0.5)]}]}]
    _svg(SC.profiles(panels, title="t", x_labels=["a", "b"], y_label="y"))
    _svg(SC.hbars([{"label": "x", "est": -0.2, "lo": -0.3, "hi": -0.1}], title="t", x_label="x"))
    assert "<td class=\"num\">1</td>" in SC.table(["a", "b"], [["r", 1]])
