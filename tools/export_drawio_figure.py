"""Export the editable draw.io architecture figure to SVG and PDF.

The repository stores Figure 1 as a simple, uncompressed diagrams.net XML file.
This helper renders the subset of draw.io shapes used by the paper figure
directly to SVG so the LaTeX source can include a stable vector PDF without
depending on the draw.io desktop CLI.
"""

from __future__ import annotations

import html
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas


def parse_style(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in style.split(";"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        out[key] = value
    return out


def clean_label(value: str) -> list[str]:
    if not value:
        return []
    text = html.unescape(value)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</?(div|span|font|b|strong|i|em)[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def svg_escape(value: str) -> str:
    return html.escape(value, quote=True)


def get_geom(cell: ET.Element) -> dict[str, float]:
    geom = cell.find("mxGeometry")
    if geom is None:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    return {
        "x": float(geom.attrib.get("x", 0)),
        "y": float(geom.attrib.get("y", 0)),
        "width": float(geom.attrib.get("width", 0)),
        "height": float(geom.attrib.get("height", 0)),
    }


def center(box: dict[str, float]) -> tuple[float, float]:
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def perimeter_point(box: dict[str, float], toward: tuple[float, float]) -> tuple[float, float]:
    cx, cy = center(box)
    tx, ty = toward
    dx, dy = tx - cx, ty - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return cx, cy
    hw, hh = max(box["width"] / 2, 1e-9), max(box["height"] / 2, 1e-9)
    scale = min(hw / abs(dx) if dx else math.inf, hh / abs(dy) if dy else math.inf)
    return cx + dx * scale, cy + dy * scale


def render_text(
    x: float,
    y: float,
    width: float,
    height: float,
    lines: list[str],
    style: dict[str, str],
    anchor: str = "middle",
) -> str:
    if not lines:
        return ""
    font_size = float(style.get("fontSize", "16"))
    family = style.get("fontFamily", "Georgia")
    weight = "700" if style.get("fontStyle") == "1" else "400"
    line_height = font_size * 1.18
    total = line_height * (len(lines) - 1)
    cx = x + width / 2 if anchor == "middle" else x
    start_y = y + height / 2 - total / 2 + font_size * 0.35
    spans = []
    for i, line in enumerate(lines):
        spans.append(
            f'<text x="{cx:.2f}" y="{start_y + i * line_height:.2f}" '
            f'font-family="{svg_escape(family)}" font-size="{font_size:.1f}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="#111111">{svg_escape(line)}</text>'
        )
    return "\n".join(spans)


def render_vertex(cell: ET.Element) -> str:
    style = parse_style(cell.attrib.get("style", ""))
    geom = get_geom(cell)
    lines = clean_label(cell.attrib.get("value", ""))
    if "text" in cell.attrib.get("style", ""):
        return render_text(geom["x"], geom["y"], geom["width"], geom["height"], lines, style)

    fill = style.get("fillColor", "#ffffff")
    fill = "none" if fill == "none" else fill
    stroke = style.get("strokeColor", "#222222")
    stroke_width = float(style.get("strokeWidth", "1"))
    dashed = ' stroke-dasharray="7 5"' if style.get("dashed") == "1" else ""
    rounded = style.get("rounded") == "1"
    rx = min(10.0, geom["height"] * 0.18) if rounded else 0
    rect = (
        f'<rect x="{geom["x"]:.2f}" y="{geom["y"]:.2f}" '
        f'width="{geom["width"]:.2f}" height="{geom["height"]:.2f}" '
        f'rx="{rx:.2f}" ry="{rx:.2f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.2f}"{dashed}/>'
    )
    text = render_text(geom["x"], geom["y"], geom["width"], geom["height"], lines, style)
    return rect + ("\n" + text if text else "")


def edge_points(
    cell: ET.Element, vertices: dict[str, dict[str, float]]
) -> list[tuple[float, float]]:
    source = vertices[cell.attrib["source"]]
    target = vertices[cell.attrib["target"]]
    geom = cell.find("mxGeometry")
    middle: list[tuple[float, float]] = []
    if geom is not None:
        for point in geom.findall(".//mxPoint"):
            middle.append((float(point.attrib["x"]), float(point.attrib["y"])))
    first_toward = middle[0] if middle else center(target)
    last_from = middle[-1] if middle else center(source)
    start = perimeter_point(source, first_toward)
    end = perimeter_point(target, last_from)
    return [start, *middle, end]


def render_edge(cell: ET.Element, vertices: dict[str, dict[str, float]]) -> str:
    style = parse_style(cell.attrib.get("style", ""))
    pts = edge_points(cell, vertices)
    stroke = style.get("strokeColor", "#111111")
    width = float(style.get("strokeWidth", "2"))
    dashed = ' stroke-dasharray="7 5"' if style.get("dashed") == "1" else ""
    marker = ' marker-end="url(#arrow)"' if style.get("endArrow") else ""
    path = " ".join(
        [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
        + [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
    )
    svg = (
        f'<path d="{path}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width:.2f}" stroke-linejoin="round" '
        f'stroke-linecap="round"{dashed}{marker}/>'
    )
    label_lines = clean_label(cell.attrib.get("value", ""))
    if label_lines:
        mid = pts[len(pts) // 2]
        label = label_lines[0]
        fs = float(style.get("fontSize", "16"))
        approx_w = max(36, len(label) * fs * 0.48)
        lx, ly = mid[0], mid[1] - 6
        svg += (
            f'\n<rect x="{lx - approx_w / 2:.2f}" y="{ly - fs:.2f}" '
            f'width="{approx_w:.2f}" height="{fs + 6:.2f}" fill="#ffffff" '
            f'opacity="0.92"/>'
            f'\n<text x="{lx:.2f}" y="{ly:.2f}" font-family="Georgia" '
            f'font-size="{fs:.1f}" text-anchor="middle" fill="#111111">'
            f"{svg_escape(label)}</text>"
        )
    return svg


def parse_color(value: str | None, default: colors.Color = colors.black):
    if not value or value == "none":
        return None if value == "none" else default
    try:
        return colors.HexColor(value)
    except Exception:
        return default


def pdf_y(page_height: float, y: float) -> float:
    return page_height - y


def draw_pdf_text(
    c: canvas.Canvas,
    page_height: float,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: list[str],
    style: dict[str, str],
) -> None:
    if not lines:
        return
    font_size = float(style.get("fontSize", "16"))
    bold = style.get("fontStyle") == "1"
    font_name = "Times-Bold" if bold else "Times-Roman"
    line_height = font_size * 1.18
    total = line_height * (len(lines) - 1)
    cx = x + width / 2
    baseline_top = y + height / 2 - total / 2 + font_size * 0.35
    c.setFillColor(colors.black)
    c.setFont(font_name, font_size)
    for i, line in enumerate(lines):
        text = str(line)
        c.drawCentredString(cx, pdf_y(page_height, baseline_top + i * line_height), text)


def draw_pdf_arrowhead(
    c: canvas.Canvas,
    page_height: float,
    start: tuple[float, float],
    end: tuple[float, float],
    color,
) -> None:
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    length = 12.0
    width = 7.0
    back_x = ex - length * math.cos(angle)
    back_y = ey - length * math.sin(angle)
    left = (
        back_x + width * math.sin(angle),
        back_y - width * math.cos(angle),
    )
    right = (
        back_x - width * math.sin(angle),
        back_y + width * math.cos(angle),
    )
    p = c.beginPath()
    p.moveTo(ex, pdf_y(page_height, ey))
    p.lineTo(left[0], pdf_y(page_height, left[1]))
    p.lineTo(right[0], pdf_y(page_height, right[1]))
    p.close()
    c.setFillColor(color)
    c.drawPath(p, fill=1, stroke=0)


def render_pdf(
    pdf_path: Path,
    width: float,
    height: float,
    cells: list[ET.Element],
    vertices: dict[str, dict[str, float]],
) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=(width, height))
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    for cell in cells:
        if cell.attrib.get("vertex") != "1":
            continue
        style = parse_style(cell.attrib.get("style", ""))
        geom = get_geom(cell)
        lines = clean_label(cell.attrib.get("value", ""))
        if "text" in cell.attrib.get("style", ""):
            draw_pdf_text(c, height, geom["x"], geom["y"], geom["width"], geom["height"], lines, style)
            continue
        fill = parse_color(style.get("fillColor"), colors.white)
        stroke = parse_color(style.get("strokeColor"), colors.black)
        c.setStrokeColor(stroke or colors.black)
        c.setLineWidth(float(style.get("strokeWidth", "1")))
        if style.get("dashed") == "1":
            c.setDash(7, 5)
        else:
            c.setDash()
        if fill is None:
            c.setFillColor(colors.white)
            fill_flag = 0
        else:
            c.setFillColor(fill)
            fill_flag = 1
        rx = min(10.0, geom["height"] * 0.18) if style.get("rounded") == "1" else 0
        x, y, w, h = geom["x"], geom["y"], geom["width"], geom["height"]
        if rx:
            c.roundRect(x, pdf_y(height, y + h), w, h, rx, stroke=1, fill=fill_flag)
        else:
            c.rect(x, pdf_y(height, y + h), w, h, stroke=1, fill=fill_flag)
        draw_pdf_text(c, height, x, y, w, h, lines, style)

    for cell in cells:
        if cell.attrib.get("edge") != "1":
            continue
        style = parse_style(cell.attrib.get("style", ""))
        pts = edge_points(cell, vertices)
        stroke = parse_color(style.get("strokeColor"), colors.black) or colors.black
        c.setStrokeColor(stroke)
        c.setLineWidth(float(style.get("strokeWidth", "2")))
        if style.get("dashed") == "1":
            c.setDash(7, 5)
        else:
            c.setDash()
        p = c.beginPath()
        p.moveTo(pts[0][0], pdf_y(height, pts[0][1]))
        for x, y in pts[1:]:
            p.lineTo(x, pdf_y(height, y))
        c.drawPath(p, fill=0, stroke=1)
        if style.get("endArrow"):
            draw_pdf_arrowhead(c, height, pts[-2], pts[-1], stroke)
        c.setDash()
        label_lines = clean_label(cell.attrib.get("value", ""))
        if label_lines:
            fs = float(style.get("fontSize", "16"))
            label = label_lines[0]
            mid = pts[len(pts) // 2]
            approx_w = max(36, len(label) * fs * 0.48)
            lx, ly = mid[0], mid[1] - 6
            c.setFillColor(colors.white)
            c.rect(lx - approx_w / 2, pdf_y(height, ly - fs) - (fs + 6), approx_w, fs + 6, stroke=0, fill=1)
            c.setFillColor(colors.black)
            c.setFont("Times-Roman", fs)
            c.drawCentredString(lx, pdf_y(height, ly), label)

    c.showPage()
    c.save()


def export_drawio(drawio_path: Path, svg_path: Path, pdf_path: Path) -> None:
    root = ET.parse(drawio_path).getroot()
    model = root.find(".//mxGraphModel")
    if model is None:
        raise RuntimeError("No mxGraphModel found in draw.io file")
    width = float(model.attrib.get("pageWidth", 1180))
    height = float(model.attrib.get("pageHeight", 430))
    cells = list(model.findall(".//mxCell"))
    vertices = {
        cell.attrib["id"]: get_geom(cell)
        for cell in cells
        if cell.attrib.get("vertex") == "1"
    }
    vertex_svg = [
        render_vertex(cell) for cell in cells if cell.attrib.get("vertex") == "1"
    ]
    edge_svg = [
        render_edge(cell, vertices) for cell in cells if cell.attrib.get("edge") == "1"
    ]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
          markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#111111"/>
  </marker>
</defs>
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
<g>
{chr(10).join(vertex_svg)}
</g>
<g>
{chr(10).join(edge_svg)}
</g>
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")
    render_pdf(pdf_path, width, height, cells, vertices)


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: export_drawio_figure.py INPUT.drawio OUTPUT.svg OUTPUT.pdf",
            file=sys.stderr,
        )
        return 2
    export_drawio(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
