"""Build docs/CliniCore_technical_report.pdf from docs/technical-report.md.

    pip install markdown playwright && python -m playwright install chromium
    python scripts/build_report_pdf.py

GIFs cannot play in a PDF, so each animation is replaced by its static PNG.
"""
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"
SRC, HTML, PDF = DOCS / "technical-report.md", DOCS / "_report.html", DOCS / "CliniCore_technical_report.pdf"

CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
body { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; color: #1a2233; font-size: 10.2pt; line-height: 1.5; }
h1 { font-size: 24pt; color: #0b1220; margin: 0 0 4pt; }
h2 { font-size: 15pt; color: #0b1220; border-bottom: 2px solid #2dd4bf; padding-bottom: 3pt; margin-top: 20pt; page-break-after: avoid; }
h3 { font-size: 12pt; color: #134e4a; margin-top: 14pt; page-break-after: avoid; }
h2 { page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }
img { max-width: 100%; border-radius: 6px; margin: 6pt 0; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.9pt; page-break-inside: avoid; }
th { background: #0b1220; color: #e6ecf5; text-align: left; padding: 4pt 6pt; }
td { border-bottom: 1px solid #d5dbe6; padding: 3.5pt 6pt; vertical-align: top; }
tr:nth-child(even) td { background: #f4f7fb; }
code { background: #eef2f7; padding: 1px 4px; border-radius: 3px; font-size: 8.8pt; }
pre { background: #0b1220; color: #e6ecf5; padding: 8pt; border-radius: 6px; font-size: 8pt; white-space: pre-wrap; }
pre code { background: none; color: inherit; }
blockquote { border-left: 4px solid #f59e0b; margin: 8pt 0; padding: 4pt 10pt; background: #fff8eb; }
hr { border: none; border-top: 1px solid #d5dbe6; margin: 10pt 0; }
"""


def main():
    md = SRC.read_text(encoding="utf-8")
    md = re.sub(r"\(img/(model_stage\d)\.gif\)", r"(img/\1.png)", md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page()
        pg.goto(HTML.as_uri()); pg.wait_for_load_state("networkidle")
        pg.pdf(path=str(PDF), format="A4", print_background=True, display_header_footer=True,
               header_template="<div></div>",
               footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'>CliniCore technical report · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
               margin=dict(top="16mm", bottom="18mm", left="14mm", right="14mm"))
        b.close()
    HTML.unlink()
    print("wrote", PDF)


if __name__ == "__main__":
    main()
