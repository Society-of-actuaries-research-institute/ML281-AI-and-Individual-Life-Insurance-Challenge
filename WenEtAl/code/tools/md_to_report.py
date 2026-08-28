# -*- coding: utf-8 -*-
"""
MD to HTML Report Converter
============================
Converts .md files to styled actuarial report HTML.
Open the output .html in any browser -> Ctrl+P -> Save as PDF.

Usage:
    python code/tools/md_to_report.py data/reports/validation_and_governance_report.md
    python code/tools/md_to_report.py data/reports/actuarial_report.md
    python code/tools/md_to_report.py --all   (convert all .md in data/reports/)
"""

import sys
import argparse
from pathlib import Path
import markdown

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #1a1a1a;
    background: #fff;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 60px;
}

/* Page-like appearance */
@media screen {
    body {
        background: #e8e8e8;
        padding: 0;
    }
    .page {
        background: white;
        max-width: 900px;
        margin: 30px auto;
        padding: 60px 80px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
}

@media print {
    body { background: white; }
    .page { box-shadow: none; margin: 0; padding: 40px 50px; }
    h1 { page-break-before: always; }
    h1:first-child { page-break-before: avoid; }
    table { page-break-inside: avoid; }
}

/* Header */
h1 {
    font-size: 18pt;
    font-weight: bold;
    color: #1a3a5c;
    border-bottom: 3px solid #1a3a5c;
    padding-bottom: 8px;
    margin-top: 36px;
    margin-bottom: 16px;
}
h1:first-of-type { margin-top: 0; }

h2 {
    font-size: 13pt;
    font-weight: bold;
    color: #1a3a5c;
    border-bottom: 1px solid #c0d0e0;
    padding-bottom: 4px;
    margin-top: 28px;
    margin-bottom: 12px;
}

h3 {
    font-size: 11.5pt;
    font-weight: bold;
    color: #2c5282;
    margin-top: 20px;
    margin-bottom: 8px;
}

h4 {
    font-size: 11pt;
    font-weight: bold;
    color: #333;
    margin-top: 16px;
    margin-bottom: 6px;
}

/* Paragraph */
p {
    margin-bottom: 10px;
    text-align: justify;
}

/* Lists */
ul, ol {
    margin: 8px 0 12px 24px;
}
li {
    margin-bottom: 4px;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 10pt;
}
thead tr {
    background: #1a3a5c;
    color: white;
}
th {
    padding: 8px 10px;
    text-align: left;
    font-weight: bold;
    border: 1px solid #1a3a5c;
}
td {
    padding: 7px 10px;
    border: 1px solid #c0d0e0;
    vertical-align: top;
}
tbody tr:nth-child(even) {
    background: #f0f5fa;
}
tbody tr:hover {
    background: #dceeff;
}

/* Code / inline code */
code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 9.5pt;
    background: #f0f4f8;
    border: 1px solid #d0dce8;
    border-radius: 3px;
    padding: 1px 4px;
    color: #1a3a5c;
}
pre {
    background: #f8fafc;
    border: 1px solid #d0dce8;
    border-left: 4px solid #1a3a5c;
    border-radius: 4px;
    padding: 14px 16px;
    overflow-x: auto;
    margin: 12px 0;
    font-size: 9.5pt;
    line-height: 1.5;
}
pre code {
    background: none;
    border: none;
    padding: 0;
    color: #1a1a1a;
}

/* Blockquote */
blockquote {
    border-left: 4px solid #2c80c4;
    background: #f0f7ff;
    margin: 12px 0;
    padding: 10px 16px;
    font-style: italic;
    color: #2c4a6e;
}

/* Horizontal rule */
hr {
    border: none;
    border-top: 1px solid #c0d0e0;
    margin: 24px 0;
}

/* Strong / emphasis */
strong {
    font-weight: bold;
    color: #1a1a1a;
}
em {
    font-style: italic;
    color: #333;
}

/* Status badges */
.badge-pass { color: #276749; font-weight: bold; }
.badge-fail { color: #c53030; font-weight: bold; }
.badge-warn { color: #b7791f; font-weight: bold; }

/* Report header block */
.report-meta {
    background: #f0f5fa;
    border: 1px solid #c0d0e0;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 24px;
    font-size: 10pt;
}

/* Footer */
.footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #c0d0e0;
    font-size: 9pt;
    color: #666;
    font-style: italic;
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css}
    </style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>
"""


def convert(md_path: Path, out_path: Path = None):
    if not md_path.exists():
        print(f"[ERROR] File not found: {md_path}")
        return

    md_text = md_path.read_text(encoding='utf-8')

    # Extract title from first H1
    title = md_path.stem.replace('_', ' ').title()
    for line in md_text.splitlines():
        if line.startswith('# '):
            title = line[2:].strip()
            break

    # Convert markdown to HTML
    md_converter = markdown.Markdown(
        extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'nl2br',
            'toc',
            'attr_list',
        ],
        extension_configs={
            'codehilite': {'css_class': 'highlight', 'guess_lang': False},
        }
    )
    body_html = md_converter.convert(md_text)

    # Post-process: highlight [OK], [DETECT], [MISS] badges
    body_html = body_html.replace('[OK]', '<span class="badge-pass">[OK]</span>')
    body_html = body_html.replace('[DETECT]', '<span class="badge-pass">[DETECT]</span>')
    body_html = body_html.replace('[MISS]', '<span class="badge-fail">[MISS]</span>')
    body_html = body_html.replace('[WARN]', '<span class="badge-warn">[WARN]</span>')
    body_html = body_html.replace('✓', '<span class="badge-pass">&#10003;</span>')

    html = HTML_TEMPLATE.format(title=title, css=CSS, body=body_html)

    if out_path is None:
        out_path = md_path.with_suffix('.html')

    out_path.write_text(html, encoding='utf-8')
    print(f"[OK] {md_path.name} -> {out_path.name}")
    print(f"     Open in browser: file:///{out_path.as_posix()}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Convert .md report to styled HTML')
    parser.add_argument('files', nargs='*', help='.md file paths')
    parser.add_argument('--all', action='store_true', help='Convert all .md in data/reports/')
    parser.add_argument('--out', help='Output directory (default: same as input)')
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent

    if args.all:
        reports_dir = root / 'data' / 'reports'
        paths = list(reports_dir.glob('*.md'))
        if not paths:
            print(f"No .md files found in {reports_dir}")
            return
    elif args.files:
        paths = [Path(f) for f in args.files]
    else:
        # Default: convert all reports
        reports_dir = root / 'data' / 'reports'
        paths = list(reports_dir.glob('*.md'))

    out_dir = Path(args.out) if args.out else None

    converted = []
    for p in paths:
        out = (out_dir / p.with_suffix('.html').name) if out_dir else None
        result = convert(p, out)
        if result:
            converted.append(result)

    print(f"\nConverted {len(converted)} file(s).")
    if converted:
        print("\nTo view as PDF:")
        print("  1. Open any .html file in Chrome/Edge")
        print("  2. Ctrl+P -> Destination: Save as PDF -> Print")


if __name__ == '__main__':
    main()
