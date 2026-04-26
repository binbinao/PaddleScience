---
name: md2pdf
description: "Convert Markdown files to beautifully styled PDF documents with Chinese font support and Mermaid diagram rendering. Use when user wants to export, convert, or generate PDF from Markdown files."
---

# Markdown to PDF Converter

Convert Markdown files to professionally styled PDF documents with full CJK (Chinese/Japanese/Korean) font support and optional Mermaid diagram rendering.

## When to Use This

**Trigger this skill when the user wants to:**
- Convert a Markdown file to PDF
- Export an article/document as PDF
- Generate a printable version of a Markdown document
- Create a PDF report from Markdown source

**Key phrases that should trigger:**
- "转换成PDF"
- "导出PDF"
- "生成PDF"
- "convert to PDF"
- "export as PDF"
- "markdown to pdf"
- "打印版本"

## Prerequisites

The conversion script is located at `scripts/md2pdf.mjs`. It requires:

1. **Node.js** — Runtime environment
2. **pandoc** — Markdown to HTML conversion (`brew install pandoc`)
3. **Google Chrome / Edge / Chromium** — One of these browsers must be installed locally for Puppeteer PDF rendering
4. **puppeteer-core** — Installed as a project dependency or available via mermaid-cli

### Quick Check

Before running, verify dependencies:

```bash
which node && which pandoc && echo "✅ Ready"
```

If Chrome is at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`, you're good to go.

## Usage

### Basic Command

```bash
node scripts/md2pdf.mjs <input.md> [output.pdf]
```

- `<input.md>` — **Required.** Path to the Markdown file to convert.
- `[output.pdf]` — Optional. Output PDF path. Defaults to the same filename with `.pdf` extension.

### Examples

```bash
# Convert with auto-named output (input.pdf)
node scripts/md2pdf.mjs docs/writing/my-article.md

# Convert with custom output path
node scripts/md2pdf.mjs docs/writing/my-article.md ~/Desktop/report.pdf
```

## What the Script Does

The conversion follows a 6-step pipeline:

```
Markdown → Extract Mermaid → Render Diagrams → Replace with Images → pandoc HTML → Styled HTML → Puppeteer PDF
```

### Step-by-Step

1. **Extract Mermaid blocks** — Finds all ` ```mermaid ` code blocks in the Markdown
2. **Render Mermaid to PNG** — Uses `mmdc` (mermaid-cli) to render each diagram as a high-res PNG
3. **Replace code blocks with images** — Swaps Mermaid source with inline base64 `<img>` tags
4. **Convert to HTML** — Uses `pandoc` to convert processed Markdown to HTML5
5. **Apply styled template** — Wraps HTML in a professional template with:
   - CJK font stack: PingFang SC, Hiragino Sans GB, Microsoft YaHei, Noto Sans CJK SC
   - Styled headings, tables, blockquotes, code blocks
   - Print-optimized CSS with proper page breaks
   - A4 page size with balanced margins
6. **Generate PDF** — Uses Puppeteer with system Chrome to render the styled HTML to PDF with:
   - Page numbers in footer
   - Background colors preserved
   - Proper CJK text rendering

## Features

| Feature | Supported |
|---------|-----------|
| Chinese/CJK text | ✅ Full support via system fonts |
| Markdown tables | ✅ Styled with alternating row colors |
| Code blocks | ✅ Syntax-highlighted with monospace font |
| Mermaid diagrams | ✅ Auto-rendered to high-res PNG |
| Blockquotes | ✅ Styled with blue left border |
| Nested lists | ✅ Ordered and unordered |
| Bold/Italic | ✅ Preserved |
| Links | ✅ Clickable in PDF |
| Page breaks | ✅ Auto at H1, avoid-break on tables |
| YAML frontmatter | ✅ Automatically stripped |
| Page numbers | ✅ Footer with current/total pages |

## PDF Styling

The generated PDF uses these design specs:

- **Page size:** A4 (210mm × 297mm)
- **Margins:** 20mm top/bottom, 18mm left/right
- **Body font:** 11pt, line-height 1.8
- **H1:** 22pt, dark blue (#1a365d), blue bottom border, page break before
- **H2:** 16pt, medium blue (#2c5282), light blue bottom border
- **H3:** 13pt, dark gray (#2d3748)
- **Tables:** Blue header (#2b6cb0), alternating row backgrounds
- **Code:** SF Mono / Fira Code, light gray background

## Troubleshooting

### "No Chrome/Edge/Chromium found"
Install Google Chrome from https://www.google.com/chrome/ — the script looks for browsers at standard macOS paths.

### "pandoc: command not found"
Install pandoc: `brew install pandoc`

### Mermaid diagrams not rendering
Install mermaid-cli: `npm install -g @mermaid-js/mermaid-cli`

If mermaid-cli fails, diagrams will be kept as code blocks in the PDF — the rest of the document will still convert correctly.

### CJK characters showing as boxes
Ensure you have CJK fonts installed. macOS ships with PingFang SC and Hiragino Sans GB by default.

## Integration with Document Superpowers

This skill works as a natural companion to the `document-superpowers` writing workflow:

```
📚 Reference → 💡 Brainstorm → 📋 Plan → ✍️ Write → 🔍 Review → 📄 PDF Export
```

After completing a writing project with `document-superpowers`, use this skill to export the final article:

```bash
node scripts/md2pdf.mjs docs/writing/YYYY-MM-DD-topic-final.md
```

## Agent Instructions

When the user requests a PDF conversion:

1. **Identify the input file** — Ask or infer the Markdown file path
2. **Run the script** — Execute `node scripts/md2pdf.mjs <input> [output]`
3. **Report the result** — Confirm success and provide the output file path
4. **Handle errors** — If the script fails, check prerequisites and suggest fixes

**Example agent flow:**

```
User: "把这篇文章转成PDF"
Agent: 
  1. Identify file: docs/writing/2026-04-12-article-final.md
  2. Run: node scripts/md2pdf.mjs docs/writing/2026-04-12-article-final.md
  3. Report: "PDF已生成: docs/writing/2026-04-12-article-final.pdf"
```

Do NOT attempt to write custom PDF generation code — always use the existing `scripts/md2pdf.mjs` script.
