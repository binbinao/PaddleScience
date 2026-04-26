---
name: tech-doc-image-search
description: "Enhancement plugin for managing images and visual assets in technical documentation. Generates image placeholders, alt text, and source recommendations. Invoked automatically by tech-doc-writing during execution stage."
---

# Tech Doc Image Search

## Overview

Enhancement plugin that manages visual assets during the execution stage of technical document writing. Since AI agents cannot capture screenshots or search image databases directly, this skill focuses on creating structured image placeholders, writing descriptive alt text, recommending image sources, and ensuring visual density is appropriate.

**Invoked by:** `tech-doc-writing` during execution stage, or manually when a section needs visual elements.

**Trigger signals:**
- Section describes a UI workflow or dashboard
- Section shows terminal output that benefits from a screenshot
- Section explains a visual concept (network topology, deployment layout)
- The document has gone 800+ words without any visual element
- Section references "as shown in the screenshot" or similar

## Image Types for Technical Documentation

### 1. Product Screenshots

Captures of application UI, admin panels, configuration screens.

**When to use:** Tutorials, how-to guides, troubleshooting docs where the reader needs to see what to click or where to look.

**Placeholder format:**
```markdown
![Dashboard showing the monitoring overview with CPU and memory graphs](images/monitoring-dashboard-overview.png)
<!-- IMAGE-TODO: Screenshot of the monitoring dashboard at /admin/monitoring
     - Browser: Chrome, viewport 1280x800
     - State: Show 24h view with sample data loaded
     - Highlight: Circle the "Alert Rules" button in the top-right -->
```

### 2. Terminal / CLI Output

Screenshots of terminal sessions showing command output, especially when formatting matters (colors, tables, progress bars).

**When to use:** When the output formatting is important to the reader's understanding, or when the output is too long/complex for a code block.

**Placeholder format:**
```markdown
![Terminal showing kubectl describe pod output with events section highlighted](images/kubectl-describe-output.png)
<!-- IMAGE-TODO: Terminal screenshot
     - Command: kubectl describe pod api-server-7d4b8c6f9-x2k4m -n production
     - Terminal: Dark theme, 120 columns wide
     - Highlight: The "Events" section at the bottom
     - Note: If output is short enough, prefer a code block instead -->
```

**Prefer code blocks** when the output is plain text and short. Use screenshots only when visual formatting adds value.

### 3. Architecture Diagrams (External Tools)

Complex diagrams beyond Mermaid's capabilities — C4 models, detailed network topologies, cloud infrastructure layouts.

**When to use:** When Mermaid cannot adequately represent the diagram (too complex, needs icons, requires precise layout).

**Placeholder format:**
```markdown
![AWS architecture showing VPC with public/private subnets, ALB, ECS cluster, and RDS](images/aws-architecture.png)
<!-- IMAGE-TODO: Create using draw.io or Excalidraw
     - Components: VPC, 2 public subnets, 2 private subnets, ALB, ECS Fargate, RDS Aurora
     - Show: Traffic flow from internet → ALB → ECS → RDS
     - Include: Security groups boundaries, NAT Gateway
     - Style: Use official AWS architecture icons
     - Export: PNG at 2x resolution, max width 1200px
     - Source file: Save .drawio alongside the PNG -->
```

### 4. Comparison / Benchmark Charts

Performance graphs, benchmark results, feature comparison visuals.

**When to use:** Technology comparison docs, performance tuning guides, capacity planning documents.

**Placeholder format:**
```markdown
![Bar chart comparing response latency: Redis 0.5ms, PostgreSQL 2.1ms, MongoDB 1.8ms](images/latency-comparison.png)
<!-- IMAGE-TODO: Create using matplotlib, gnuplot, or a charting tool
     - Type: Horizontal bar chart
     - Data: Redis=0.5ms, PostgreSQL=2.1ms, MongoDB=1.8ms, SQLite=3.2ms
     - Labels: Include exact values on each bar
     - Source: Link to benchmark script in /benchmarks/latency-test.py
     - Note: Include test conditions (hardware, dataset size, concurrency) in caption -->
```

### 5. Logos and Icons

Technology logos used in comparison tables or architecture diagrams.

**When to use:** Technology comparison articles, tool selection guides.

**Guidelines:**
- Use official logos from brand guidelines pages
- Ensure licensing allows usage (most tech logos permit editorial use)
- Use SVG format when available for crisp rendering
- Keep logos at consistent size (32x32 or 64x64 for inline, larger for headers)

## Image Placeholder Standard

Every image placeholder MUST include:

### Required Elements

```markdown
![Descriptive alt text that explains what the image shows](images/YYYY-MM-DD-topic/descriptive-name.png)
<!-- IMAGE-TODO: Detailed instructions for creating or obtaining this image
     - Source: Where to get it (screenshot, create, download)
     - Specifications: Size, format, resolution
     - Context: What state/view/page to capture
     - Highlights: What to emphasize (circles, arrows, annotations) -->
```

### File Naming Convention

```
images/
└── YYYY-MM-DD-topic/
    ├── 01-dashboard-overview.png
    ├── 02-create-user-form.png
    ├── 03-api-response-terminal.png
    ├── 04-architecture-diagram.png
    └── 04-architecture-diagram.drawio    # Source file for editable diagrams
```

Rules:
- Prefix with section number for ordering (`01-`, `02-`, ...)
- Use lowercase kebab-case for file names
- Keep names descriptive but concise (3-5 words)
- Store source files (`.drawio`, `.excalidraw`) alongside exports
- Use PNG for screenshots, SVG for diagrams when possible

## Alt Text Guidelines

Alt text serves readers using screen readers and also acts as documentation for the image content.

### Good Alt Text

- Describes the **content and purpose** of the image
- Includes **key data** visible in the image
- Provides **context** for screenshots (what page, what state)

**Examples:**

```markdown
<!-- Good: Describes content and key information -->
![Grafana dashboard showing API latency p99 at 45ms with a spike to 120ms at 14:30](...)

<!-- Good: Describes UI state for tutorial -->
![AWS Console EC2 launch wizard on step 3, showing t3.medium instance type selected](...)

<!-- Good: Describes diagram content -->
![Sequence diagram showing OAuth2 authorization code flow between client, auth server, and resource server](...)
```

### Bad Alt Text

```markdown
<!-- Bad: Too vague -->
![Dashboard](...)

<!-- Bad: Describes appearance, not content -->
![A screenshot with blue and green colors](...)

<!-- Bad: Just a filename -->
![screenshot-2025-01-15.png](...)
```

## Image Source Recommendations

### Priority 1: Create Your Own

- **Screenshots:** Capture from actual running application
- **Diagrams:** Create with draw.io, Excalidraw, or Mermaid
- **Charts:** Generate from actual data with matplotlib, gnuplot, or charting libraries
- **Advantage:** Full control, no licensing issues, always up to date

### Priority 2: Official Documentation

- Many open-source projects provide architecture diagrams and screenshots under permissive licenses
- Always check the license (usually found in the repo's LICENSE file)
- Link to the source and credit the project

### Priority 3: Open-Source Tools

| Tool | Best For | Format |
|---|---|---|
| draw.io / diagrams.net | Architecture diagrams, flowcharts | SVG, PNG |
| Excalidraw | Hand-drawn style diagrams | SVG, PNG |
| Mermaid (use tech-doc-mermaid) | Simple diagrams in markdown | Rendered inline |
| PlantUML | UML diagrams | SVG, PNG |
| Kroki | Multiple diagram types via API | SVG, PNG |

### Avoid

- Stock photo sites (generic, adds no technical value)
- Unlicensed screenshots from proprietary software
- Low-resolution or outdated screenshots
- Images with visible personal information or real credentials

## Visual Density Guidelines

Technical documentation needs visual breaks to maintain readability.

| Document Length | Minimum Visual Elements | Guideline |
|---|---|---|
| 500-1000 words | 1-2 | At least one diagram or screenshot |
| 1000-2000 words | 2-4 | Visual element every 400-600 words |
| 2000-3000 words | 4-6 | Mix of diagrams, code blocks, and screenshots |
| 3000+ words | 6+ | No more than 800 words without a visual break |

**Visual elements include:** Mermaid diagrams, code blocks (>5 lines), screenshots, tables (>3 rows), and images.

**Note:** Code blocks count as visual elements. A tutorial heavy with code blocks may not need additional images.

## Quality Checklist

Before finalizing a section with images, verify:

- [ ] Every `![](...)` has descriptive alt text (not empty, not just a filename)
- [ ] Every image placeholder has an `IMAGE-TODO` comment with clear creation instructions
- [ ] Images are placed in context (near the text that references them)
- [ ] File paths follow the naming convention (`images/YYYY-MM-DD-topic/NN-name.png`)
- [ ] No section exceeds 800 words without a visual element (diagram, code block, image, or table)
- [ ] Screenshot instructions specify browser, viewport, and application state
- [ ] Diagram source files are referenced alongside exported images
- [ ] No images contain sensitive data (credentials, personal info, internal URLs)
- [ ] Image format is appropriate (PNG for screenshots, SVG for diagrams)

## Handling IMAGE-TODO in Review

During the review stage, `IMAGE-TODO` markers should be tracked:

```markdown
## Image Status

**Total image placeholders:** 6
**Created:** 2 (diagram generated with draw.io)
**Pending:** 4

### Pending Images
1. `01-dashboard-overview.png` — Needs screenshot from staging environment
2. `03-api-response-terminal.png` — Can be replaced with code block (simpler)
3. `05-performance-chart.png` — Waiting for benchmark results
4. `06-deployment-flow.png` — Can be replaced with Mermaid diagram

### Recommendations
- Image #3: Convert to code block (terminal output is plain text)
- Image #6: Use tech-doc-mermaid to generate inline diagram
```

## Key Principles

- **Placeholder over nothing** — A well-described IMAGE-TODO is better than no visual plan
- **Alt text is documentation** — Write it for both accessibility and future maintainers
- **Create over search** — Custom screenshots and diagrams are always more relevant
- **Visual density matters** — Long walls of text lose readers; break them up
- **Track completeness** — Every IMAGE-TODO should be resolved before publication
- **Source files alongside exports** — Keep .drawio/.excalidraw files for future edits
- **Prefer Mermaid when possible** — Inline diagrams are easier to maintain than external image files
