---
name: reference-material
description: "Reads and extracts context from reference materials (txt, markdown, pdf, images) placed in docs/reference/. Use at the start of any writing project to ground content in existing materials."
---

# Reference Material Reader

## Overview

Scan, read, and extract useful context from reference materials before and during writing. Reference files are placed in `docs/reference/` and can include plain text, Markdown, PDF documents, and images.

This skill acts as a **context bridge** — it turns raw reference files into structured, actionable inputs for the brainstorming, planning, and execution stages of the writing workflow.

**Announce at start:** "I'm using the reference-material skill to scan available reference files."

**Position in workflow:** This is **Stage 0** — it runs BEFORE brainstorming. After completing reference extraction, the terminal state is invoking `brainstorming`.

<HARD-GATE>
When a writing project begins, you MUST scan `docs/reference/` BEFORE invoking brainstorming. If the folder contains files, read and extract context first. If the folder is empty or doesn't exist, note this and proceed directly to brainstorming. Do NOT skip this check.
</HARD-GATE>

## When to Invoke

**Automatically invoke this skill when:**
- The user mentions they have reference materials, background docs, or existing content
- The `docs/reference/` folder contains files at the start of a writing project
- The user says "use this as reference", "based on this document", "here's some background"

**Manual trigger phrases:**
- "Read my reference materials"
- "Check docs/reference for context"
- "I have some background docs to use"

## Supported File Types

| Type | Extensions | Read Method | Extraction |
|------|-----------|-------------|------------|
| Plain Text | `.txt` | Direct file read | Full text, key paragraphs |
| Markdown | `.md` | Direct file read | Headings, sections, key points |
| PDF | `.pdf` | PDF-to-text read | Full text extraction, page-aware |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` | Image analysis | Visual description, text in images, diagram interpretation |

## Process

### Step 1: Scan Reference Folder

Scan `docs/reference/` recursively for supported files.

Present inventory to user:

```markdown
## Available Reference Materials

Found **N files** in `docs/reference/`:

| # | File | Type | Size | Path |
|---|------|------|------|------|
| 1 | brand-guidelines.md | Markdown | 12 KB | docs/reference/brand-guidelines.md |
| 2 | competitor-report.pdf | PDF | 2.3 MB | docs/reference/competitor-report.pdf |
| 3 | architecture.png | Image | 450 KB | docs/reference/architecture.png |
| 4 | meeting-notes.txt | Text | 3 KB | docs/reference/meeting-notes.txt |

**Which files should I read for this project?** (Enter numbers, "all", or "skip")
```

### Step 2: Read Selected Files

For each selected file, read and extract content based on file type:

**Plain Text (`.txt`):**
1. Read the full file content
2. Identify key topics, facts, and quotes
3. Note any structure (numbered lists, sections separated by blank lines)

**Markdown (`.md`):**
1. Read the full file content
2. Parse heading structure to understand document organization
3. Extract key sections, definitions, and structured data (tables, lists)
4. Identify links and references for further context

**PDF (`.pdf`):**
1. Read using PDF extraction (the Read tool supports PDF natively)
2. Extract text content page by page
3. Identify document structure (headings, sections, tables)
4. Note any figures or charts described in text
5. For long PDFs (>20 pages), focus on table of contents, executive summary, and conclusion first

**Images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`):**
1. Read using image analysis (the Read tool supports images natively)
2. Describe what the image shows (diagrams, charts, screenshots, photos)
3. Extract any visible text (labels, annotations, captions)
4. Identify relationships and flows in diagrams
5. Note color coding or visual hierarchies

### Step 3: Generate Reference Summary

After reading all selected files, produce a structured summary:

```markdown
## Reference Material Summary

**Files processed:** N
**Total content extracted:** ~X words

### Key Themes
1. [Theme A] — appears in [file1, file2]
2. [Theme B] — appears in [file3]
3. [Theme C] — appears in [file1, file4]

### Key Facts & Data Points
- [Fact 1] (source: file1)
- [Fact 2] (source: file2, page 5)
- [Statistic] (source: file3)

### Key Definitions
- **[Term A]**: [definition] (source: file1)
- **[Term B]**: [definition] (source: file2)

### Visual Assets
- [Image 1]: [description] — usable as [suggested use]
- [Image 2]: [description] — usable as [suggested use]

### Quotes & Notable Passages
> "[Quote]" — source: file1
> "[Quote]" — source: file2

### Gaps & Questions
- [What's missing that the article needs?]
- [What needs clarification?]
- [What contradictions exist between sources?]
```

### Step 4: Integrate with Writing Workflow

The reference summary feeds into subsequent stages:

**→ Brainstorming (Stage 1):**
- Use extracted themes to suggest topic angles
- Use key facts to ground the core message
- Present relevant quotes as potential evidence
- Flag any constraints or requirements found in reference materials

**→ Planning (Stage 2):**
- Map extracted content to planned sections
- Assign specific facts/quotes to sections where they support key points
- Identify which sections need additional research beyond references
- Use document structure from references as inspiration for article organization

**→ Execution (Stage 3):**
- Provide section writers with relevant excerpts from reference materials
- Include specific data points and quotes for each section
- Reference images that can be reused or described in the article
- Cite sources accurately when using reference material content

**→ Review (Stage 4):**
- Cross-check article claims against reference material facts
- Verify that key information from references was incorporated
- Ensure proper attribution of ideas from reference materials

## Reference Summary File

Save the reference summary to: `docs/writing/YYYY-MM-DD-<topic>-references.md`

```markdown
---
title: Reference Material Summary
date: YYYY-MM-DD
topic: [topic name]
files_processed: N
---

# Reference Material Summary for [Topic]

[Generated summary content...]
```

## Handling Large Files

**PDF documents over 20 pages:**
1. Read the table of contents / outline first
2. Read executive summary or abstract
3. Read conclusion / recommendations
4. Ask the user which specific sections are most relevant
5. Deep-read only the selected sections

**Multiple files (>10):**
1. Present the full inventory
2. Ask the user to prioritize (top 5)
3. Read prioritized files first
4. Offer to read remaining files if needed

## Handling Mixed Languages

Reference materials may be in different languages than the target article:
- Read and understand content in its original language
- Extract and translate key points to match the article's target language
- Note the original language for attribution

## Error Handling

| Situation | Action |
|-----------|--------|
| `docs/reference/` doesn't exist | Create it and inform user to add files |
| Folder is empty | Inform user, suggest adding materials, proceed without |
| File can't be read | Log the error, skip the file, continue with others |
| PDF is scanned (image-only) | Note limitation, attempt image-based text extraction |
| Image is too small/unclear | Note limitation, describe what's visible |
| File is very large (>5MB text) | Read in chunks, summarize progressively |

## Transition to Brainstorming

After saving the reference summary (or noting that no references exist), transition to brainstorming:

**"Reference material scan complete. [N files processed / No reference files found.]**

**Next step:** Invoke brainstorming skill to explore the topic."

- **REQUIRED NEXT SKILL:** `document-superpowers:brainstorming`
- Pass the reference summary context to brainstorming so it can inform questions and enrich the brief
- Do NOT invoke writing-article-plan or any execution skill directly

**The terminal state is invoking brainstorming.** This skill feeds INTO brainstorming, not around it.

## Integration Points

This skill connects to the main workflow as Stage 0:

```
docs/reference/ (input)
       │
       ▼
reference-material (Stage 0 — this skill)
       │
       ▼
brainstorming (Stage 1) ← reference context informs questions & brief
       │
       ▼
planning (Stage 2) ← reference content mapped to sections
       │
       ▼
execution (Stage 3) ← source material provided per section
       │
       ▼
review (Stage 4) ← fact-checking against sources
```

## Key Principles

- **Read before writing** — always process references before starting content creation
- **Selective reading** — let the user choose which files matter; don't read everything blindly
- **Structured extraction** — raw content is less useful than organized summaries
- **Source attribution** — always track which file each fact came from
- **Format-aware** — use the right extraction method for each file type
- **Graceful degradation** — if a file can't be read, skip it and continue
