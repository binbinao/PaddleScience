---
name: document-superpowers
description: "Complete document writing workflow: reference materials → brainstorming → planning → execution → review. Use when user wants to write articles, blog posts, technical docs, or any long-form content."
---

# Document Superpowers

A systematic writing workflow that produces professional-quality articles, blog posts, and documentation.

## When to Use This

**Trigger this skill when the user wants to:**
- Write a blog post
- Create technical documentation
- Draft an article or essay
- Produce any long-form content (>500 words)
- Write something that needs structure and quality

**Key phrases that should trigger:**
- "Write a blog post about..."
- "I need to create documentation for..."
- "Help me write an article..."
- "I want to write about..."

## Overview

Document Superpowers enforces a structured writing process:

```
📚 Reference Materials → 💡 Brainstorming → 📋 Planning → 🔎 Research (optional) → ✍️ Execution → 🔍 Review
```

Each stage has strict gates - you cannot skip ahead without completing the previous stage. The workflow ALWAYS begins by checking for reference materials.

## The Writing Stages

### Stage 0: Reference Materials
**Skill:** `document-superpowers:reference-material`

**What it does:**
- Scans `docs/reference/` for background materials (txt, markdown, pdf, images)
- Presents file inventory and lets user select which to read
- Reads and extracts key context — themes, facts, quotes, visuals
- Generates structured summary to feed into all subsequent stages

**Trigger:** Automatically at the start of every writing project. If `docs/reference/` is empty or doesn't exist, notes this and proceeds to brainstorming.

**Output:** `docs/writing/YYYY-MM-DD-<topic>-references.md`

**Next:** Automatically invokes `brainstorming`

### Stage 1: Brainstorming
**Skill:** `document-superpowers:brainstorming`

**What it does:**
- Asks clarifying questions one at a time
- Understands audience, purpose, core message
- Presents topic brief for approval
- Saves approved brief

**Output:** `docs/writing/YYYY-MM-DD-<topic>-brief.md`

**Next:** Automatically invokes `writing-article-plan`

### Stage 2: Planning
**Skill:** `document-superpowers:writing-article-plan`

**What it does:**
- Creates detailed section-by-section outline
- Specifies key points, evidence needs, word counts
- Plans transitions between sections
- Identifies research requirements upfront

**Output:** `docs/writing/YYYY-MM-DD-<topic>-plan.md`

**Next:** Offers choice of research gathering or execution mode

### Stage 2.5: Research & Gathering (Optional)
**Skill:** `document-superpowers:research-gathering`

**What it does:**
- Reads the approved plan and generates search strategy per section
- Presents strategy for user approval before executing
- Searches for web articles, data, and images
- Organizes results by section for easy reference during writing
- Downloads key images to local assets directory

**Output:**
- Research document: `docs/writing/YYYY-MM-DD-<topic>-research.md`
- Image assets: `docs/writing/assets/YYYY-MM-DD-<topic>/`

**Next:** Offers choice of execution mode

### Stage 3: Execution
**Skills:** 
- `document-superpowers:executing-article-plan` (batch mode)
- `document-superpowers:subagent-driven-writing` (iterative mode)

**What it does:**
- Writes sections according to plan
- Self-checks each section
- Provides checkpoints for human review
- Merges into full draft

**Output:** 
- Section drafts: `docs/writing/drafts/YYYY-MM-DD-<topic>/section-N-*.md`
- Full draft: `docs/writing/YYYY-MM-DD-<topic>-draft.md`

**Next:** Automatically invokes `reviewing-article`

### Stage 4: Review
**Skill:** `document-superpowers:reviewing-article`

**What it does:**
- Four-pass systematic review:
  1. Logic (is it sound?)
  2. Evidence (is it supported?)
  3. Flow (is it smooth?)
  4. Polish (is it clean?)
- Categorizes issues by severity
- Documents all fixes
- Produces final article

**Output:**
- Review log: `docs/writing/YYYY-MM-DD-<topic>-review.md`
- Final article: `docs/writing/YYYY-MM-DD-<topic>-final.md`

## Quick Start

### For Users

1. Place any background docs in `docs/reference/` (optional)
2. Say: "I want to write about [topic]"
3. Select which reference materials to use (if any exist)
4. Answer brainstorming questions
5. Approve the brief
6. Approve the plan
7. Choose execution mode (batch or iterative)
8. Review checkpoints as writing progresses
9. Approve final article

### For Agents

1. **Detect writing intent** - Look for phrases like "write", "create article", "blog post"
2. **Scan references first** - ALWAYS check `docs/reference/` before brainstorming
3. **Invoke brainstorming** - Start with questions (informed by reference context), never skip to writing
4. **Follow the chain** - Each skill invokes the next automatically
5. **Offer research** - After planning, ask if user wants to gather materials before writing
5. **Enforce gates** - Do not skip stages or write before planning
6. **Provide checkpoints** - Regular human validation points

## File Structure

A complete writing project creates:

```
docs/
├── reference/                              # Reference materials (input)
│   ├── README.md                           # Folder guide
│   ├── *.txt, *.md, *.pdf                  # Text-based references
│   └── *.png, *.jpg, *.gif, *.webp        # Image references
└── writing/
    ├── YYYY-MM-DD-topic-references.md      # Reference summary (optional)
    ├── YYYY-MM-DD-topic-brief.md           # Stage 1 output
    ├── YYYY-MM-DD-topic-plan.md            # Stage 2 output
    ├── YYYY-MM-DD-topic-research.md        # Stage 2.5 output (optional)
    ├── assets/YYYY-MM-DD-topic/            # Stage 2.5 downloaded images (optional)
    ├── drafts/YYYY-MM-DD-topic/            # Stage 3 working directory
    │   ├── section-1-intro.md
    │   ├── section-2-argument.md
    │   └── ...
    ├── YYYY-MM-DD-topic-draft.md           # Stage 3 output
    ├── YYYY-MM-DD-topic-review.md          # Stage 4 review log
    └── YYYY-MM-DD-topic-final.md           # Stage 4 final output
```

## Execution Modes

### Batch Execution (Default)
- Write 2-3 sections at a time
- Checkpoint after each batch
- Fewer interruptions, longer focus
- **Good for:** Straightforward articles, experienced writers

### Subagent-Driven (Advanced)
- Fresh subagent per section
- Two-stage review (spec + quality)
- Tight feedback loop
- **Good for:** Complex articles, high-stakes content, learning the system

## Key Principles

### 1. No Skipping Brainstorming
Every article goes through all four stages. Even "simple" topics need briefs (they can be short, but they must exist and be approved).

### 2. One Question at a Time
Don't overwhelm users with forms or multiple questions. Build understanding through conversation.

### 3. Bite-Sized Sections
Sections are 150-400 words - small enough to write in one sitting, large enough to develop an idea.

### 4. Checkpoints Not Marathons
Review after every 2-3 sections. Catch issues early.

### 5. Four-Pass Review
Each review pass focuses on one dimension. Don't mix logic checking with grammar checking.

## Anti-Patterns to Avoid

❌ **"This is too simple to need a brief"**
- Every article needs a brief, even if it's three sentences

❌ **Writing before planning**
- Always create the plan before executing

❌ **Skipping self-checks**
- Validate each section before moving on

❌ **Marathon writing sessions**
- Use checkpoints every 2-3 sections

❌ **Single-pass review**
- Four passes catch more issues than one combined pass

## Example Usage

See `examples/typescript-blog-post.md` for a complete walkthrough.

**Quick example:**
```
User: "I want to write a blog post about code reviews"

Agent: [invokes brainstorming]
"Who's your target audience?"

User: "Junior developers"

Agent: "What should they do after reading?"

User: "Write better code review comments"

Agent: [continues questions → presents brief → gets approval]
Agent: [creates plan → offers execution choice]

User: "Batch execution"

Agent: [writes sections 1-3]
Agent: "Checkpoint: sections 1-3 complete. Continue?"

User: "Yes"

Agent: [completes draft → runs review → fixes issues]
Agent: "Article complete and ready for publication!"
```

## Tech Doc Enhancement Plugins

When writing IT technical documentation, four enhancement plugins automatically enrich the core workflow:

### tech-doc-writing (Orchestrator)
**Skill:** `document-superpowers:tech-doc-writing`

Detects technical documentation intent and injects enhancements into each stage:
- **Brainstorming:** Adds tech-specific questions (tech stack, reader level, prerequisites, code language)
- **Planning:** Provides 8 document-type templates (Tutorial, Architecture, API, Troubleshooting, Concept, Comparison, Best Practice, Release Note)
- **Review:** Adds Pass 5 — Technical Accuracy check (code correctness, version consistency, prerequisite completeness)

### tech-doc-code-blocks (Execution Plugin)
**Skill:** `document-superpowers:tech-doc-code-blocks`

Generates high-quality code blocks during execution:
- Enforces language tags, meaningful comments, complete imports
- Uses progressive complexity (Minimal → Production → Advanced)
- Provides templates for shell commands, config files, API requests, Dockerfiles

### tech-doc-mermaid (Execution Plugin)
**Skill:** `document-superpowers:tech-doc-mermaid`

Generates Mermaid diagrams during execution:
- Selects appropriate diagram type (flowchart, sequence, state, ER, timeline, git graph)
- Provides syntax templates and best practices
- Maps document types to recommended diagram types

### tech-doc-image-search (Execution Plugin)
**Skill:** `document-superpowers:tech-doc-image-search`

Manages visual assets during execution:
- Creates structured image placeholders with IMAGE-TODO markers
- Generates descriptive alt text for accessibility
- Recommends image sources and tools (draw.io, Excalidraw, screenshots)
- Ensures visual density (at least one visual element per 500-800 words)

## Integration with Other Skills

Document Superpowers works well with:

- **Reference materials** - Place background docs in `docs/reference/` for automatic context extraction
- **Research skills** - Gather evidence during planning stage
- **SEO skills** - Optimize keywords during review
- **Publishing skills** - Export to Medium/WordPress after finalization
- **Style guide skills** - Enforce organization standards in review

## Troubleshooting

**Agent skips brainstorming:**
- Ensure skill descriptions are registered
- Check HARD-GATE tags in brainstorming/SKILL.md

**Sections too long/short:**
- Adjust word targets in the plan
- Review if key points need more/less development

**Review finds no issues:**
- Review may be too shallow
- Try providing specific examples of what to look for
- Check if four passes are actually happening

**Subagent fails repeatedly:**
- Plan may be unclear - revise specifications
- Consider switching to batch execution
- Add more context to task briefs

## Files in This Skill

```
document-superpowers/
├── SKILL.md                                    # This file (entry point)
├── README.md                                   # Full documentation
├── docs/
│   └── reference/                             # Reference materials folder
│       └── README.md                          # Folder guide
├── skills/
│   ├── reference-material/SKILL.md            # Pre-stage: reference reader
│   ├── brainstorming/SKILL.md                 # Stage 1
│   ├── writing-article-plan/SKILL.md          # Stage 2
│   ├── executing-article-plan/SKILL.md        # Stage 3 (batch)
│   ├── subagent-driven-writing/SKILL.md       # Stage 3 (iterative)
│   ├── reviewing-article/SKILL.md             # Stage 4
│   ├── research-gathering/SKILL.md           # Research & gathering (optional stage)
│   ├── tech-doc-writing/SKILL.md             # Tech doc orchestrator plugin
│   ├── tech-doc-code-blocks/SKILL.md         # Code block generation plugin
│   ├── tech-doc-mermaid/SKILL.md             # Mermaid diagram plugin
│   └── tech-doc-image-search/SKILL.md        # Image management plugin
└── examples/
    └── typescript-blog-post.md                # Complete walkthrough
```

## Next Steps

**To use this skill:**
1. Install the skills in your agent's skills directory
2. Say "I want to write about [topic]"
3. Follow the prompts through each stage
4. Get publication-ready content

**To learn more:**
- Read `README.md` for full documentation
- See `examples/typescript-blog-post.md` for walkthrough
- Check individual `skills/*/SKILL.md` files for stage details

---

**Pro tip:** Trust the process. It feels slower at first but produces better results faster than iterative rewrites.
