---
name: tech-doc-writing
description: "Enhancement plugin for IT technical documentation. Enriches brainstorming with tech-specific questions, provides 8 document-type templates for planning, and adds a Technical Accuracy review pass. Automatically triggers tech-doc-code-blocks, tech-doc-mermaid, and tech-doc-image-search during execution."
---

# Tech Doc Writing

## Overview

Enhancement plugin that augments the core document-superpowers workflow when the writing target is IT technical documentation. This skill does NOT replace the four-stage workflow — it enriches each stage with technical-documentation-specific behaviors.

**This skill activates when the user wants to write:**
- Tutorials or how-to guides
- Architecture design documents
- API reference documentation
- Troubleshooting / operations manuals
- Technical concept explainers
- Technology comparison articles
- Best practice / style guides
- Release notes / changelogs

**Key phrases that should trigger:**
- "Write a tutorial for..."
- "Document the architecture of..."
- "Create API docs for..."
- "Write a troubleshooting guide..."
- "Explain what X is..."
- "Compare X vs Y..."
- "Write best practices for..."
- "Draft release notes for..."

## How This Plugin Works

```
Core Workflow:        Brainstorming → Planning → Execution → Review
                          ↑              ↑           ↑          ↑
Tech Doc Plugins:    [questions]    [templates]  [code/dia]  [accuracy]
                     tech-doc-       tech-doc-    tech-doc-   tech-doc-
                     writing         writing      code-blocks writing
                                                  tech-doc-
                                                  mermaid
                                                  tech-doc-
                                                  image-search
```

When the brainstorming skill detects a technical documentation intent, this plugin injects additional questions, provides document-type-specific templates during planning, triggers companion plugins during execution, and adds a fifth review pass for technical accuracy.

## Stage 1 Enhancement: Brainstorming

When brainstorming detects a technical writing intent, ask these **additional** questions (one at a time, after the standard brainstorming questions):

### Required Tech Questions

1. **Document type** — Present as multiple choice:
   - Tutorial / How-To Guide
   - Architecture Design Document
   - API Reference
   - Troubleshooting / Operations Manual
   - Technical Concept Explainer
   - Technology Comparison
   - Best Practice / Style Guide
   - Release Note / Changelog

2. **Target technology stack** — What technologies, frameworks, or tools does this document cover? Include version numbers if relevant.

3. **Reader technical level** — Multiple choice:
   - Beginner (knows basic programming, new to this technology)
   - Intermediate (has used the technology, needs deeper understanding)
   - Advanced (experienced practitioner, looking for edge cases or internals)
   - Mixed (document should serve multiple levels)

4. **Prerequisites** — What must the reader already know or have installed before they start?

5. **Code language preference** — What programming language(s) should code examples use?

### Optional Tech Questions (ask if relevant)

6. **Environment setup needed?** — Does the reader need to set up a local environment first?
7. **Operating system scope** — Linux only, macOS, Windows, or cross-platform?
8. **Existing documentation** — Are there existing docs to reference or update?

### Enhanced Brief Fields

Add these fields to the standard topic brief:

```markdown
## Technical Context

**Document Type:** [one of the 8 types]
**Technology Stack:** [technologies and versions]
**Reader Level:** [beginner/intermediate/advanced/mixed]
**Prerequisites:** [what readers need before starting]
**Code Language:** [primary language for examples]
**Environment Setup:** [yes/no — if yes, brief description]
**OS Scope:** [linux/macos/windows/cross-platform]
```

Save the brief with `doc_type` in the frontmatter:

```markdown
---
doc_type: tutorial
tech_stack: "Kubernetes v1.28, Helm v3"
reader_level: intermediate
code_language: bash, yaml
---
```

## Stage 2 Enhancement: Planning

When the brief contains a `doc_type` field, provide the matching template as the starting point for the article plan. The writer can modify the template, but it provides proven section structures.

### Template: Tutorial / How-To Guide

```markdown
## Section 1: Introduction (100-150 words)
- What you will build/achieve
- Why this approach
- Time estimate

## Section 2: Prerequisites (100-200 words)
- Required software and versions
- Required knowledge
- Environment setup instructions

## Section 3-N: Step-by-Step (200-400 words each)
- One logical step per section
- Each step: action → explanation → verification
- Include code blocks for every action
- Show expected output after each step

## Section N+1: Verification (150-250 words)
- How to confirm everything works
- Expected final state
- Common success indicators

## Section N+2: Troubleshooting (200-300 words)
- Common errors and fixes
- FAQ from real user experience
- Where to get help
```

### Template: Architecture Design Document

```markdown
## Section 1: Context & Problem (200-300 words)
- Business context
- Current state and pain points
- Why a new design is needed

## Section 2: Requirements (150-250 words)
- Functional requirements
- Non-functional requirements (performance, scale, security)
- Constraints and assumptions

## Section 3: Design Overview (200-400 words)
- High-level architecture diagram (Mermaid)
- Key design decisions
- Technology choices with rationale

## Section 4: Component Details (300-400 words per component)
- Component responsibility
- Interfaces and APIs
- Data flow
- Dependencies

## Section 5: Trade-offs & Alternatives (200-300 words)
- What was considered and rejected
- Pros/cons of chosen approach
- Known limitations

## Section 6: Deployment & Operations (200-300 words)
- Deployment strategy
- Monitoring and alerting
- Rollback plan
```

### Template: API Reference

```markdown
## Section 1: Overview (150-200 words)
- What this API does
- Base URL and versioning
- Rate limits and quotas

## Section 2: Authentication (200-300 words)
- Auth method (API key, OAuth, JWT)
- How to obtain credentials
- Example authenticated request

## Section 3-N: Endpoints (300-400 words each)
- HTTP method + path
- Description
- Parameters (path, query, body) with types
- Request example (curl + language SDK)
- Response example (success + error)
- Status codes

## Section N+1: Data Models (200-300 words)
- Object schemas with field descriptions
- Enum values
- Relationships between models

## Section N+2: Error Handling (150-250 words)
- Error response format
- Common error codes
- Troubleshooting guide
```

### Template: Troubleshooting / Operations Manual

```markdown
## Section 1: Overview (100-150 words)
- System/component being covered
- When to use this guide
- Escalation path

## Section 2-N: Problem Scenarios (250-400 words each)
- **Symptom:** What the user sees
- **Diagnosis:** How to investigate (commands, logs, metrics)
- **Root Cause:** Why this happens
- **Solution:** Step-by-step fix
- **Prevention:** How to avoid recurrence

## Section N+1: Monitoring & Alerting (200-300 words)
- Key metrics to watch
- Alert thresholds
- Dashboard links

## Section N+2: Runbook Quick Reference (150-200 words)
- Common commands cheat sheet
- Important file paths
- Contact information
```

### Template: Technical Concept Explainer

```markdown
## Section 1: What Is It? (150-250 words)
- One-sentence definition
- Analogy or mental model
- Where it fits in the bigger picture

## Section 2: Why Does It Matter? (150-250 words)
- Problems it solves
- Benefits over alternatives
- Real-world impact

## Section 3: How It Works (300-400 words)
- Core mechanism explained step by step
- Diagram showing internal flow (Mermaid)
- Key concepts and terminology

## Section 4: When To Use It (150-250 words)
- Good use cases
- Anti-patterns (when NOT to use it)
- Decision criteria

## Section 5: Hands-On Example (300-400 words)
- Minimal working example
- Walk through the code/config
- Expected output and explanation
```

### Template: Technology Comparison

```markdown
## Section 1: Introduction (150-200 words)
- What category of tools
- Why this comparison matters
- Scope and methodology

## Section 2: Evaluation Criteria (200-300 words)
- Criteria definitions and weighting
- How each criterion was assessed
- Disclaimers and biases

## Section 3: Feature Matrix (200-300 words)
- Comparison table (features x tools)
- Data sources for each claim
- Version/date of comparison

## Section 4-N: Deep Dives (250-400 words per technology)
- Strengths with evidence
- Weaknesses with evidence
- Best suited for [use case]
- Code/config example showing approach

## Section N+1: Recommendation (200-300 words)
- Summary matrix
- Recommendation by use case
- Migration considerations
```

### Template: Best Practice / Style Guide

```markdown
## Section 1: Introduction (100-150 words)
- Scope of the guide
- Who should follow this
- How to use this document

## Section 2-N: Principles (250-400 words each)
- **Principle:** Clear statement
- **Why:** Rationale
- **Do:** Correct example with code
- **Don't:** Anti-pattern with code
- **Exceptions:** When to break this rule

## Section N+1: Checklist (150-200 words)
- Actionable checklist for code review
- Quick-reference summary

## Section N+2: References (100-150 words)
- Source materials
- Related internal docs
- External standards
```

### Template: Release Note / Changelog

```markdown
## Section 1: Release Highlights (100-200 words)
- Version number and date
- One-paragraph summary
- Key headline features

## Section 2: Breaking Changes (200-300 words)
- What changed and why
- Migration steps with code examples
- Deprecation timeline

## Section 3: New Features (200-400 words)
- Feature description
- Usage example
- Configuration options

## Section 4: Bug Fixes (150-250 words)
- Issue description and fix
- Affected versions
- Workaround removal notes

## Section 5: Migration Guide (200-300 words)
- Step-by-step upgrade path
- Before/after code comparison
- Rollback instructions
```

## Stage 3 Enhancement: Execution

During section writing, automatically invoke companion plugins when appropriate:

### Plugin Trigger Rules

| Content Signal | Plugin to Invoke | Action |
|---|---|---|
| Section requires code examples | `tech-doc-code-blocks` | Generate formatted, validated code blocks |
| Section describes architecture/flow/relationships | `tech-doc-mermaid` | Generate appropriate Mermaid diagram |
| Section benefits from screenshots/visuals | `tech-doc-image-search` | Generate image placeholders with descriptions |
| Section references CLI commands | `tech-doc-code-blocks` | Format as shell blocks with expected output |
| Section describes data models | `tech-doc-mermaid` | Generate ER diagram |
| Section shows UI workflows | `tech-doc-image-search` | Add screenshot placeholders |

### Execution Self-Check (Additional)

After standard self-check, also verify:
- [ ] Code examples match the specified language preference
- [ ] Technical terms are defined on first use
- [ ] Version numbers are consistent throughout
- [ ] Prerequisites mentioned in body are listed in prerequisites section
- [ ] Commands include expected output
- [ ] All acronyms are expanded on first use

## Stage 4 Enhancement: Review

Add **Pass 5: Technical Accuracy** after the standard four passes (Logic, Evidence, Flow, Polish):

### Pass 5: Technical Accuracy

**Focus:** Is the technical content correct and usable?

**Questions to ask:**
- [ ] Are code examples syntactically correct?
- [ ] Do commands produce the described output?
- [ ] Are version numbers current and consistent?
- [ ] Are technical terms used correctly?
- [ ] Are prerequisites complete (nothing missing)?
- [ ] Are file paths and URLs valid?
- [ ] Are configuration values realistic (not placeholder)?
- [ ] Do diagrams accurately represent the described system?
- [ ] Are security implications mentioned where relevant?
- [ ] Is the technical depth appropriate for the stated reader level?

**Common issues:**
- Outdated version numbers
- Missing import/dependency in code examples
- Commands that work on one OS but not others
- Incorrect configuration syntax
- Missing error handling in examples
- Placeholder values left in production examples

**Log format:**
```markdown
### Pass 5: Technical Accuracy

**Issues Found:**

**[CRITICAL]** Section 3, code block 2
- **Issue:** Missing import statement
- **Current:** Code uses `requests.get()` without `import requests`
- **Fix:** Add `import requests` at top of code block

**[MODERATE]** Section 2, prerequisites
- **Issue:** Version mismatch
- **Current:** Says "Python 3.8+" but code uses walrus operator (3.8+) and match statement (3.10+)
- **Fix:** Update prerequisite to "Python 3.10+"

**[MINOR]** Section 5
- **Issue:** curl example missing Content-Type header
- **Fix:** Add `-H "Content-Type: application/json"`

**Technical Strengths:**
- Code examples are well-structured and progressive
- Diagrams accurately represent the architecture
```

## Enhanced Final Article Metadata

For technical documents, add these fields to the final article metadata:

```markdown
---
title: [Article Title]
date: YYYY-MM-DD
status: final
doc_type: [document type]
tech_stack: [technologies covered]
reader_level: [target level]
code_language: [languages used]
word_count: XXXX
reviewed: true
tech_reviewed: true
---
```

## Key Principles

- **Detect, don't force** — Only activate tech enhancements when writing intent is technical
- **Templates are starting points** — Writers can modify section structure freely
- **Companion plugins are automatic** — Code blocks, diagrams, and images are suggested without manual invocation
- **Technical accuracy is non-negotiable** — Pass 5 catches issues that general review misses
- **Reader level drives depth** — Beginner docs need more explanation; advanced docs need more edge cases
- **Version everything** — Technical docs go stale; always include version context
