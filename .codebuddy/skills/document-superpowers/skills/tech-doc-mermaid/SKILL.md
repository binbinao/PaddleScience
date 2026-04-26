---
name: tech-doc-mermaid
description: "Enhancement plugin for generating Mermaid diagrams in technical documentation. Provides diagram type selection, syntax templates, and validation. Invoked automatically by tech-doc-writing during execution stage."
---

# Tech Doc Mermaid

## Overview

Enhancement plugin that generates Mermaid diagrams during the execution stage of technical document writing. Diagrams visualize architectures, flows, relationships, and processes that are difficult to explain with text alone.

**Invoked by:** `tech-doc-writing` during execution stage, or manually when a section needs visual explanation.

**Trigger signals:**
- Section describes system architecture or component relationships
- Section explains a multi-step process or workflow
- Section covers request/response flows between services
- Section documents data models or state transitions
- Text says "as shown in the diagram" or similar

## When to Use Diagrams

**Use a diagram when:**
- The concept involves 3+ interacting components
- A process has branching logic or multiple paths
- The relationship between entities needs visual clarity
- Text alone would require 200+ words to describe what a diagram shows in seconds

**Skip a diagram when:**
- The concept is linear and simple (use a numbered list instead)
- A table would communicate the information better (e.g., feature comparisons)
- The diagram would have fewer than 3 nodes (too trivial)
- The diagram would have more than 15-20 nodes (too complex — split into multiple diagrams)

## Diagram Type Selection Guide

| Content You're Documenting | Recommended Diagram | Mermaid Type |
|---|---|---|
| System architecture / component layout | Architecture diagram | `flowchart` or `graph` |
| Request flow between services | Sequence diagram | `sequenceDiagram` |
| Object lifecycle / status transitions | State diagram | `stateDiagram-v2` |
| Database schema / entity relationships | ER diagram | `erDiagram` |
| CI/CD pipeline / deployment flow | Flow diagram with subgraphs | `flowchart` |
| Decision logic / troubleshooting tree | Flowchart with diamonds | `flowchart` |
| Class hierarchy / interface relationships | Class diagram | `classDiagram` |
| Project timeline / release roadmap | Timeline | `timeline` |
| Git branching strategy | Git graph | `gitGraph` |
| User journey / experience flow | User journey | `journey` |

## Document Type to Diagram Mapping

Each document type benefits from specific diagram types:

| Document Type | Primary Diagram | Secondary Diagram |
|---|---|---|
| Tutorial / How-To | Step flowchart | — |
| Architecture | Component flowchart | Sequence diagram |
| API Reference | Sequence diagram | ER diagram |
| Troubleshooting | Decision flowchart | — |
| Concept Explainer | Flowchart / State diagram | — |
| Comparison | — (use tables instead) | — |
| Best Practice | — (use code examples instead) | — |
| Release Note | Timeline | — |

## Mermaid Syntax Best Practices

### Mandatory Rules

1. **Node IDs: camelCase, no spaces**

Good:
```mermaid
flowchart LR
    apiGateway[API Gateway] --> userService[User Service]
```

Bad:
```
flowchart LR
    API Gateway --> User Service
```

2. **Special characters in labels: use double quotes**

Good:
```mermaid
flowchart LR
    loadBalancer["Load Balancer (nginx)"] --> app["App Server: Port 8080"]
```

3. **Avoid reserved words as node IDs**

Reserved: `end`, `subgraph`, `graph`, `flowchart`, `style`, `class`, `click`

Good:
```mermaid
flowchart LR
    processEnd[End] --> cleanup[Cleanup]
```

4. **Subgraphs: use `subgraph id [Label]` format**

```mermaid
flowchart TB
    subgraph backend [Backend Services]
        apiServer[API Server]
        workerQueue[Worker Queue]
    end
    subgraph dataLayer [Data Layer]
        postgres[(PostgreSQL)]
        redis[(Redis Cache)]
    end
    apiServer --> postgres
    apiServer --> redis
    workerQueue --> postgres
```

5. **No inline styling** — Let the rendering theme handle colors

Bad:
```
style apiServer fill:#f9f,stroke:#333
```

6. **Edge labels with special characters: wrap in quotes**

Good:
```mermaid
flowchart LR
    A -->|"O(n) complexity"| B
    C -->|"Step 1: Validate"| D
```

## Diagram Templates

### Architecture Diagram

```mermaid
flowchart TB
    client[Client App] --> lb[Load Balancer]
    
    subgraph backend [Backend Cluster]
        lb --> apiA[API Server A]
        lb --> apiB[API Server B]
    end
    
    subgraph dataStores [Data Stores]
        postgres[(PostgreSQL)]
        redis[(Redis)]
        s3[(S3 Storage)]
    end
    
    apiA --> postgres
    apiA --> redis
    apiB --> postgres
    apiB --> redis
    apiA --> s3
    apiB --> s3
    
    subgraph external [External Services]
        emailService[Email Service]
        paymentGateway[Payment Gateway]
    end
    
    apiA --> emailService
    apiB --> paymentGateway
```

### Sequence Diagram (API Request Flow)

```mermaid
sequenceDiagram
    participant C as Client
    participant G as API Gateway
    participant A as Auth Service
    participant U as User Service
    participant D as Database

    C->>G: POST /api/users
    G->>A: Validate token
    A-->>G: Token valid
    G->>U: Create user request
    U->>D: INSERT user
    D-->>U: User created
    U-->>G: 201 Created
    G-->>C: 201 + user object
```

### State Diagram (Object Lifecycle)

```mermaid
stateDiagram-v2
    [*] --> Draft
    Draft --> PendingReview: Submit
    PendingReview --> InReview: Reviewer assigned
    InReview --> ChangesRequested: Request changes
    InReview --> Approved: Approve
    ChangesRequested --> PendingReview: Resubmit
    Approved --> Published: Publish
    Published --> Archived: Archive
    Archived --> [*]
```

### ER Diagram (Data Model)

```mermaid
erDiagram
    USER ||--o{ ORDER : places
    USER {
        uuid id PK
        string email UK
        string name
        timestamp created_at
    }
    ORDER ||--|{ ORDER_ITEM : contains
    ORDER {
        uuid id PK
        uuid user_id FK
        decimal total
        string status
        timestamp created_at
    }
    ORDER_ITEM {
        uuid id PK
        uuid order_id FK
        uuid product_id FK
        int quantity
        decimal price
    }
    PRODUCT ||--o{ ORDER_ITEM : "included in"
    PRODUCT {
        uuid id PK
        string name
        decimal price
        int stock
    }
```

### Decision Flowchart (Troubleshooting)

```mermaid
flowchart TD
    start[Service Not Responding] --> checkPod{Pod running?}
    checkPod -->|No| restartPod[Restart pod]
    checkPod -->|Yes| checkLogs{Errors in logs?}
    checkLogs -->|Yes| analyzeError[Analyze error message]
    checkLogs -->|No| checkNetwork{Network reachable?}
    checkNetwork -->|No| checkDNS[Check DNS and service mesh]
    checkNetwork -->|Yes| checkResources{Resource limits hit?}
    checkResources -->|Yes| scaleUp[Increase resource limits]
    checkResources -->|No| escalate[Escalate to platform team]
    
    restartPod --> verify{Service healthy?}
    analyzeError --> applyFix[Apply fix based on error]
    applyFix --> verify
    checkDNS --> verify
    scaleUp --> verify
    verify -->|Yes| resolved[Resolved]
    verify -->|No| escalate
```

### CI/CD Pipeline

```mermaid
flowchart LR
    commit[Git Push] --> lint[Lint & Format]
    lint --> unitTest[Unit Tests]
    unitTest --> build[Build Image]
    build --> integTest[Integration Tests]
    
    integTest --> stagingDeploy[Deploy to Staging]
    stagingDeploy --> e2eTest[E2E Tests]
    e2eTest --> approval{Manual Approval}
    approval -->|Approved| prodDeploy[Deploy to Production]
    approval -->|Rejected| fixIssues[Fix Issues]
    fixIssues --> commit
    
    prodDeploy --> smokeTest[Smoke Tests]
    smokeTest --> monitor[Monitor Metrics]
```

### Timeline (Release Roadmap)

```mermaid
timeline
    title Product Roadmap 2025
    section Q1
        v2.0 : Core API redesign
             : Breaking changes migration
    section Q2
        v2.1 : Plugin system
             : Webhook support
    section Q3
        v2.2 : Dashboard UI
             : Real-time analytics
    section Q4
        v3.0 : Multi-tenant support
             : Enterprise features
```

### Git Graph (Branching Strategy)

```mermaid
gitGraph
    commit id: "init"
    branch develop
    commit id: "feat-base"
    branch feature/auth
    commit id: "add-auth"
    commit id: "add-jwt"
    checkout develop
    merge feature/auth id: "merge-auth"
    branch feature/api
    commit id: "add-endpoints"
    checkout develop
    merge feature/api id: "merge-api"
    checkout main
    merge develop id: "release-v1.0" tag: "v1.0"
```

## Diagram Placement Guidelines

- Place the diagram **immediately after** the text that introduces it
- Add a brief caption below the diagram explaining what it shows
- Reference the diagram in surrounding text: "As shown in the diagram below..."
- If a diagram is referenced from multiple sections, place it in the first reference and use text references from later sections

**Example placement:**

```markdown
The system follows a standard three-tier architecture. The API gateway
handles authentication before forwarding requests to the backend cluster.

```mermaid
flowchart LR
    client --> gateway --> backend --> database
```

*Figure 1: High-level request flow from client to database.*

Each component in this architecture runs as a separate Kubernetes deployment...
```

## Complexity Guidelines

| Diagram Size | Nodes | Recommended Action |
|---|---|---|
| Simple | 3-6 | Single diagram, inline |
| Medium | 7-12 | Single diagram with subgraphs |
| Complex | 13-20 | Consider splitting into 2 focused diagrams |
| Too complex | 20+ | Must split — overview diagram + detail diagrams |

## Quality Checklist

Before finalizing a section with Mermaid diagrams, verify:

- [ ] Diagram renders without syntax errors
- [ ] Node IDs use camelCase (no spaces)
- [ ] Labels with special characters are quoted
- [ ] No reserved words used as node IDs
- [ ] No inline styling or color definitions
- [ ] Diagram has 3-15 nodes (not too simple, not too complex)
- [ ] Labels are concise (2-4 words per node)
- [ ] Diagram accurately represents the described system
- [ ] A brief caption or description accompanies the diagram
- [ ] Diagram type matches the content (not forcing a flowchart when a sequence diagram fits better)

## Key Principles

- **Diagram complements text** — It should clarify, not duplicate what the text says
- **Right diagram for the job** — Use the selection guide; don't default to flowcharts for everything
- **Simplicity wins** — If you need more than 15 nodes, split into multiple diagrams
- **No styling** — Let the theme handle colors; diagrams should work in light and dark mode
- **Accurate over pretty** — The diagram must correctly represent the system
- **Caption everything** — A diagram without context is a puzzle
