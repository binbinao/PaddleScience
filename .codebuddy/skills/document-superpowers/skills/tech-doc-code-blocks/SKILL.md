---
name: tech-doc-code-blocks
description: "Enhancement plugin for generating high-quality code blocks in technical documentation. Ensures code examples are properly formatted, syntactically correct, runnable, and progressively complex. Invoked automatically by tech-doc-writing during execution stage."
---

# Tech Doc Code Blocks

## Overview

Enhancement plugin that generates, formats, and validates code blocks during the execution stage of technical document writing. Ensures every code example is properly annotated, syntactically correct, and follows progressive complexity patterns.

**Invoked by:** `tech-doc-writing` during execution stage, or manually when any section needs code examples.

**Trigger signals:**
- Section plan mentions code examples, commands, or configuration
- Writer is documenting an API, CLI tool, library, or framework
- Section includes installation, setup, or usage instructions

## Code Block Standards

Every code block in a technical document MUST follow these rules:

### Rule 1: Language Tag Required

Every fenced code block must include a language identifier.

**Correct:**
````markdown
```python
print("hello")
```
````

**Wrong:**
````markdown
```
print("hello")
```
````

**Common language tags:** `python`, `javascript`, `typescript`, `go`, `java`, `rust`, `bash`, `shell`, `sql`, `yaml`, `json`, `toml`, `xml`, `html`, `css`, `dockerfile`, `hcl`, `graphql`, `protobuf`, `markdown`

### Rule 2: Comments Explain Why, Not What

```python
# Rate-limit to avoid hitting API quota (100 req/min)
time.sleep(0.6)
```

Not:
```python
# Sleep for 0.6 seconds
time.sleep(0.6)
```

### Rule 3: Shell Commands Show Expected Output

Use separate blocks for input and output, or annotate within the block:

```bash
$ kubectl get pods -n production
NAME                    READY   STATUS    RESTARTS   AGE
api-server-7d4b8c6f9-x2k4m   1/1     Running   0          3h
worker-5c8d9f7b2-p9n3q       1/1     Running   0          3h
```

For commands without meaningful output, indicate success:

```bash
$ docker build -t myapp:latest .
# ... build output ...
Successfully built a1b2c3d4e5f6
```

### Rule 4: Configuration Files Annotate Key Fields

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
spec:
  replicas: 3                    # Scale based on expected load
  selector:
    matchLabels:
      app: api-server
  template:
    spec:
      containers:
      - name: api
        image: myapp:v2.1.0      # Pin to specific version, not :latest
        resources:
          limits:
            memory: "512Mi"      # Based on profiling — OOMs below 256Mi
            cpu: "500m"
```

### Rule 5: Imports and Dependencies Are Explicit

Show complete, runnable code. Do not assume the reader knows what to import.

```python
import os
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")
```

## Progressive Complexity Pattern

For each concept, provide examples in increasing complexity:

### Level 1: Minimal Example

The simplest possible working code. Strips away error handling, configuration, and edge cases.

```python
import requests

response = requests.get("https://api.example.com/users")
users = response.json()
print(users)
```

### Level 2: Production Example

Adds error handling, configuration, logging, and real-world considerations.

```python
import logging
import os
from typing import list

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

def get_users(base_url: str, api_key: str) -> list[dict]:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        response = session.get(
            f"{base_url}/users",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch users: %s", e)
        raise
```

### Level 3: Advanced Example (Optional)

Covers edge cases, performance optimization, or advanced patterns.

```python
import asyncio
from typing import AsyncIterator

import httpx

async def get_users_paginated(
    base_url: str,
    api_key: str,
    page_size: int = 100,
) -> AsyncIterator[dict]:
    """Stream users with async pagination to handle large datasets."""
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    ) as client:
        cursor = None
        while True:
            params = {"limit": page_size}
            if cursor:
                params["cursor"] = cursor

            response = await client.get("/users", params=params)
            response.raise_for_status()
            data = response.json()

            for user in data["items"]:
                yield user

            cursor = data.get("next_cursor")
            if not cursor:
                break
```

## Code Block Type Templates

### Shell Command Block

```bash
# Install dependencies
$ pip install requests python-dotenv

# Set environment variable
$ export API_KEY="your-api-key-here"

# Run the script
$ python fetch_users.py
Fetched 42 users successfully.
```

### Configuration File Block

```yaml
# config.yaml — Application configuration
# See https://docs.example.com/config for all options

server:
  host: 0.0.0.0
  port: 8080
  workers: 4            # Set to CPU count for production

database:
  url: postgresql://user:pass@localhost:5432/mydb
  pool_size: 20          # Max concurrent connections
  pool_timeout: 30       # Seconds to wait for connection

logging:
  level: INFO            # DEBUG in development, INFO in production
  format: json           # Use 'text' for local development
```

### API Request/Response Block

```bash
# Create a new user
$ curl -X POST https://api.example.com/v1/users \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jane Smith",
    "email": "jane@example.com",
    "role": "admin"
  }'
```

Response (`201 Created`):
```json
{
  "id": "usr_a1b2c3d4",
  "name": "Jane Smith",
  "email": "jane@example.com",
  "role": "admin",
  "created_at": "2025-01-15T10:30:00Z"
}
```

Error response (`409 Conflict`):
```json
{
  "error": {
    "code": "USER_EXISTS",
    "message": "A user with this email already exists",
    "details": {
      "field": "email",
      "value": "jane@example.com"
    }
  }
}
```

### Dockerfile Block

```dockerfile
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8080
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8080/health || exit 1

CMD ["gunicorn", "app:create_app()", "-b", "0.0.0.0:8080", "-w", "4"]
```

### Diff / Before-After Block

Show changes clearly using diff syntax or side-by-side comparison:

```diff
- DATABASE_URL=sqlite:///local.db
+ DATABASE_URL=postgresql://user:pass@db:5432/production
 
- DEBUG=true
+ DEBUG=false
 
+ SENTRY_DSN=https://key@sentry.io/12345
```

Or use explicit before/after sections:

**Before:**
```python
data = json.loads(response.text)
```

**After:**
```python
try:
    data = response.json()
except requests.exceptions.JSONDecodeError:
    logger.error("Invalid JSON response: %s", response.text[:200])
    raise
```

## Quality Checklist

Before finalizing any section with code blocks, verify:

- [ ] Every code block has a language tag
- [ ] Shell commands use `$` prefix to distinguish input from output
- [ ] All imports and dependencies are shown
- [ ] Comments explain *why*, not *what*
- [ ] Configuration values are realistic (not `TODO` or `xxx`)
- [ ] Error responses are shown alongside success responses (for API docs)
- [ ] Code is syntactically correct for the specified language
- [ ] Environment variables and secrets use placeholder patterns (`$API_KEY`, `your-api-key-here`)
- [ ] Progressive complexity is used when explaining a concept (minimal → production)
- [ ] Long output is truncated with `# ... output ...` to keep focus

## Anti-Patterns

**Do NOT:**
- Show code without language tags
- Use `foo`, `bar`, `baz` as names in production examples
- Leave `TODO` or `FIXME` in published code blocks
- Show only the happy path without error handling
- Assume the reader knows which package to install
- Mix different languages in one code block without explanation
- Show outdated syntax or deprecated APIs

## Key Principles

- **Runnable over elegant** — Readers will copy-paste; make it work
- **Progressive disclosure** — Simple first, complex later
- **Complete context** — Show imports, setup, and teardown
- **Realistic values** — Use plausible data, not placeholders
- **Output matters** — Show what the reader should expect to see
- **Version-aware** — Note which version of a tool/library the code targets
