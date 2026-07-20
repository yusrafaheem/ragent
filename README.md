# ragent

A retrieval-augmented, tool-using research-paper Q&A agent, built as two
independently deployable microservices rather than a single script — because
the interesting engineering problems in a real RAG system (index ownership,
service-to-service failure handling, horizontal scaling of stateless vs.
stateful components) only show up once retrieval and generation are
actually separated by a network boundary.

```
                         ┌─────────────────────┐
   user  ── HTTP ──────▶ │   agent service      │
  "what is HNSW?"        │   (stateless,        │
                         │    horizontally       │
                         │    scalable)          │
                         │                        │
                         │  ReAct loop:           │
                         │   Thought → Action →   │
                         │   Observation → ...    │
                         │   → Final Answer        │
                         └──────────┬─────────────┘
                                    │ HTTP (tool call)
                                    ▼
                         ┌─────────────────────┐
                         │  retrieval service    │
                         │  (owns the embedding   │
                         │   model + vector index)│
                         │                        │
                         │  /ingest  /search       │
                         └─────────────────────┘
```

The agent service holds no document index of its own. To answer a question
it calls the retrieval service's `/search` endpoint as a tool — the same way
it would call any other external API — which is what makes it stateless and
safe to run behind a horizontal pod autoscaler while the retrieval service
(which does hold state: an embedding model and a vector index in memory)
stays pinned to a single replica unless that index moves to a shared backing
store. See `infra/k8s/retrieval.yaml` and `infra/k8s/agent.yaml` for where
that distinction is encoded in the actual deployment config.

## Why this project exists

It's a companion to [vectorgrad](https://github.com/yusrafaheem/vectorgrad)
(a from-scratch autodiff engine, focused on the numerics/performance side of
ML systems). This one is focused on the *serving* side: retrieval-augmented
generation, agentic tool use, vector databases, and the container
orchestration and infrastructure-as-code needed to actually run that as a
production-shaped system rather than a notebook.

## Design decision: real backends with dependency-free fallbacks

Every ML-heavy component in this codebase — the embedding model, the vector
index, the LLM — is written against a small interface with two
implementations: a real one (sentence-transformers, FAISS, an
OpenAI-compatible chat API) and a zero-dependency reference implementation
(a hashing-trick bag-of-words embedder, a brute-force NumPy cosine-similarity
index, a deterministic rule-based ReAct policy). `get_embedder()`,
`get_vectorstore()`, and `get_llm_client()` each resolve to the real backend
when it's installed/configured and fall back otherwise, logging why.

This isn't just a defensive coding pattern — it's what makes the entire
system's core logic (chunking, retrieval ranking, the ReAct control flow,
tool dispatch, both FastAPI services) unit-testable **deterministically and
offline**, with no API key and no multi-hundred-MB model download required
to run the test suite. `StubLLM` in particular isn't a test mock bolted on
after the fact; it's a real (if simple) ReAct policy that understands the
agent's prompt format well enough to decide when to call a tool versus emit
a final answer, which is what lets `tests/test_agent.py` assert on exact
multi-step agent behavior instead of on non-reproducible model output.

The real backends (FAISS, sentence-transformers) are exercised for real in
CI's `test-real-backends` job, which installs them fresh from PyPI — see
[What's verified where](#whats-verified-where) below for the honest version
of this story, including where local development hit a real constraint.

## Running it locally

```bash
docker compose up --build
# retrieval service: http://localhost:8001
# agent service:     http://localhost:8000

python scripts/ingest_corpus.py --dir data/sample_papers
python scripts/demo.py "What does retrieval augmented generation do?"
```

Or without Docker:

```bash
pip install -e .
uvicorn apps.retrieval_service:app --port 8001 &
RETRIEVAL_SERVICE_URL=http://localhost:8001 uvicorn apps.agent_service:app --port 8000 &
python scripts/ingest_corpus.py
python scripts/demo.py "What is HNSW?"
```

By default this runs entirely offline against `StubLLM` and the NumPy/hashing
fallback backends — no API key needed. To use a real LLM:

```bash
export OPENAI_API_KEY=sk-...
# or point at any OpenAI-compatible endpoint (vLLM, Gemini's compat layer, etc.)
export OPENAI_BASE_URL=https://your-endpoint/v1
```

## Deploying it for real

**Kubernetes** (`infra/k8s/`): `kubectl apply -k infra/k8s` — a Namespace, a
shared ConfigMap/Secret, and a Deployment+Service per microservice. The agent
Deployment runs 2+ replicas behind a `HorizontalPodAutoscaler`; the retrieval
Deployment is intentionally pinned to 1 replica, with a comment in the
manifest explaining why (in-memory index, not yet backed by shared storage).

**AWS** (`infra/terraform/`): provisions two ECR repositories, an ECS Fargate
cluster running both services, an internet-facing ALB in front of the agent
service only (the retrieval service's security group only accepts traffic
from the agent service's security group — it's never reachable from outside
the cluster), and AWS Cloud Map for service discovery between the two ECS
services (Fargate tasks get a fresh IP on every deploy, so something has to
keep `retrieval.ragent.local` pointed at the current one).

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
# then build + push both images to the ECR repos in the output, and either
# re-run `terraform apply` with the new image tag or force a new deployment
```

## What's verified where

Being direct about this rather than letting a green CI badge imply more than
it does:

- `tests/test_chunking.py`, `test_embeddings.py`, `test_vectorstore.py`,
  `test_retriever.py`, `test_tools.py`, `test_llm.py`, `test_agent.py` — the
  core RAG + agent logic — run and pass locally with nothing installed
  beyond NumPy, and run in CI on Python 3.10/3.11/3.12.
- `tests/test_retrieval_api.py` and `test_agent_api.py` exercise the two
  FastAPI services via Starlette's `TestClient`. FastAPI/httpx are core
  dependencies, not optional extras, so these need `pip install -e .` to
  even import — the sandbox this project was originally developed in had no
  PyPI access, so these two files were written and reasoned through
  carefully but only actually executed for the first time in CI, not
  locally. The `test` job runs them on every PR alongside everything else.
- `tests/test_real_backends.py` exercises real FAISS and sentence-transformers
  usage, skipped (with a visible reason) unless those optional extras are
  installed. CI's `test-real-backends` job installs them from PyPI and runs
  it for real.
- `compose-smoke-test` in CI is the strongest end-to-end check: it builds
  both Docker images, brings them up with `docker compose`, and drives the
  actual system over HTTP — ingest a document into the retrieval service,
  ask the agent service a question about it, assert the answer cites the
  ingested document — proving the two containers can actually find and talk
  to each other, not just that each one individually builds and imports.
- `infra/k8s/*.yaml` is validated for structural correctness (every document
  has the required top-level Kubernetes fields) in CI; it has not been
  applied against a real cluster.
- `infra/terraform/*.tf` is checked with `terraform fmt -check` and
  `terraform validate` in CI (real HCL parsing and provider-schema
  validation, no AWS credentials needed for either); it has not been
  `terraform apply`'d against a real AWS account.

## Repository layout

```
src/ragent/         core package: chunking, embeddings, vectorstore,
                     retriever, llm, tools, agent (the ReAct loop)
apps/                the two FastAPI services (retrieval_service.py,
                     agent_service.py) -- thin HTTP wrappers around the
                     core package
infra/docker/        one Dockerfile per service
infra/k8s/            Kubernetes manifests (kubectl apply -k infra/k8s)
infra/terraform/     AWS ECS Fargate + ALB + Cloud Map + ECR IaC
tests/                 unit tests for every module above, plus the two
                     FastAPI services and the real-backend integrations
data/sample_papers/  a few short synthetic "papers" for local demoing
scripts/               ingest_corpus.py, demo.py -- CLI wrappers over the
                     running services' HTTP APIs
```
