# Component 3: FastAPI Backend and Persistence

## Purpose

Provide the production application boundary for ticket submission, AI
inference, persistence, and agent retrieval. The backend owns orchestration;
the frontend must not load model files or call model libraries directly.

## Scope

- Configure a SQLite database and SQLAlchemy models.
- Load the triage model and local LLM once at application startup.
- Validate ticket input and return typed API responses.
- Run urgency classification and response drafting in the correct order.
- Persist and retrieve tickets for the agent dashboard.

## Dependencies

```text
fastapi
uvicorn
sqlalchemy
pydantic
transformers
torch
bitsandbytes
```

## Implementation

### 1. Establish configuration

Create `backend/core/config.py` for environment-aware settings, including:

- Database URL.
- Paths to the triage and LLM artifacts.
- Device selection (`cuda` when available, otherwise CPU where supported).
- Generation limits and model loading options.
- CORS origins for the local frontend.

Do not hard-code machine-specific absolute paths. Resolve model paths relative
to the backend package or from explicit environment variables.

### 2. Configure the database

Create `backend/core/database.py` with the SQLite engine, session factory, and
dependency used by API routes. Create `backend/models/domain.py` with a
`Ticket` table containing at least:

```text
id
user_message
language
urgency_level
ai_draft_response
status
created_at
```

Use explicit types, indexes where useful, and a status vocabulary such as
`pending`, `approved`, and `sent`. Add `backend/models/schemas.py` for request
and response validation so database objects are not exposed implicitly.

### 3. Implement model services

Create `backend/services/ml_service.py` to:

- Load the triage checkpoint and vocabulary once.
- Validate artifact compatibility.
- Tokenize messages using the exported preprocessing configuration.
- Run inference under evaluation/no-gradient mode.
- Return the urgency value and label.

Create `backend/services/llm_service.py` to:

- Load the local tokenizer and model once.
- Use `BitsAndBytesConfig(load_in_4bit=True)` where supported.
- Keep generation behind a typed service method.
- Apply the shared prompt template and language instruction.
- Bound output length and return an explicit error if loading or generation
  fails.

Avoid broad exception handling or success-shaped fallback responses. A model
load failure must be visible during startup or in a clear API error.

### 4. Add application lifecycle management

Implement a FastAPI lifespan context manager in `backend/main.py`. It should
initialize the ML and LLM services once, store them in application state, and
release resources cleanly on shutdown. Do not load either model on every
request.

### 5. Implement API routes

Create `backend/api/routes.py` with:

#### `POST /api/tickets`

1. Validate the customer message and optional language.
2. Detect or confirm the input language.
3. Classify urgency with the triage service.
4. Generate an AI draft with the local LLM.
5. Persist the complete ticket in one database transaction.
6. Return a typed ticket response.

If a downstream model fails, return an explicit server error and avoid saving
an incomplete ticket unless the product later defines a deliberate partial
status.

#### `GET /api/tickets`

Return tickets with urgent items first, then newest items within the same
urgency level. Support pagination before the dataset becomes large. Return
only fields required by the dashboard.

Add an update endpoint for agent approval or sending only if the UI requires
it; its status transition must validate allowed state changes.

### 6. Test the backend

Add tests for:

- Request validation and empty messages.
- Database creation and ticket persistence.
- Route ordering by urgency and creation time.
- Model service initialization with test doubles.
- Failure propagation when a model cannot load.
- CORS and health behavior if those routes are added.

Use mocked model services in unit tests so the test suite does not download or
load the 7B model.

## Completion criteria

- The API starts only when required artifacts and configuration are valid.
- A ticket submission runs both AI services and persists the result.
- Ticket retrieval returns stable, urgency-first ordering.
- Models are loaded once per process.
- Tests cover validation, orchestration, persistence, and failure paths.

<!-- BEGIN MASTER PLAN SYNC -->
## Master plan alignment

> This section is generated from [smart-ticketing-system.md](./smart-ticketing-system.md). Do not edit it directly.

## Phase 3: Software Engineering (Backend & Database)

**Goal:** Build a robust, modular REST API that manages the database and serves the AI models efficiently.

* **Requirements & Libraries:** `fastapi`, `uvicorn`, `sqlalchemy`, `pydantic`, `transformers`, `torch`.

* **Tasks for AI Assistant:**
  * [ ] **Database Setup:** Create the `Ticket` table (id, user_message, language, urgency_level, ai_draft_response, status).
  * [ ] **Model Services (CRITICAL):**
    * Implement `ml_service.py` to load the PyTorch model and preprocessor.
    * Implement `llm_service.py` using `transformers` and `BitsAndBytesConfig(load_in_4bit=True)`.
  * [ ] **API Endpoints:**
    * `POST /api/tickets`: Accepts a user message. Calls ML service for urgency, calls LLM for draft response, saves to DB, returns JSON.
    * `GET /api/tickets`: Returns all tickets (sorted by urgency).
  * [ ] **Lifespan Manager:** Use FastAPI's lifespan context manager in `main.py` to load both models into GPU memory *once* on startup.

<!-- END MASTER PLAN SYNC -->
