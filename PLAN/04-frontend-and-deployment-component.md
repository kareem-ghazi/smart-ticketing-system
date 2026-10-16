# Component 4: React Interface and Full-Stack Execution

## Purpose

Deliver the customer and support-agent experiences and document how to run
the complete system locally. The interface must make AI output reviewable and
must not imply that a draft has been sent before an agent approves it.

## Scope

- Initialize the Vite React frontend with Tailwind CSS.
- Build the customer ticket form.
- Build the agent dashboard and urgency visualization.
- Integrate the typed backend API.
- Provide local startup, model placement, and verification instructions.

## Dependencies

```text
react
vite
tailwindcss
axios
lucide-react
```

## Implementation

### 1. Set up the frontend

Create the Vite application under `frontend/`. Configure Tailwind CSS and
centralize the backend base URL in an environment variable such as
`VITE_API_URL`. Keep API access in `frontend/src/services/api.js`; components
should not duplicate Axios configuration.

### 2. Build the customer view

Create `frontend/src/components/CustomerForm.jsx` with:

- A required textarea for the customer message.
- Clear validation for empty or excessively long input.
- Optional language selection if automatic detection is not sufficient.
- A loading state while the ticket is processed.
- A success state showing the created ticket ID and draft status.
- An error state that explains that submission failed without exposing
  internal stack traces.

The form must submit to `POST /api/tickets` and reset only after a successful
response.

### 3. Build the agent dashboard

Create `frontend/src/components/AgentDashboard.jsx` that:

- Fetches tickets from `GET /api/tickets`.
- Displays message, language, urgency, status, timestamp, and AI draft.
- Uses red or similarly prominent styling for `Urgent` and neutral styling for
  `Normal`.
- Separates the original customer message from the generated draft.
- Provides an `Approve & Send` action only when supported by a backend status
  endpoint.
- Shows loading, empty, refresh, and error states.

Use accessible labels, keyboard-friendly controls, and responsive layout for
Arabic right-to-left content as well as English left-to-right content.

### 4. Wire the application shell

Update `frontend/src/App.jsx` to present the customer form and agent
dashboard. Keep reusable status badges, ticket cards, and language-direction
logic in small components where that improves clarity. Use
`frontend/src/main.jsx` for the application entry point and global styles.

### 5. Verify frontend behavior

Test at minimum:

- Successful ticket submission.
- Validation of empty input.
- API failure rendering.
- Urgent and normal visual states.
- Arabic text direction and display.
- Dashboard empty and loading states.
- Approval control visibility based on ticket status.

Use mocked API responses for component tests, and manually verify the full
flow against the running FastAPI service.

## Local execution strategy

1. Train the triage model and place its exported artifacts in
   `backend/saved_models/`.
2. Fine-tune the local LLM and place the tokenizer/model files in
   `backend/saved_models/llm-finetuned/`.
3. Configure the backend environment variables and initialize the SQLite
   database.
4. Start the backend:

   ```text
   cd backend
   uvicorn main:app --reload
   ```

5. Start the frontend in a second terminal:

   ```text
   cd frontend
   npm run dev
   ```

6. Open the Vite URL, normally `http://localhost:5173`.
7. Submit one English and one Arabic ticket, then confirm that urgency,
   generated draft, persistence, and dashboard ordering all work.

## Operational checks

- Confirm model paths are local and no network download occurs at runtime.
- Confirm the backend logs model-load failures clearly.
- Confirm CORS allows only the configured frontend origin.
- Confirm SQLite data survives a backend restart.
- Confirm the UI distinguishes an AI draft from an approved or sent response.

## Completion criteria

- Customers can submit bilingual tickets from the browser.
- Agents can view urgency-sorted tickets and review drafts.
- Frontend states cover success, loading, empty, and failure cases.
- The documented commands start both services.
- An end-to-end smoke test verifies submission through persistence and display.

<!-- BEGIN MASTER PLAN SYNC -->
## Master plan alignment

> This section is generated from [smart-ticketing-system.md](./smart-ticketing-system.md). Do not edit it directly.

## Phase 4: Frontend Web UI (React)

**Goal:** Build the user interface for customers to submit tickets and agents to manage them.

* **Requirements:** `react` (via Vite), `tailwindcss`, `axios`, `lucide-react` (icons).

* **Tasks for AI Assistant:**
  * [ ] **Setup:** Initialize a Vite React project with TailwindCSS.
  * [ ] **Customer View (Component):** Create a simple form (`<textarea>`) for users to submit a complaint. Show a success message upon submission.
  * [ ] **Agent Dashboard (Component):**
    * Fetch tickets from `GET /api/tickets`.
    * Display them in a grid or list.
    * Apply color-coding based on the PyTorch urgency label (e.g., Red for `Urgent`, Gray for `Normal`).
    * Display the LLM's `ai_draft_response` with an "Approve & Send" button.
  * [ ] **API Integration:** Create an `api.js` file using `axios` to handle the backend communication.

## Phase 5: Deployment & Execution Strategy

**How to run the full stack (Instructions for the developer):**

1. Train ML model and place `triage_model.pth` in `backend/saved_models/`.
2. Train LLM and place adapter/merged model in `backend/saved_models/llm-finetuned/`.
3. Open terminal 1: `cd backend && uvicorn main:app --reload`
4. Open terminal 2: `cd frontend && npm run dev`
5. Access the app via `localhost:5173`

<!-- END MASTER PLAN SYNC -->
