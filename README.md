# Smart Loan Underwriter

An advanced, multi-agent AI system for dynamic loan underwriting. Built with LangGraph, it coordinates specialized AI agents mimicking distinct departments (Quantitative, Risk, and Legal) to evaluate complex loan applications autonomously. It features a robust FastAPI backend with SQL persistence.

## Key Features

- Multi-Agent Supervisor Architecture: Instead of a monolithic prompt, a Supervisor router evaluates the application state and dynamically delegates tasks to three specialized worker agents.
- Financial Capacity Agent (Quantitative Engine): Utilizes a sandboxed Python REPL tool to execute complex and deterministic financial math (EMI, DTI, buffer) rather than hallucinating arithmetic.
- Credit Risk Agent (Truth Verifier): A specialized investigator that employs mock bureau and employment deterministic APIs to evaluate behavioral history (credit score, past defaults, job stability).
- Compliance & Policy Agent (Legal Gatekeeper): Enforces KYC/AML checks against an internal policy checker and utilizes live web searches (duckduckgo-search) to verify adherence to current lending regulations.
- Intelligent Routing & Short-Circuit Logic: The Supervisor identifies critical flaws early (e.g., unacceptable DTI or hard AML failure) and halts execution, drastically saving LLM token costs limit delays.
- State Persistence (Checkpoints): Built-in sqlite3 integration to persist the LangGraph's state between API requests and server restarts.

## Tech Stack

- Orchestration: LangGraph, LangChain, Google Gemini API
- Backend API: FastAPI, Pydantic, Uvicorn, SQLite
- Agent Tools: PythonREPL, DuckDuckGoSearchRun, Custom deterministic mock tools

## Getting Started

Ensure you have your LLM API keys set in a .env file (e.g., GOOGLE_API_KEY=...).

1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Start the Backend API (FastAPI)
Run the API on port 8000. It manages persistence and exposes the /v1/execute endpoint.
```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Project Structure

- main.py - FastAPI application, Pydantic models, API endpoints, token tracking, and database checkpoint initialization.
- graph.py - Core LangGraph logic, Agent Nodes, Supervisor Prompts, and AI Tool implementations.
- EVAL.md - Extensive project analysis, architectural justifications, LLM failure mitigation logic, and system stress-testing evaluations.
- test_cases.json - High-complexity pre-configured testing profiles designed to push the multi-agent orchestration limits (gray areas, conflicts, logic traps). 
