"""
main.py — FastAPI Service for Smart Loan Underwriter
=====================================================
Production-ready API wrapping the LangGraph supervisor system with
trace logging, token tracking, and SQLite-backed state persistence.
"""

import os
import uuid
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph

load_dotenv()

# ── Persistence ──────────────────────────────────────────────────────

DB_PATH = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")

def _get_checkpointer():
    """Create a SQLite checkpointer that persists across restarts."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)

# ── Application Lifespan ─────────────────────────────────────────────

checkpointer = None
compiled_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, compiled_graph
    checkpointer = _get_checkpointer()
    compiled_graph = build_graph(checkpointer=checkpointer)
    print(f"✅ Graph compiled. Checkpoints persisted at: {DB_PATH}")
    yield
    if checkpointer and hasattr(checkpointer, "conn"):
        checkpointer.conn.close()
    print("🛑 Shutting down — checkpoint connection closed.")

# ── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Loan Underwriter",
    description=(
        "A LangGraph-powered multi-agent system that coordinates Financial Capacity, "
        "Credit & Risk, and Compliance & Policy agents via a Supervisor to produce "
        "comprehensive loan underwriting decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request / Response Schemas ───────────────────────────────────────

class LoanApplication(BaseModel):
    applicant_name: str = Field(..., examples=["Ravi Sharma"])
    age: int = Field(..., ge=1, examples=[35])
    annual_income: float = Field(..., gt=0, examples=[1_200_000])
    monthly_expenses: float = Field(..., ge=0, examples=[35_000])
    existing_debts: float = Field(0, ge=0, description="Monthly debt payments", examples=[10_000])
    savings: float = Field(0, ge=0, examples=[500_000])
    loan_amount_requested: float = Field(..., gt=0, examples=[2_000_000])
    loan_purpose: str = Field(..., examples=["home"])
    loan_term_months: int = Field(..., gt=0, examples=[240])
    employment_type: str = Field("salaried", examples=["salaried"])
    industry: str = Field("technology", examples=["technology", "crypto", "healthcare"])
    years_employed: float = Field(0, ge=0, examples=[8.0])
    additional_info: str = Field("", examples=["First-time home buyer"])


class ExecuteRequest(BaseModel):
    loan_application: LoanApplication
    thread_id: Optional[str] = Field(
        None,
        description="Reuse a thread ID to resume a previous session (persistence).",
    )


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ExecuteResponse(BaseModel):
    thread_id: str
    final_decision: str
    financial_analysis: str
    credit_risk_analysis: str
    compliance_analysis: str
    execution_trace: list[dict]
    token_usage: TokenUsage
    duration_seconds: float


# ── Endpoints ────────────────────────────────────────────────────────



def _parse_gemini_text(content) -> str:
    """Helper to extract plain text from Gemini's complex response blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # If Gemini returns a list of blocks, extract the text from the 'text' keys
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and "text" in block
        )
    return str(content)

@app.post("/v1/execute", response_model=ExecuteResponse, tags=["Underwriting"])
def execute_loan_underwriting(req: ExecuteRequest):
    """
    Submit a loan application for full multi-agent underwriting.

    Full execution trace and token usage are returned.
    """
    if compiled_graph is None:
        raise HTTPException(503, "Graph not initialized yet.")

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [],
        "loan_application": req.loan_application.model_dump(),
        "next_agent": "",
        "financial_analysis": "",
        "credit_risk_analysis": "",
        "compliance_analysis": "",
        "final_decision": "",
        "execution_trace": [],
    }

    start = datetime.now()

    try:
        final_state = compiled_graph.invoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(500, f"Graph execution failed: {e}")

    elapsed = (datetime.now() - start).total_seconds()

    # Token Aggregation (Gemini)
    prompt_tokens = 0
    completion_tokens = 0
    
    for trace_entry in final_state.get("execution_trace", []):
        usage = trace_entry.get("token_usage", {})
        prompt_tokens += usage.get("input_tokens", 0)
        completion_tokens += usage.get("output_tokens", 0)

    # Capture final supervisor decision tokens if available
    msgs = final_state.get("messages", [])
    if msgs and hasattr(msgs[-1], "usage_metadata") and msgs[-1].usage_metadata:
        prompt_tokens += msgs[-1].usage_metadata.get("input_tokens", 0)
        completion_tokens += msgs[-1].usage_metadata.get("output_tokens", 0)

    total_tokens = prompt_tokens + completion_tokens
    
    # Cost estimation (gemini-3.1-flash-lite-preview pricing: ~$0.075 / 1M input, $0.30 / 1M output)
    estimated_cost = (prompt_tokens * 0.075 / 1_000_000) + (completion_tokens * 0.30 / 1_000_000)

    return ExecuteResponse(
        thread_id=thread_id,
        final_decision=_parse_gemini_text(final_state.get("final_decision", "No decision reached.")),
        financial_analysis=_parse_gemini_text(final_state.get("financial_analysis", "")),
        credit_risk_analysis=_parse_gemini_text(final_state.get("credit_risk_analysis", "")),
        compliance_analysis=_parse_gemini_text(final_state.get("compliance_analysis", "")),
        execution_trace=final_state.get("execution_trace", []),
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
        ),
        duration_seconds=round(elapsed, 2),
    )


@app.get("/v1/health", tags=["System"])
def health_check():
    """Liveness probe."""
    return {
        "status": "healthy",
        "graph_ready": compiled_graph is not None,
        "checkpoint_db": DB_PATH,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/v1/thread/{thread_id}", tags=["Persistence"])
def get_thread_state(thread_id: str):
    """Retrieve the persisted state for a given thread (proves persistence works)."""
    if compiled_graph is None:
        raise HTTPException(503, "Graph not initialized.")
    try:
        state = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
        if state is None or state.values is None:
            raise HTTPException(404, f"No state found for thread {thread_id}")
        vals = state.values
        return {
            "thread_id": thread_id,
            "has_final_decision": bool(vals.get("final_decision")),
            "financial_done": bool(vals.get("financial_analysis")),
            "credit_done": bool(vals.get("credit_risk_analysis")),
            "compliance_done": bool(vals.get("compliance_analysis")),
            "final_decision_preview": (vals.get("final_decision") or "")[:500],
            "trace_steps": len(vals.get("execution_trace", [])),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error retrieving state: {e}")


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
