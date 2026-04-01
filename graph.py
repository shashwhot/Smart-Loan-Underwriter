"""
graph.py — Smart Loan Underwriter: Multi-Agent Supervisor Architecture
======================================================================
LangGraph StateGraph with a Supervisor pattern coordinating three
specialized worker agents for comprehensive loan underwriting.
"""

import os
import time
import json
import hashlib
import operator
from typing import Annotated, TypedDict, Literal
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# ── Configuration ────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite-preview")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── State Schema ─────────────────────────────────────────────────────

class LoanUnderwriterState(TypedDict):
    """Shared state for the loan underwriting supervisor graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    loan_application: dict
    next_agent: str
    financial_analysis: str
    credit_risk_analysis: str
    compliance_analysis: str
    final_decision: str
    execution_trace: Annotated[list[dict], operator.add]


# ── Supervisor Routing Schema ────────────────────────────────────────

AGENT_NAMES = [
    "financial_capacity_agent",
    "credit_risk_agent",
    "compliance_policy_agent",
]


class SupervisorDecision(BaseModel):
    """The supervisor's routing decision."""
    next: Literal[
        "financial_capacity_agent",
        "credit_risk_agent",
        "compliance_policy_agent",
        "FINISH",
    ] = Field(description="Next agent to delegate to, or FINISH.")
    reasoning: str = Field(description="Why this agent was chosen or why finishing.")


# ── Deterministic Mock-Data Helper ───────────────────────────────────

def _name_hash(name: str) -> int:
    return int(hashlib.sha256(name.lower().strip().encode()).hexdigest(), 16)


# ── Tools: Credit & Risk Agent ──────────────────────────────────────

@tool
def check_credit_score(applicant_name: str) -> str:
    """Query the Indian credit bureau (CIBIL/Experian) for the applicant's credit score & history."""
    h = _name_hash(applicant_name)
    scores = [560, 590, 620, 645, 670, 700, 725, 750, 780, 820]
    score = scores[h % len(scores)]
    utilization = round(0.08 + (h % 80) / 100, 2)
    history_years = 1 + (h % 20)
    rating = (
        "Excellent" if score >= 750 else
        "Good" if score >= 700 else
        "Fair" if score >= 650 else "Poor"
    )
    return json.dumps({
        "applicant": applicant_name,
        "credit_score": score,
        "credit_utilization_ratio": utilization,
        "open_credit_accounts": 2 + (h % 10),
        "credit_history_length_years": history_years,
        "recent_hard_inquiries_2yr": h % 6,
        "rating": rating,
        "report_date": datetime.now().strftime("%Y-%m-%d"),
    }, indent=2)


@tool
def check_default_history(applicant_name: str) -> str:
    """Look up defaults, late payments, and delinquencies for the applicant."""
    h = _name_hash(applicant_name)
    defaults = 1 if (h % 7 == 0) else 0
    bankruptcies = 1 if (h % 13 == 0) else 0
    late_90 = 1 if (h % 9 == 0) else 0
    return json.dumps({
        "applicant": applicant_name,
        "past_defaults": defaults,
        "bankruptcies": bankruptcies,
        "late_payments_30_days": h % 5,
        "late_payments_60_days": h % 3,
        "late_payments_90_plus_days": late_90,
        "accounts_in_collections": 1 if (h % 11 == 0) else 0,
        "last_negative_event_years_ago": (2 + h % 8) if (defaults or bankruptcies or late_90) else None,
        "report_date": datetime.now().strftime("%Y-%m-%d"),
    }, indent=2)


@tool
def check_employment_stability(applicant_name: str) -> str:
    """Retrieve employment verification and stability data."""
    h = _name_hash(applicant_name)
    emp_types = ["Full-time Salaried", "Part-time", "Self-employed",
                 "Contract", "Full-time Salaried", "Full-time Salaried"]
    industries = ["Technology", "Healthcare", "Finance",
                  "Manufacturing", "Retail", "Education", "Government"]
    years_current = round(0.5 + (h % 15) + (h % 10) / 10, 1)
    job_changes = h % 5
    stability = (
        "High" if years_current >= 5 and job_changes <= 1 else
        "Medium" if years_current >= 2 else "Low"
    )
    return json.dumps({
        "applicant": applicant_name,
        "employment_type": emp_types[h % len(emp_types)],
        "current_employer_tenure_years": years_current,
        "job_changes_last_5_years": job_changes,
        "industry": industries[h % len(industries)],
        "income_verified": h % 4 != 0,
        "stability_rating": stability,
        "verification_date": datetime.now().strftime("%Y-%m-%d"),
    }, indent=2)


# ── Tools: Compliance & Policy Agent ────────────────────────────────

_ddg = DuckDuckGoSearchRun()


@tool
def search_regulations(query: str) -> str:
    """Search the web for current Indian lending regulations (RBI) and compliance info."""
    try:
        india_query = f"India RBI lending regulation {query}"
        return _ddg.run(india_query)[:2000]
    except Exception as e:
        return f"Search unavailable: {e}. Falling back to cached regulatory knowledge."


@tool
def check_policy_rules(
    loan_type: str,
    loan_amount: float,
    applicant_age: int,
    annual_income: float,
) -> str:
    """Check internal lending-policy rules for eligibility."""
    violations, warnings = [], []

    # Age
    if applicant_age < 18:
        violations.append("Applicant must be ≥18 years old.")
    elif applicant_age > 65:
        violations.append("Applicant exceeds max age (65).")
    elif applicant_age > 60:
        warnings.append("Near max-age; restricted terms may apply.")

    # Income minimums by loan type
    min_income = {"personal": 250_000, "home": 500_000, "auto": 300_000,
                  "business": 600_000, "education": 200_000}
    req = min_income.get(loan_type.lower(), 300_000)
    if annual_income < req:
        violations.append(
            f"Min income ₹{req:,.0f} required for {loan_type}; applicant has ₹{annual_income:,.0f}."
        )

    # Loan-amount caps
    max_amounts = {"personal": 2_500_000, "home": 50_000_000, "auto": 5_000_000,
                   "business": 25_000_000, "education": 4_000_000}
    cap = max_amounts.get(loan_type.lower(), 5_000_000)
    if loan_amount > cap:
        violations.append(
            f"₹{loan_amount:,.0f} exceeds cap ₹{cap:,.0f} for {loan_type} loans."
        )

    # Income ratio
    ratio = loan_amount / annual_income if annual_income > 0 else float("inf")
    max_ratio = {"personal": 3, "home": 8, "auto": 2, "business": 5, "education": 4}
    allowed = max_ratio.get(loan_type.lower(), 3)
    if ratio > allowed:
        warnings.append(f"Loan-to-income ratio {ratio:.1f}x exceeds guideline {allowed}x.")

    status = "FAIL" if violations else ("REVIEW" if warnings else "PASS")
    return json.dumps({
        "policy_check_status": status,
        "loan_type": loan_type,
        "violations": violations,
        "warnings": warnings,
        "applicable_limits": {"min_age": 18, "max_age": 65,
                              "min_income": req, "max_loan": cap,
                              "max_income_ratio": allowed},
        "check_date": datetime.now().strftime("%Y-%m-%d"),
    }, indent=2)

@tool
def check_aml_watchlist(applicant_name: str, industry: str) -> str:
    """Check international Anti-Money Laundering (AML) and Politically Exposed Persons (PEP) watchlists."""
    h = _name_hash(applicant_name)
    
    # 1 in 25 chance of being flagged as a PEP, 1 in 40 for Sanctions
    is_pep = (h % 25 == 0) 
    is_sanctioned = (h % 40 == 0)
    
    # High-risk industries for money laundering
    high_risk_industries = ["crypto", "casino", "cash-business", "import-export"]
    industry_risk = "HIGH" if any(word in industry.lower() for word in high_risk_industries) else "LOW"

    status = "CLEAR"
    warnings = []
    
    if is_sanctioned:
        status = "FATAL_FAIL"
        warnings.append("MATCH FOUND ON OFAC SANCTIONS LIST. LENDING IS ILLEGAL.")
    elif is_pep:
        status = "REQUIRES_ENHANCED_DUE_DILIGENCE"
        warnings.append("Applicant identified as a Politically Exposed Person (PEP).")
        
    if industry_risk == "HIGH":
        warnings.append(f"Industry '{industry}' flagged as high-risk for money laundering.")

    return json.dumps({
        "aml_status": status,
        "industry_money_laundering_risk": industry_risk,
        "regulatory_warnings": warnings
    }, indent=2) 


# ── Tools: Financial Capacity Agent ─────────────────────────────────

_repl = PythonREPLTool()


@tool
def financial_calculator(python_code: str) -> str:
    """Execute Python code for financial calculations (DTI, EMI, savings buffer).
    Always use print() to display results."""
    try:
        result = _repl.run(python_code)
        return result if result.strip() else "(no output — use print())"
    except Exception as e:
        return f"Calculation error: {e}"


# ── Agent Loop Helper ───────────────────────────────────────────────

def _run_agent_loop(
    llm: ChatGoogleGenerativeAI,
    tools: list,
    system_prompt: str,
    user_message: str,
    agent_name: str,
    max_iterations: int = 6,
) -> tuple[str, list[dict]]:
    """Run a ReAct tool-calling loop; returns (final_text, trace_entries)."""
    tools_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    trace: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        response = llm_with_tools.invoke(msgs)
        msgs.append(response)

        usage = dict(response.usage_metadata) if getattr(response, "usage_metadata", None) else {}

        if not response.tool_calls:
            trace.append({
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name, "action": "final_response",
                "iteration": iteration, "output_preview": (response.content or "")[:500],
                "token_usage": usage,
            })
            return response.content or "", trace

        for tc in response.tool_calls:
            trace.append({
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name, "action": f"tool_call:{tc['name']}",
                "iteration": iteration, "input": tc["args"], "token_usage": usage,
            })
            try:
                result = tools_map[tc["name"]].invoke(tc["args"]) if tc["name"] in tools_map else f"Unknown tool: {tc['name']}"
            except Exception as e:
                result = f"Tool error: {e}"
            trace.append({
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name, "action": f"tool_result:{tc['name']}",
                "iteration": iteration, "output_preview": str(result)[:500],
            })
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            
        time.sleep(2)

    return f"[max iterations reached] {msgs[-1].content if hasattr(msgs[-1], 'content') else ''}", trace


# ── Supervisor Prompt ────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are the Chief Underwriting Supervisor. Your job is to dynamically route loan applications to specialized agents and make a final decision.

CRITICAL DYNAMIC TRIAGE INSTRUCTIONS:
You do NOT follow a fixed order. You must act as a smart triage manager. Read the initial applicant data and route to the agent most likely to reject the application first, saving processing time:

1. EVALUATE OBVIOUS RED FLAGS FIRST:
   - If you see legal/policy blockers (e.g., Age under 18, requested amount wildly over standard limits): Route to 'compliance_policy_agent' first.
   - If you see explicit credit warnings (e.g., past defaults mentioned, recent bankruptcies): Route to 'credit_risk_agent' first.
   - If you see obvious financial gaps (e.g., high expenses leaving zero room for an EMI): Route to 'financial_capacity_agent' first.
   - If there are no obvious fatal flaws, default to starting with the 'compliance_policy_agent' to establish baseline eligibility.

2. EARLY EXIT (SHORT-CIRCUIT LOGIC):
   - If ANY agent returns a FATAL REJECTION or UNACCEPTABLE rating, immediately output 'FINISH' and deny the loan. Do not waste time consulting the remaining agents.

3. PROGRESSION & COMPLEX CASES (STRESS TESTS):
   - If an agent passes or gives a conditional/mixed warning, evaluate the remaining data and route to the next most relevant agent.
   - For complex "gray area" cases (like an applicant with high savings but a past default, or high income but high debt), you MUST gather reports from ALL THREE agents before you output 'FINISH' to ensure a comprehensive audit.

4. FINAL SYNTHESIS:
   - Once you have either a fatal rejection OR all necessary reports for a complex case, output 'FINISH'. Do not route to the same agent twice.
"""


# ── Node: Supervisor ─────────────────────────────────────────────────

def supervisor_node(state: LoanUnderwriterState) -> dict:
    time.sleep(1)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_retries=5)
    router = llm.with_structured_output(SupervisorDecision)

    app = state.get("loan_application", {})
    fin = state.get("financial_analysis", "")
    cred = state.get("credit_risk_analysis", "")
    comp = state.get("compliance_analysis", "")

    parts = [f"## Loan Application\n```json\n{json.dumps(app, indent=2)}\n```"]
    for label, val in [("Financial Capacity", fin), ("Credit & Risk", cred), ("Compliance & Policy", comp)]:
        parts.append(f"## {label} Analysis\n{val}" if val else f"## {label} Analysis: **NOT YET PERFORMED**")

    decision = router.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content="\n\n".join(parts)),
    ])

    trace_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": "supervisor",
        "action": f"route_to:{decision.next}",
        "reasoning": decision.reasoning,
    }

    result: dict = {"next_agent": decision.next, "execution_trace": [trace_entry]}

    if decision.next == "FINISH":
        final_resp = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_retries=5).invoke([
            SystemMessage(content=SUPERVISOR_PROMPT + "\n\nAll analyses are complete. Provide the FINAL decision."),
            HumanMessage(content="\n\n".join(parts) + "\n\nProvide the comprehensive final underwriting decision."),
        ])
        result["final_decision"] = final_resp.content
        result["messages"] = [AIMessage(content=f"[Supervisor Final Decision]\n{final_resp.content}")]
    else:
        result["messages"] = [AIMessage(content=f"[Supervisor] → {decision.next}: {decision.reasoning}")]

    return result


# ── Node: Financial Capacity Agent ───────────────────────────────────

_FIN_PROMPT = """You are the **Indian Financial Capacity Agent**. Use the `financial_calculator` tool for ALL calculations.

Required analysis:
1. EMI = P × r × (1+r)^n / ((1+r)^n − 1)  (assume standard Indian lending rates, e.g., 10-12% annual rate)
2. DTI = (existing_debts + EMI) / monthly_income × 100
3. Savings Buffer = savings / (monthly_expenses + EMI) in months
4. Net Disposable Income = monthly_income − expenses − debts − EMI
5. Affordability: Comfortable / Manageable / Stretched / Unaffordable

Output a structured report; rate capacity as STRONG / ADEQUATE / WEAK / INSUFFICIENT."""


def financial_capacity_agent_node(state: LoanUnderwriterState) -> dict:
    time.sleep(1)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_retries=5)
    app = state.get("loan_application", {})
    text, trace = _run_agent_loop(
        llm, [financial_calculator], _FIN_PROMPT,
        f"Analyze financial capacity:\n```json\n{json.dumps(app, indent=2)}\n```\nUse the calculator tool.",
        "financial_capacity_agent",
    )
    return {
        "financial_analysis": text,
        "execution_trace": trace,
        "messages": [AIMessage(content=f"[Financial Capacity Agent]\n{text}")],
    }


# ── Node: Credit & Risk Agent ───────────────────────────────────────

_CREDIT_PROMPT = """You are the **Indian Credit & Risk Agent**. You MUST call all three tools:
`check_credit_score`, `check_default_history`, `check_employment_stability`.

Assessment criteria:
- CIBIL / Credit Score bands: Poor <650 | Fair 650-699 | Good 700-749 | Excellent 750+
- Default risk from history of late payments, collections, bankruptcies
- Employment stability from tenure, type, industry

Output: combined probability of default (LOW/MODERATE/HIGH/VERY HIGH) and risk recommendation (ACCEPTABLE/MARGINAL/UNACCEPTABLE)."""


def credit_risk_agent_node(state: LoanUnderwriterState) -> dict:
    time.sleep(1)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_retries=5)
    app = state.get("loan_application", {})
    name = app.get("applicant_name", "Unknown")
    text, trace = _run_agent_loop(
        llm, [check_credit_score, check_default_history, check_employment_stability],
        _CREDIT_PROMPT,
        f"Evaluate credit risk for **{name}**:\n```json\n{json.dumps(app, indent=2)}\n```\nUse ALL three tools.",
        "credit_risk_agent",
    )
    return {
        "credit_risk_analysis": text,
        "execution_trace": trace,
        "messages": [AIMessage(content=f"[Credit & Risk Agent]\n{text}")],
    }


# ── Node: Compliance & Policy Agent ──────────────────────────────────

_COMP_PROMPT = """You are the **Indian Compliance & Policy Agent** and Lead Fraud Investigator. You MUST use ALL THREE tools:
1. `check_policy_rules` — internal policy engine
2. `search_regulations` — web search for Indian lending regulations
3. `check_aml_watchlist` — Anti-Money Laundering and Sanctions check

Checks required:
- Age/Income eligibility and Loan-to-Income ratios.
- RBI KYC/AML norms.
- AML Watchlist status and Industry Risk.
- LOGICAL FRAUD: Cross-reference data (e.g., If Age - years_employed < 18, flag as HIGH FRAUD RISK).

Output: compliance status (COMPLIANT / NON-COMPLIANT / NEEDS REVIEW), violations, warnings, and AML status. If AML returns a FATAL_FAIL or you detect obvious logical fraud, output a FATAL NON-COMPLIANT rating."""


def compliance_policy_agent_node(state: LoanUnderwriterState) -> dict:
    time.sleep(1)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, max_retries=5)
    app = state.get("loan_application", {})
    text, trace = _run_agent_loop(
        llm, [check_policy_rules, search_regulations, check_aml_watchlist], _COMP_PROMPT,
        (f"Check compliance:\n```json\n{json.dumps(app, indent=2)}\n```\n"
         f"Use `check_policy_rules` with loan_type='{app.get('loan_purpose','personal')}', "
         f"loan_amount={app.get('loan_amount_requested',0)}, "
         f"applicant_age={app.get('age',0)}, annual_income={app.get('annual_income',0)}.\n"
         f"Also `search_regulations` for relevant lending rules.\n"
         f"For AML checks, use industry='{app.get('industry', 'unknown')}'."),
        "compliance_policy_agent",
    )
    return {
        "compliance_analysis": text,
        "execution_trace": trace,
        "messages": [AIMessage(content=f"[Compliance & Policy Agent]\n{text}")],
    }


# ── Graph Construction ───────────────────────────────────────────────

def _route(state: LoanUnderwriterState) -> str:
    nxt = state.get("next_agent", "FINISH")
    return END if nxt == "FINISH" else nxt


def build_graph(checkpointer=None):
    """Build and compile the Loan Underwriter StateGraph."""
    g = StateGraph(LoanUnderwriterState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("financial_capacity_agent", financial_capacity_agent_node)
    g.add_node("credit_risk_agent", credit_risk_agent_node)
    g.add_node("compliance_policy_agent", compliance_policy_agent_node)

    g.set_entry_point("supervisor")

    g.add_conditional_edges("supervisor", _route, {
        "financial_capacity_agent": "financial_capacity_agent",
        "credit_risk_agent": "credit_risk_agent",
        "compliance_policy_agent": "compliance_policy_agent",
        END: END,
    })

    for agent in AGENT_NAMES:
        g.add_edge(agent, "supervisor")

    return g.compile(checkpointer=checkpointer)
