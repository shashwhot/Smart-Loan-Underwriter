# EVAL.md: Multi-Agent Supervisor System Analysis

## 1. Problem Definition & Agent Personas
The system tackles the complex domain of **Automated Loan Underwriting**. Traditional lending requires consensus across siloed departments (Legal, Quantitative, and Risk). A single monolithic LLM is prone to hallucination when balancing these distinct domains. Therefore, I implemented a Multi-Agent Supervisor architecture to delegate tasks to highly specialized, role-playing worker nodes:
*   **The Financial Capacity Agent (Quantitative Engine):** Acts as the math expert. It uses a Python REPL tool to perform deterministic calculations (EMI, Debt-to-Income, Savings Buffer) to evaluate objective affordability.
*   **The Compliance & Policy Agent (Legal Gatekeeper):** Acts as the risk and regulatory officer. It utilizes internal policy checkers, AML watchlists, and search tools to ensure RBI guideline adherence and prevent money laundering.
*   **The Credit Risk Agent (Truth Verifier):** Acts as the bureau investigator. It uses mock APIs to cross-reference self-reported application data against verified historical records, scoring creditworthiness and employment stability.

## 2. Architectural Correctness & Tool Usage
The system implements a true **Supervisor Pattern** rather than a hard-coded sequential chain. The Supervisor acts as a dynamic router, evaluating the global state after every worker node execution to determine the next best step. 

Worker agents successfully demonstrated autonomous tool usage (`bind_tools`) throughout the stress tests. The most prominent example is the Financial Capacity agent, which consistently generated, executed, and parsed raw Python code to calculate complex loan amortizations instead of relying on the LLM's native (and often flawed) arithmetic capabilities. 

## 3. Stress Test Analysis: Routing Logic
To evaluate the Supervisor's decision-making, I executed 6 highly complex stress tests. The routing logic proved to be dynamic, efficient, and capable of deep conflict resolution.

*   **Early Exits & Token Efficiency (Short-Circuit Logic):** The Supervisor does not blindly force every application through a full audit if a fatal flaw is detected. 
    *   *Mathematical Short-Circuit:* In one high-income test case (a ₹90 Lakh earner requesting a luxury vehicle), the Financial agent calculated a 78% DTI. The Supervisor recognized this as fundamentally unviable and immediately terminated the graph, skipping the Credit and Compliance checks entirely.
    *   *Credit Short-Circuit:* Similarly, when an applicant presented a recent bankruptcy, the Supervisor instantly routed to `FINISH` upon receiving the "Unacceptable" rating from the Credit agent, proving it prioritizes fatal risks over completing unnecessary steps.
*   **Cross-Domain Conflict Resolution:** The system proved it does not simply take a "majority vote." In one case, an applicant possessed massive liquidity (₹1.2 Crore savings), earning a "Strong" financial rating. However, the Compliance agent flagged their "Import-Export" industry for AML risk, and the Credit agent caught them lying about their employment. The Supervisor synthesized these conflicting domain reports to correctly issue a Fraud Rejection.
*   **Nuanced & Conditional Approvals:** When applications fell into "gray areas," the Supervisor exhibited advanced critical thinking rather than defaulting to a binary Pass/Fail. 
    *   When an applicant had a pristine profile but an 86% credit utilization ratio, the Supervisor issued an **Approved with Stipulations**, requiring a written liquidity explanation. 
    *   When an applicant had excellent credit but a discrepancy in their job title data, the Supervisor issued an **Approved Subject to Verification (ASV)**, pausing final disbursement until manual documentation was provided.

## 4. Feature Robustness 
*   **State Persistence across Restarts:** The system's memory persistence was validated using a SQLite checkpointer (`checkpoints.sqlite`). I verified robustness by executing a multi-agent loop, shutting down the Uvicorn/FastAPI server, restarting the application, and successfully retrieving the historical LangGraph state and final decisions via the `GET /v1/thread/{thread_id}` endpoint.
*   **Telemetry & Traceability:** The API successfully captures and returns the full `execution_trace` (documenting exact tool inputs and timestamped iterations) alongside an `estimated_cost_usd` block, providing full transparency into token consumption for the entire supervisor loop.

## 5. Critical Thinking: LLM Failures & Prompt Optimization

Building a Multi-Agent system exposed several inherent limitations and failure modes of Large Language Models when operating autonomously. I identified and addressed the following LLM failures during development:

### A. The "Passive Reporter" vs. "Active Investigator"
* **The Failure:** Early iterations of the Credit Risk and Compliance agents suffered from "passive reporting." If the user application claimed a 15-year tenure, but the mock verification API returned a 3-year tenure with 4 job changes, the LLM would occasionally just list both facts neutrally without flagging the logical contradiction as fraud.
* **The Optimization:** I had to optimize the agent system prompts to mandate "Active Cross-Referencing." By explicitly instructing the agents to *compare* the `loan_application` JSON against the tool outputs and calculate discrepancies (e.g., "Subtract the years employed from the applicant's age to check for logical impossibility"), the agents transitioned from passive summarizers into active fraud investigators.


### B. State Dilution & The "Lost in the Middle" Phenomenon
* **The Failure:** Because LangGraph maintains a shared global state, the Supervisor's prompt grows massively by the end of the loop. It contains the original application, the Financial report, the Compliance report, and the Credit Risk report. In some edge cases, the Supervisor exhibited the "Lost in the Middle" phenomenon, overlooking a minor warning in the middle of the Compliance report because it was heavily focused on the final Credit Risk report appended at the very bottom.
* **The Optimization:** To mitigate context dilution, I enforced strict output formatting for the worker agents. Instead of allowing conversational output, agents are required to output highly structured Markdown with bolded **FATAL**, **WARNING**, or **CLEAR** tags. This structural anchoring forces the Supervisor's attention graph to weigh critical flags properly, regardless of their position in the context window.


## 6. Future Optimization for Ambiguity
Currently, if an application lacks vital context (e.g., missing business purpose), the system leans toward rejection based on high risk. To handle ambiguous user requests more effectively, I would update the Supervisor's system prompt and graph structure to incorporate a **"Clarification Routing" state**. This would allow the Supervisor to pause execution, query the user for missing documents (like a business plan or tax return), and resume the graph once the ambiguity is resolved, creating a more robust Human-in-the-Loop system.