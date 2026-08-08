import os
import io
import sys
import traceback

from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

from langserve import add_routes


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# 2. STATE DEFINITION
# ============================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return its output or error trace."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""

    prompt = (
        "You are a Senior QA Engineer. "
        "Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'.\n"
        "Include standard cases and edge cases. "
        "Return them as a numbered list."
    )

    response = llm.invoke(prompt)

    return response.content if hasattr(response, "content") else str(response)


# ============================================================
# 4. GRAPH NODES
# ============================================================

def task_input_node(state: CrewState):

    return {
        "next_step": "developer"
    }


def real_time_developer(state: CrewState):

    print("\n" + "=" * 50)
    print("              DEVELOPER ROLE")
    print("=" * 50)

    task = state["messages"][-1].content

    dev_prompt = (
        "Write a clean Python script to solve this coding task:\n\n"
        f"{task}\n\n"
        "Only return the Python code. "
        "Do not include explanations or markdown formatting."
    )

    response = llm.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        code_str = "\n".join(parts)

    else:
        code_str = str(content)

    print("[Developer] Code generated successfully.")

    return {
        "code": code_str
    }


def real_time_tester(state: CrewState):

    print("\n" + "=" * 50)
    print("                TESTER ROLE")
    print("=" * 50)

    task = state["messages"][-1].content

    print("[Tester] Generating test scenarios...")

    test_cases = generate_test_cases.invoke(task)

    cases_str = str(test_cases)

    print("[Tester] Executing generated code...")

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS:\n"
        f"{cases_str}"
    )

    print("[Tester] Testing completed.")

    return {
        "report": report,
        "next_step": "manager"
    }


def manager_decision_node(state: CrewState):

    print("\n" + "=" * 50)
    print("               MANAGER ROLE")
    print("=" * 50)

    print("[Manager] Reviewing tester report...")

    return {
        "next_step": "completed"
    }


def archiver_node(state: CrewState):

    print("\n" + "=" * 50)
    print("              ARCHIVER ROLE")
    print("=" * 50)

    print("[Archiver] Task processing completed.")

    return {
        "next_step": "exit"
    }


# ============================================================
# 5. GRAPH CONSTRUCTION
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", real_time_developer)
workflow.add_node("tester", real_time_tester)
workflow.add_node("manager_decision", manager_decision_node)
workflow.add_node("archiver", archiver_node)

workflow.add_edge(START, "task_input")
workflow.add_edge("task_input", "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager_decision")
workflow.add_edge("manager_decision", "archiver")
workflow.add_edge("archiver", END)

rt_app = workflow.compile()

print("LangGraph workflow compiled successfully.")


# ============================================================
# 6. FASTAPI APPLICATION
# ============================================================

app = FastAPI(title="LangGraph Coding Agent")


# ============================================================
# 7. LANGSERVE PLAYGROUND
# ============================================================

add_routes(
    app,
    rt_app,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 8. API REQUEST MODEL
# ============================================================

class CodingRequest(BaseModel):
    task: str


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "LangGraph Coding Agent is running"
    }


# ============================================================
# 10. CODING ENDPOINT
# ============================================================

@app.post("/run")
def run_agent(request: CodingRequest):

    initial_state: CrewState = {
        "messages": [
            HumanMessage(content=request.task)
        ],
        "next_step": None,
        "code": None,
        "report": None,
    }

    result = rt_app.invoke(
        initial_state,
        config={
            "recursion_limit": 50
        }
    )

    return {
        "task": request.task,
        "generated_code": result.get("code"),
        "report": result.get("report"),
    }
