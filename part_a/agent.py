from __future__ import annotations

import os
import sys
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PART_A_ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(PART_A_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_A_ROOT))


# ============================================================
# IMPORT TOOLS
# ============================================================

from part_a.tools.langchain_tools import (
    compare_output_on_high_dam_price,
    compare_weekly_capacity_factor,
    get_dam_summary,
    get_turbine_capacity_factor,
    get_turbine_summary,
    lookup_compliance_rule,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(ROOT / ".env")


# ============================================================
# MODEL
# ============================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI operations assistant for a renewable-energy
Power-to-X facility.

You answer operational questions using supplied:

- turbine telemetry data
- DAM electricity price data
- deterministic calculations
- RFNBO compliance rulebook information

IMPORTANT:

The numerical results provided to you come from deterministic
Python tools operating on the supplied dataset.

Do not invent numerical values.

Do not invent compliance requirements.

Do not add facts that are not supported by the tool output.

If the tool output says that information is unavailable,
clearly state that limitation.

For compliance questions, rely only on the supplied
compliance rulebook.

For numerical questions, rely only on the supplied
dataset/tool output.

Give concise answers and briefly explain the evidence
supporting the answer.
"""


# ============================================================
# HELPER — CHECK PHRASES
# ============================================================

def _contains_any(
    question: str,
    phrases: list[str]
) -> bool:

    lowered = question.lower()

    return any(
        phrase.lower() in lowered
        for phrase in phrases
    )


# ============================================================
# HELPER — EXTRACT TURBINE ID
# ============================================================

def _extract_turbine_id(
    question: str
) -> str | None:

    match = re.search(
        r"\bT0[1-5]\b",
        question.upper()
    )

    if match:
        return match.group(0)

    return None


# ============================================================
# HELPER — EXTRACT DAM THRESHOLD
# ============================================================

def _extract_threshold(
    question: str
) -> float:

    # Handles:
    # > ₹4
    # > 4
    # exceeded ₹4
    # above ₹4
    # greater than ₹4

    patterns = [
        r">\s*₹?\s*([0-9]+(?:\.[0-9]+)?)",

        r"(?:exceeded|above|greater than|higher than)"
        r"\s*₹?\s*([0-9]+(?:\.[0-9]+)?)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question.lower()
        )

        if match:
            return float(match.group(1))

    # Assignment example uses ₹4/kWh
    return 4.0


# ============================================================
# ROUTER
# ============================================================

def _route_question(question: str):

    lowered = question.lower().strip()


    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not lowered:

        return (
            None,
            "Please ask a non-empty question."
        )


    # --------------------------------------------------------
    # 1. COMPLIANCE / RULEBOOK
    # --------------------------------------------------------

    if _contains_any(
        lowered,
        [
            "temporal correlation",
            "hourly matching",
            "monthly averaging",
            "electrolyzer",
            "rfnbo",
            "compliance",
            "rulebook",
        ],
    ):

        return (
            lambda: lookup_compliance_rule.invoke(
                {
                    "query": question
                }
            ),
            "lookup_compliance_rule",
        )


    # --------------------------------------------------------
    # 2. AVERAGE DAM PRICE
    #
    # IMPORTANT:
    # This must come BEFORE generic "dam price"
    # matching.
    # --------------------------------------------------------

    if _contains_any(
        lowered,
        [
            "average dam price",
            "average price",
            "dam summary",
            "dam price average",
        ],
    ):

        return (
            lambda: get_dam_summary.invoke({}),
            "get_dam_summary",
        )


    # --------------------------------------------------------
    # 3. DAM THRESHOLD / HIGH PRICE ANALYSIS
    # --------------------------------------------------------

    if (
        "dam" in lowered
        and _contains_any(
            lowered,
            [
                "exceeded",
                "above",
                "greater than",
                "higher than",
                "threshold",
            ],
        )
    ):

        threshold = _extract_threshold(
            lowered
        )

        return (
            lambda: compare_output_on_high_dam_price.invoke(
                {
                    "threshold": threshold
                }
            ),
            "compare_output_on_high_dam_price",
        )


    # --------------------------------------------------------
    # 4. WEEKLY CAPACITY FACTOR COMPARISON
    # --------------------------------------------------------

    if _contains_any(
        lowered,
        [
            "week 1",
            "week one",
            "second week",
            "first week",
            "compare the first and second week",
            "first and second week",
        ],
    ):

        return (
            lambda: compare_weekly_capacity_factor.invoke(
                {}
            ),
            "compare_weekly_capacity_factor",
        )


    # --------------------------------------------------------
    # 5. EXTRACT TURBINE ID
    # --------------------------------------------------------

    turbine_id = _extract_turbine_id(
        lowered
    )


    # --------------------------------------------------------
    # 6. TURBINE CAPACITY FACTOR
    # --------------------------------------------------------

    if (
        turbine_id
        and "capacity factor" in lowered
    ):

        return (
            lambda: get_turbine_capacity_factor.invoke(
                {
                    "turbine_id": turbine_id
                }
            ),
            "get_turbine_capacity_factor",
        )


    # --------------------------------------------------------
    # 7. TURBINE POWER / SUMMARY
    # --------------------------------------------------------

    if (
        turbine_id
        and _contains_any(
            lowered,
            [
                "average power",
                "power output",
                "summary",
                "wind speed",
                "availability",
                "maximum power",
                "minimum power",
            ],
        )
    ):

        return (
            lambda: get_turbine_summary.invoke(
                {
                    "turbine_id": turbine_id
                }
            ),
            "get_turbine_summary",
        )


    # --------------------------------------------------------
    # 8. CAPACITY FACTOR WITHOUT TURBINE
    # --------------------------------------------------------

    if "capacity factor" in lowered:

        return (
            None,
            "Please specify a turbine ID such as T01, T02, "
            "T03, T04, or T05."
        )


    # --------------------------------------------------------
    # 9. POWER OUTPUT WITHOUT TURBINE
    # --------------------------------------------------------

    if _contains_any(
        lowered,
        [
            "average power",
            "power output",
        ],
    ):

        return (
            None,
            "Please specify a turbine ID such as T01-T05."
        )


    # --------------------------------------------------------
    # 10. UNKNOWN QUESTION
    # --------------------------------------------------------

    return (
        None,
        "I could not map that question to the supplied "
        "telemetry, DAM, or rulebook tools."
    )


# ============================================================
# LLM ANSWER SYNTHESIS
# ============================================================

def _synthesize_answer(
    question: str,
    tool_output: str,
    tool_name: str | None = None,
) -> str:

    # If Groq isn't configured, return the deterministic
    # tool result directly.

    if not os.getenv("GROQ_API_KEY"):

        return tool_output.strip()


    tool_context = ""

    if tool_name:

        tool_context = (
            f"\nTool used: {tool_name}\n"
        )


    messages = [

        SystemMessage(
            content=(
                SYSTEM_PROMPT
                + tool_context
                + """
Return only the final answer to the user's question.

Do not describe what you should have done.

Do not say that you failed to use a tool.

The tool output already contains the evidence required
to answer the question.

If the tool output does not contain enough information,
say so explicitly.
"""
            )
        ),

        HumanMessage(
            content=(
                f"User question:\n{question}\n\n"
                f"Tool output:\n{tool_output}"
            )
        ),
    ]


    try:

        response = llm.invoke(
            messages
        )

        content = getattr(
            response,
            "content",
            ""
        )

        if content:

            return content.strip()


    except Exception:

        # Graceful fallback:
        # return deterministic tool output if
        # LLM synthesis fails.

        pass


    return tool_output.strip()


# ============================================================
# MAIN AGENT FUNCTION
# ============================================================

def ask_agent(
    question: str,
    use_llm: bool = True
):

    tool_function, tool_name_or_error = (
        _route_question(question)
    )


    # --------------------------------------------------------
    # Routing error
    # --------------------------------------------------------

    if tool_function is None:

        return tool_name_or_error


    tool_name = tool_name_or_error


    # --------------------------------------------------------
    # Execute deterministic tool
    # --------------------------------------------------------

    try:

        tool_output = tool_function()

    except Exception as e:

        return (
            "I encountered an error while retrieving "
            f"the required data: {e}"
        )


    # --------------------------------------------------------
    # No LLM mode
    # --------------------------------------------------------

    if not use_llm:

        return tool_output


    # --------------------------------------------------------
    # LLM synthesis
    # --------------------------------------------------------

    return _synthesize_answer(
        question,
        tool_output,
        tool_name,
    )


# ============================================================
# INTERACTIVE CLI
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Greenko Renewable Energy Agent")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)


    while True:

        question = input(
            "\nYou: "
        ).strip()


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if question.lower() in [
            "exit",
            "quit",
        ]:

            print(
                "\nAgent: Goodbye!"
            )

            break


        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not question:

            continue


        # ----------------------------------------------------
        # PROCESS QUESTION
        # ----------------------------------------------------

        try:

            answer = ask_agent(
                question
            )

            print(
                "\nAgent:"
            )

            print(
                answer
            )


        except Exception as e:

            print(
                "\nAgent Error:"
            )

            print(
                e
            )