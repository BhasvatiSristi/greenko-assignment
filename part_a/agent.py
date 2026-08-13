import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain.agents import create_agent

from part_a.tools import ALL_TOOLS


# =========================================================
# Environment
# =========================================================

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Add it to your environment or .env file."
    )


# =========================================================
# Main LLM
# =========================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# =========================================================
# Agent Instructions
# =========================================================

SYSTEM_PROMPT = """
You are the Greenko Renewable Energy Operations Agent.

You answer questions about the supplied renewable-energy
telemetry, DAM prices and RFNBO compliance rulebook.

You have access to three categories of tools:

1. Structured Data Query
   - Use query_data for turbine telemetry and DAM price questions.
   - This tool queries the SQLite database.

2. Deterministic Calculators
   - Use calculator tools for formulas and numerical calculations.
   - Never perform important numerical calculations mentally when
     a calculator tool is available.

3. Compliance Rulebook
   - Use lookup_rulebook for questions about compliance requirements,
     temporal correlation, hourly matching, electrolyzer requirements,
     and other rulebook content.

---------------------------------------------------------
IMPORTANT TOOL USAGE
---------------------------------------------------------

Do NOT use keyword-based assumptions.

Understand the user's actual intent.

For example:

"What is the power output for all turbines together?"

means fleet-level total power.

Do NOT ask for a turbine ID.

Use query_data.

"What is the average power output of T01?"

Use query_data.

"What is the capacity factor of T01?"

First use query_data to obtain the average power of T01.

Then use calculate_capacity_factor.

"Compare the capacity factor of T01 and T05."

Obtain the average power for both turbines using query_data,
calculate both capacity factors, then compare them.

"What is the average DAM price?"

Use query_data.

"What was T01's power output on April 1st 2026?"

Use query_data and respect the requested date.

Do NOT silently substitute an overall average.

"According to the rulebook, what is temporal correlation?"

Use lookup_rulebook.

---------------------------------------------------------
MULTI-TOOL QUESTIONS
---------------------------------------------------------

A question may require multiple tools.

Use them sequentially when necessary.

Example:

"What is the capacity factor of T05?"

query_data
    ↓
average power
    ↓
calculate_capacity_factor
    ↓
final answer

---------------------------------------------------------
DATA GROUNDING
---------------------------------------------------------

Never invent:

- turbine values
- DAM prices
- timestamps
- compliance rules
- calculations

If the database does not contain the requested observation,
say that the requested data is unavailable.

If the rulebook does not specify something, say so.

---------------------------------------------------------
SCOPE
---------------------------------------------------------

This agent is specifically for the Greenko renewable-energy
operational dataset and compliance rulebook.

If the user asks an unrelated question such as:

"Who won the FIFA World Cup?"

say that the question is outside the scope of the supplied
Greenko operational data and compliance sources.

---------------------------------------------------------
ANSWER STYLE
---------------------------------------------------------

Keep answers concise but useful.

When a numerical result is available, provide the number and
appropriate unit.

For example:

"The average power output of T01 is 428.08 kW."

Do not unnecessarily expose internal tool calls or SQL unless
the user explicitly asks for them.

If useful, briefly explain how the result was obtained.
"""


# =========================================================
# Create Agent
# =========================================================

agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT
)


# =========================================================
# Ask Agent
# =========================================================

def ask_agent(question: str) -> str:

    try:

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        })

        messages = result.get("messages", [])

        if not messages:
            return "I could not generate an answer."

        # The final message should contain the agent's response.
        final_message = messages[-1]

        content = final_message.content

        if isinstance(content, str):
            return content

        # Handle structured content if returned
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    if item.get("type") == "text":
                        text_parts.append(
                            item.get("text", "")
                        )

            if text_parts:
                return "\n".join(text_parts)

        return str(content)

    except Exception as e:

        return (
            "I encountered an error while processing the question. "
            f"Please try again.\n\nDetails: {str(e)}"
        )


# =========================================================
# Interactive CLI
# =========================================================

def main():

    print("=" * 60)
    print("Greenko Renewable Energy Agent")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:

        try:
            question = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:

            print("Goodbye!")
            break

        print("\nAgent:")

        answer = ask_agent(question)

        print(answer)


if __name__ == "__main__":
    main()