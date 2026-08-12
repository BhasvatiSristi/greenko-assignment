import os
import pandas as pd
from docx import Document

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent


load_dotenv()


# --------------------------------------------------
# Load rulebook
# --------------------------------------------------

document = Document(
    "../data_pack/compliance_rulebook.docx"
)

rulebook_lines = [
    paragraph.text.strip()
    for paragraph in document.paragraphs
    if paragraph.text.strip()
]


# --------------------------------------------------
# Rulebook tool
# --------------------------------------------------

@tool
def lookup_compliance_rule(keyword: str) -> str:
    """
    Search the supplied RFNBO compliance rulebook
    for information relevant to a keyword.

    Use this for questions about RFNBO, temporal
    correlation, electrolyzer consumption, hourly
    matching, and compliance.
    """

    keyword = keyword.lower().strip()

    matches = [
        line
        for line in rulebook_lines
        if keyword in line.lower()
    ]

    if not matches:
        return (
            f"No relevant rule was found in the "
            f"supplied rulebook for '{keyword}'."
        )

    return "\n".join(matches)


# --------------------------------------------------
# Model
# --------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# --------------------------------------------------
# Agent
# --------------------------------------------------

agent = create_agent(
    model=llm,
    tools=[lookup_compliance_rule],
    system_prompt=(
        "You are a renewable-energy compliance "
        "assistant. Use the rulebook tool whenever "
        "the user asks about compliance. Never invent "
        "a compliance requirement."
    )
)


# --------------------------------------------------
# Test
# --------------------------------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "According to the rulebook, "
                    "what is temporal correlation?"
                )
            }
        ]
    }
)


print("\nFINAL ANSWER:")
print(
    result["messages"][-1].content
)