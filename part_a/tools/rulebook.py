from pathlib import Path

from docx import Document
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RULEBOOK_PATH = (
    PROJECT_ROOT
    / "data_pack"
    / "compliance_rulebook.docx"
)


# ---------------------------------------------------------
# Load rulebook once
# ---------------------------------------------------------

def load_rulebook():

    if not RULEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Rulebook not found at: {RULEBOOK_PATH}"
        )

    document = Document(RULEBOOK_PATH)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


RULEBOOK_TEXT = load_rulebook()


# ---------------------------------------------------------
# Rulebook LLM
# ---------------------------------------------------------

rulebook_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# ---------------------------------------------------------
# Rulebook Tool
# ---------------------------------------------------------

@tool
def lookup_rulebook(query: str) -> str:
    """
    Answer questions about the RFNBO compliance rulebook.

    Use this tool for questions about:
    - temporal correlation
    - hourly matching
    - monthly matching
    - electrolyzer requirements
    - compliance requirements
    - renewable generation correlation
    - requirements explicitly stated in the supplied rulebook

    The answer must be based only on the supplied rulebook.
    """

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a compliance assistant.

Answer the user's question using ONLY the supplied
RFNBO compliance rulebook.

Do not use outside knowledge.

Do not invent requirements.

If the rulebook does not contain enough information to answer
the question, explicitly say:

"The supplied rulebook does not specify this."

RULEBOOK:

{rulebook}
"""
        ),
        (
            "human",
            "Question:\n{question}"
        )
    ])

    chain = prompt | rulebook_llm

    response = chain.invoke({
        "rulebook": RULEBOOK_TEXT,
        "question": query
    })

    return response.content