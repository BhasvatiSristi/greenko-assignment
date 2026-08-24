import sqlite3
import re
from pathlib import Path

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
import json

from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data_pack" / "greenko_part_a.db"


# ---------------------------------------------------------
# LLM used for SQL generation
# ---------------------------------------------------------

sql_llm = ChatGroq(
    model="openai/gpt-oss-120b",  # High-capability agentic reasoning; Tool Use; 131K context window; and fast inference (~500 TPS).
    temperature=0
)


# ---------------------------------------------------------
# Get database schema
# ---------------------------------------------------------

def get_database_schema():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor() # cursor allows Python to execute SQL commands

    tables = cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        AND name NOT LIKE 'sqlite_%'
    """).fetchall()   #list of tuples

    schema_parts = []

    for (table_name,) in tables:

        columns = cursor.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()

        column_text = []

        for column in columns:
            column_name = column[1]
            column_type = column[2]

            column_text.append(
                f"- {column_name} ({column_type})"
            )

        schema_parts.append(
            f"TABLE: {table_name}\n"
            + "\n".join(column_text)
        )

    conn.close()

    return "\n\n".join(schema_parts)

# ---------------------------------------------------------
# Execute read-only SQL
# ---------------------------------------------------------

def execute_sql(sql):

    sql_clean = sql.strip()

    # Only allow SELECT / WITH queries
    if not (
        sql_clean.lower().startswith("select")
        or sql_clean.lower().startswith("with")
    ):
        raise ValueError(
            "Only read-only SELECT queries are allowed."
        )


    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro", # open in read-only mode
        uri=True
    )

    try:

        cursor = conn.cursor()

        cursor.execute(sql_clean)

        rows = cursor.fetchall()

        columns = [
            description[0]
            for description in cursor.description
        ]

        return {
            "columns": columns,
            "rows": rows
        }

    finally:
        conn.close()


# ---------------------------------------------------------
# SQL generation
# ---------------------------------------------------------

def generate_sql(user_query):

    schema = get_database_schema()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a SQLite SQL generation assistant.

Your ONLY job is to convert the user's natural-language question
into a valid SQLite SELECT query.

You have access only to the database schema provided below.

DATABASE SCHEMA:

{schema}

Rules:

1. Generate ONLY a SELECT or WITH query.
2. Never modify the database.
3. Never use INSERT.
4. Never use UPDATE.
5. Never use DELETE.
6. Never use DROP.
7. Never use ALTER.
8. Never invent tables.
9. Never invent columns.
10. Use only the supplied schema.
11. Use SQLite-compatible syntax.
12. Use appropriate aggregation such as AVG, SUM, MIN, MAX and COUNT.
13. Handle dates carefully.
14. If the user specifies a turbine ID, filter by turbine_id.
15. If the user specifies a date, filter by that date.
16. If the user asks for all turbines, do NOT require a turbine ID.
17. If two datasets are required, use the appropriate JOIN.
18. Return ONLY SQL. Do not provide explanations.
"""
        ),
        (
            "human",
            "User question:\n{question}"
        )
    ])

    chain = prompt | sql_llm

    response = chain.invoke({
        "schema": schema,
        "question": user_query
    })

    return response.content


# ---------------------------------------------------------
# Extract prices in specific window
# ---------------------------------------------------------

def get_dam_prices_for_window(user_query: str) -> dict:
    """
    Extract dates from the user query and retrieve
    DAM prices for that period.
    """

    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract the start and end dates from the user's question.

        Return ONLY JSON in this format:

        {{
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"
        }}

        Convert any date format used by the user into YYYY-MM-DD.

        If only one date is given, use that date for both
        start_date and end_date.

        If no date is given, use:
        start_date = 2026-03-01
        end_date = 2026-03-14

        Do not return any explanation or markdown. Return ONLY valid JSON.
        """
            ),
            (
                "human",
                "{question}"
            )
        ])

    chain = prompt | sql_llm

    response = chain.invoke({
        "question": user_query
    })

    dates = json.loads(response.content)

    start_date = dates["start_date"]
    end_date = dates["end_date"]

    # Make end date inclusive
    end_exclusive = (
        datetime.strptime(end_date, "%Y-%m-%d")
        + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    query = """
        SELECT timestamp, dam_price_inr_per_kwh
        FROM dam_prices
        WHERE timestamp >= ?
          AND timestamp < ?
        ORDER BY timestamp
    """

    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    try:
        cursor = conn.cursor()

        cursor.execute(
            query,
            (start_date, end_exclusive)
        )

        rows = cursor.fetchall()

        return {
            "start_date": start_date,
            "end_date": end_date,
            "rows": rows
        }

    finally:
        conn.close()

# ---------------------------------------------------------
# Main Data Query Tool
# ---------------------------------------------------------

@tool
def query_data(user_query: str) -> str:
    """
    Query the Greenko turbine telemetry and DAM price database.

    Use this tool for questions involving:
    - turbine power
    - turbine wind speed
    - turbine availability
    - total fleet generation
    - averages
    - minimum/maximum values
    - counts
    - DAM prices
    - dates and time periods
    - comparisons involving telemetry and DAM prices
    - aggregations over the structured datasets

    This tool converts the natural-language question into SQL,
    executes the SQL against the read-only SQLite database,
    and returns the actual database result.
    """

    try:

        sql = generate_sql(user_query)

        result = execute_sql(sql)

        return (
            f"Generated SQL:\n{sql}\n\n"
            f"Database result:\n{result}"
        )

    except Exception as e:

        return (
            "The structured data query could not be completed. "
            f"Reason: {str(e)}"
        )