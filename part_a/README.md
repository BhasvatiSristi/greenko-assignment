# Part A Renewable Energy Agent

## Problem Framing

This Part A agent answers operational questions over three supplied sources: the 14-day turbine telemetry CSV, the 14-day DAM price CSV, and the short RFNBO compliance rulebook. The main objective is not free-form chat; it is evidence-based question answering with deterministic tool use, so the agent must query the data and lookup the rulebook instead of guessing. The supplied telemetry has 1,680 rows across five turbines, and the DAM file has 336 hourly price observations. The rulebook is intentionally short, so a lightweight deterministic lookup is safer than a full retrieval stack.

## Architecture

User
↓
LangChain Agent + Groq
↓
├── Structured Data Tool
├── Calculator Tool
└── Compliance Rulebook Tool
↓
Evidence-based response

## Tools

- Structured data tool: `DataQueryTool` for turbine telemetry and DAM price queries.
- Calculator tool: `CalculationTool` for capacity factor, averages, percentage differences, and coverage-style metrics.
- Rulebook tool: `RulebookTool` for deterministic lookup of the supplied RFNBO extract.

## Reasoning Flow

The agent receives a natural-language question, decides whether it needs data, a derived calculation, a compliance lookup, or a combination, and then calls the smallest tool that can answer that part of the question. Numerical answers are produced by deterministic code. Compliance answers are grounded only in the supplied rulebook text. If the question cannot be answered from the provided data or rules, the agent says so explicitly.

## Why LangChain

LangChain is a good fit because the assignment is about agentic tool selection and reasoning, not building a custom orchestration layer from scratch. The current codebase already uses LangChain and `ChatGroq`, so the implementation stays compact and interview-defensible.

## Why Deterministic Rulebook Lookup Instead Of Full RAG

The supplied rulebook is short and self-contained. A deterministic lookup returns the actual rule text without introducing embedding/indexing complexity, retrieval tuning, or accidental paraphrasing. That makes the compliance behavior easier to verify and safer to explain in an interview.

## Failure Handling

- Unknown turbines return a clear invalid/unknown message.
- Unsupported compliance topics return “no relevant rule found.”
- Empty queries are rejected before the agent call.
- Missing timestamps or empty filters return explicit no-data messages.
- The agent does not invent numerical values or compliance rules.

## Example Questions

- What is the average power output of T01?
- What is the capacity factor of T05?
- What was the average capacity factor in the second week of the dataset, and how does that compare to the first week?
- On days when DAM price exceeded ₹4/kWh, was turbine output above or below the period average?
- According to the rulebook, what is temporal correlation?
- Does the rulebook specify a battery-storage requirement?

## Setup

1. Activate the provided virtual environment.
2. Ensure the `.env` file contains `GROQ_API_KEY`.
3. Run the agent from `part_a/`.

## Run Instructions

```bash
cd part_a
python agent.py
```

## Known Limitations

- The rulebook tool is intentionally deterministic and short-form; it is not a general document QA system.
- Questions that require electrolyzer consumption values cannot be fully answered unless that data is provided.
- The current implementation focuses on the 14-day sample data set, which is the Part A scope.

## Short Write-Up Draft

This agent was designed to answer operational questions over the supplied turbine telemetry, DAM market prices, and RFNBO compliance extract using a small tool-based architecture. The agent separates data retrieval, deterministic calculations, and compliance lookup so that each part of the answer is grounded in the correct source. LangChain and Groq were used because they provide a current tool-calling workflow with minimal orchestration overhead. The rulebook was handled with a deterministic lookup because the document is short, and a full vector RAG stack would add complexity without improving reliability for this assignment.

The main assumptions are that the supplied 14-day CSVs are the Part A source of truth, the turbine rated capacity is 2000 kW, and missing telemetry should never be treated as zero generation unless the rulebook says so. The main limitation is that compliance answers are only as specific as the supplied rule extract, so unsupported topics should be reported as unavailable rather than inferred. With more time, I would add stronger time-window parsing, a richer test harness around conversational tool traces, a lightweight SQL backend for more complex joins, and a formal evaluation set for correctness and refusal behavior.
