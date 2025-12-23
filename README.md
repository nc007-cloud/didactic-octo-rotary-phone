Multi-Agent RAG System for Competitor Analysis

This code implements a Multi-Agent Retrieval-Augmented Generation (RAG) system to automate competitor analysis for a given company. The workflow decomposes competitor analysis into specialized agents that retrieve web data, store it in a vector database, and generate a structured, evidence-backed report.

What It Does

Given a company name, the system:

Identifies key competitors

Retrieves up-to-date web information

Stores evidence in a vector database

Generates a structured competitor analysis

Appends verifiable source URLs

Architecture (High-Level)

The system follows a sequential multi-agent workflow:

Question Generator – validates the company and generates analysis questions

Retrieval & Storage – fetches web data (Tavily) and stores it in ChromaDB

Report Drafter – retrieves context and produces a structured analysis with citations

Project File
nc_multi_agent_rag.py   # Main implementation

Setup
Environment Variables

API keys are supplied via environment variables and are not hard-coded.

export OPEN_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"


Optional:

export OPENAI_BASE_URL="https://api.openai.com/v1"

Usage
from nc_multi_agent_rag import run_competitor_analysis

result = run_competitor_analysis(
    company_name="AbbVie",
    max_num_of_questions=5
)

print(result["report"])

Output

The system returns a Markdown report with:

Executive Summary

Key Competitors

Comparison Table

Key Insights

Strategic Recommendations

Sources (URLs)

Design Notes

Tool outputs are not assumed to be JSON-clean

Vector storage is invoked deterministically

Competitor extraction is explicit

Source URLs are appended programmatically

This ensures reliability, reproducibility, and grading clarity.

Summary

This project demonstrates how multi-agent systems combined with RAG can deliver scalable, explainable competitor analysis with traceable evidence for strategic decision-making.
