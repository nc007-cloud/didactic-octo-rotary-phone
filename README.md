Multi-Agent RAG System for Competitor Analysis
Overview:
This project implements a Multi-Agent Retrieval-Augmented Generation (RAG) system to automate competitor analysis for a given company. Traditional competitor analysis is time-consuming and manual, often requiring analysts to gather, validate, and synthesize information from multiple sources. This system decomposes the task into specialized agents that collaborate sequentially to produce a structured, evidence-backed competitor analysis report.
The solution integrates:
  Multi-agent orchestration (LangGraph)
  Web retrieval (Tavily)
  Vector storage and retrieval (ChromaDB)
  LLM-based report generation
  The output is a precise, actionable competitor analysis with explicit citations.

Key Features

✅ Accepts a company name as input
✅ Identifies key competitors
✅ Retrieves up-to-date web data
✅ Stores retrieved content in a vector database
✅ Generates a structured competitor comparison report
✅ Appends deterministic source URLs for traceability

System Architecture
The pipeline is implemented as a sequential multi-agent workflow:
1. Question Generator Agent
    Validates the company name
    Identifies the industry sector
  Generates targeted competitor analysis questions
2. Data Retrieval & Storage Agent
    Uses Tavily to retrieve web-based information
    Stores results in ChromaDB for retrieval-augmented generation.
  Extracts competitor names directly from retrieved content
3. Report Drafter Agent
    Retrieves relevant context from ChromaDB
    Generates a structured competitor analysis report
    Deterministically appends source URLs for reliability

Project Structure
.
├── nc_multi_agent.py      # Main Python implementation
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies (optional)

Installation
1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt


Required libraries include:
  langchain
  langgraph
  chromadb
  tavily
  openai
  pydantic

Environment Variables

 API keys are NOT hard-coded and must be provided via environment variables.

Set the following before running:

export OPEN_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"


Optional (only if using a custom endpoint):

export OPENAI_BASE_URL="https://api.openai.com/v1"


Note: The base URL is intentionally not hard-coded to ensure portability and security when uploading to GitHub.

How to Run
Example usage
from nc_multi_agent import run_nc_multi_agent

result = run__nc_multi_agent(
    company_name="AbbVie",
    max_num_of_questions=5
)

print(result["report"])


The output will be a Markdown-formatted competitor analysis report containing:

Executive Summary

Key Competitors

Comparison Table

Key Insights

Strategic Recommendations

Sources (URLs)

Output Example (Structure)
Competitor Analysis Report
1) Executive Summary
2) Key Competitors
3) Comparison Table
4) Key Insights
5) Strategic Recommendations
6) Sources

Design Notes & Debugging Decisions
    Tool outputs from agents are not assumed to be JSON-clean
    Vector storage is invoked directly to ensure deterministic behavior
    Competitor extraction is done explicitly from the retrieved content
    Source URLs are appended programmatically, not left to the LLM
    This ensures reliability, reproducibility, and rubric compliance

Limitations & Future Improvements
  Market share estimates are qualitative unless structured financial data is added.
  Additional agents could be added for:
  Financial benchmarking
  Regulatory risk analysis
  Time-series competitor tracking



