# Local RAG with Ollama — Resume Analyst + Multi-Model Chatbot

A production-oriented local RAG application built with Streamlit, Ollama, and LangChain.

It includes:
- A multi-model RAG chatbot (local chunks + optional live fallback)
- A Resume Analyst mode with ATS checks, skill-gap analysis, keyword optimization, revision drafting, and professional report export
- Structured JSON outputs for frontend integration

![Project Overview](assets/resume-analyst-overview.svg)

## Features

- Multi-model support (`llama3`, `llama2`, `gemma2`, `qwen3`, etc.)
- Fast vectorized similarity search for local retrieval
- PDF drag-and-drop resume upload with text extraction
- Resume analysis with strict schema validation
- Retry/repair + backup-model fallback for robust outputs
- Quality guard against empty/low-value model responses
- Confidence and evidence fields in analysis outputs
- Professional report tab + downloadable markdown report
- Evaluation logging for analysis/revision runs

## Tech Stack

- Python, Streamlit
- Ollama + LangChain
- NumPy vector similarity
- Pydantic schema validation
- PyPDF for PDF parsing

## Prerequisites

- Python `3.11` or `3.12` recommended
- Ollama installed locally

## Setup

```bash
cd Local-RAG-with-Ollama
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Pull required models:

```bash
ollama pull llama3:latest
ollama pull mxbai-embed-large
```

## Run

```bash
streamlit run rag_chatbot.py
```

Open:

`http://localhost:8501`

## Resume Analyst Workflow

1. Switch sidebar mode to **Resume Analyst**
2. Upload resume PDF or paste resume text
3. Add target role, industry, level, and optional JD/company
4. Click **Analyze Resume**
5. Review:
   - Structured JSON
   - Revised Resume
   - Improvement Plan
   - Professional Report

## Project Structure

```text
.
├── rag_chatbot.py
├── data_processor.py
├── examples.py
├── datasets/
├── logs/
├── SETUP_GUIDE.md
├── ARCHITECTURE_EXPLANATION.md
└── requirements.txt
```

## Notes

- If `ollama` is not recognized, restart terminal after installation.
- If model output quality is low, the app auto-retries and uses guarded fallback.
- Evaluation events are stored in `logs/evaluation_runs.jsonl`.

## License

For personal/educational use unless your repository defines a different license policy.
