#  LLM assisted code optimizer using LangGraph(Streamlit)

This folder contains a Streamlit app (`app.py`) and analysis logic (`slicer.py`) to inspect an LLVM `print-callgraph` output for **sqlite3.c**, Sqlite source code and prompt to  decide whether a target function can be safely removed or not.

## Features

- Parse LLVM call graph (`callgraph.txt`) and compute **direct** and **transitive** callers
- Static decision: **SAFE_TO_REMOVE** or **DEPENDENT** based on callers
- LLM decision: **REMOVE** / **DO_NOT_REMOVE** using call graph facts + code snippet + prompt
- One-click removal that writes a **modified copy** of `sqlite3.c` (downloadable)

## Requirements

- Python 3.9+
- Packages:
  - `streamlit`
  - `openai`
  - `langgraph` ( used in `slicer.py`)
  - `tiktoken`
- OpenAI API key (for LLM decision)

Install (recommended inside a venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# optional if checkpointing is enabled
python3 -m pip install langgraph-checkpoint-sqlite aiosqlite
```

OpenAI credentials (in the same shell  used to run Streamlit):

```bash
export OPENAI_API_KEY="sk-... key ..."
export MODEL="gpt-3.5-turbo"     # or another model you have access to
```

Run the app:

```bash
python3 -m streamlit run app.py
```

This will generate a printed URL.Open the URL in any browser.
