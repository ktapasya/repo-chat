PROMPT.md

Project: repo-chat
Goal: a minimal open source tool that lets anyone chat with their codebase locally with almost zero setup.

The entire system should remain extremely small. Target size roughly 800 to 1200 lines of code total. Avoid unnecessary abstractions and heavy dependencies.

The philosophy of this project is:

• local first
• zero cloud
• minimal dependencies
• simple install
• understandable code
• fast startup

The system should work well for codebases from about 10 files to about 5000 files.

The tool must run entirely locally on macOS, Linux, and ideally Windows.

The user experience must be extremely simple:

Install:

pip install repo-chat

Run inside any repository:

repochat

Then open:

http://localhost:7331

and start chatting with the codebase.

No manual model downloads. Models must automatically download on first run.

No complicated configuration required.

Optional configuration can exist later but not required.

The entire project must stay simple and readable.

⸻

SYSTEM OVERVIEW

The system consists of five components.
	1.	Code indexer
	2.	Embedding generator
	3.	Vector search
	4.	Symbol lookup
	5.	Chat server

The architecture intentionally avoids heavy frameworks and databases.

No FAISS
No LanceDB
No Chroma
No Milvus

Use only:

numpy
sqlite
fastapi
uvicorn
sentence-transformers or compatible small embedding model
huggingface hub auto download

Everything else should be standard Python.

⸻

DIRECTORY STRUCTURE

repochat/

repochat/
init.py
cli.py
server.py
indexer.py
embed.py
search.py
symbols.py
chat.py
models.py
storage.py

frontend/
index.html
app.js
style.css

.repochat/
index.sqlite
config.json

Keep frontend extremely small and dependency free.

⸻

INDEXING PIPELINE

When the user runs repochat inside a repo:

Step 1
Scan repository.

Ignore directories:

.git
node_modules
dist
build
.venv
venv
pycache

Ignore files larger than 1MB.

Index only text files.

Detect code files by extension.

Supported initially:

.py
.ts
.js
.go
.rs
.java
.cpp
.c
.h
.md
.txt

Other files can still be indexed as plain text.

⸻

CHUNKING STRATEGY

Chunk files into meaningful blocks.

Priority order:

1 functions
2 classes
3 fallback text chunks

For Python use AST when possible.

Otherwise fallback to simple heuristics.

Heuristic fallback:

chunk size approx 300 to 800 characters.

Always store metadata:

file path
line start
line end
content

⸻

SYMBOL INDEX

Build a simple symbol table during indexing.

Extract:

function names
class names

Store mapping:

symbol name
file path
line start
line end

Example entry:

login_user → auth_service.py lines 42 to 70

Use simple regex.

Examples:

Python:

def NAME(
class NAME(

JS:

function NAME(
class NAME

Go:

func NAME(

This does not need to be perfect.

Store symbols in sqlite.

⸻

EMBEDDING MODEL

Use a small local embedding model.

Recommended:

nomic-embed-text-v1.5

or

bge-small-en-v1.5

Model must auto download from HuggingFace on first run.

Embedding dimension approx 384.

Embeddings stored as float32.

⸻

DATABASE

Use sqlite.

Tables:

chunks

id
file_path
line_start
line_end
content
embedding BLOB

symbols

symbol
file_path
line_start
line_end

Convert embedding arrays to bytes.

Example:

numpy float32 → bytes

⸻

RETRIEVAL

When a user asks a question:

Step 1

Check if question references a symbol.

Example:

Where is login_user defined?

If symbol exists in symbol table:

directly fetch that code block.

Return it.

Step 2

Otherwise perform vector search.

Process:

embed query

load embeddings

compute cosine similarity

select top 6 chunks.

Combine them into context.

⸻

LLM MODEL

Use a small local LLM.

Default recommendation:

Qwen2.5-3B-Instruct
or
Qwen3-2B-Instruct if available.

Use 4bit quantized version.

Model must auto download.

Inference should work on CPU or small GPU.

Context window minimum 4k.

Prompt template:

System message:

“You are a helpful assistant answering questions about a software codebase. Use the provided code context to answer accurately.”

User message:

question

Context block:

retrieved code chunks

⸻

CHAT FLOW

User question arrives.

Pipeline:

detect symbols
retrieve code if found

otherwise:

embed query
vector search
retrieve top chunks

assemble prompt

call LLM

return answer.

Keep prompt simple.

Do not include unnecessary instructions.

⸻

FASTAPI SERVER

Expose endpoints.

POST /chat

Input:

question

Output:

answer
context sources

GET /

serve static frontend.

⸻

FRONTEND

Very simple UI.

Single page.

Chat input
chat history
scrollable messages.

Display:

assistant answer
file references

Example:

auth_service.py:42-70

Frontend implemented with pure JS.

No frameworks.

⸻

CLI

Command:

repochat

Behavior:

check if index exists

if not:

run indexing

then start server.

Output:

Server running at http://localhost:7331

⸻

PERFORMANCE TARGETS

Indexing:

5000 files should complete under 30 seconds.

Query time:

under 2 seconds typical.

Memory:

less than 2GB.

⸻

MODEL AUTO DOWNLOAD

Use huggingface hub.

First run downloads models into:

~/.cache/repochat/models

Subsequent runs reuse them.

⸻

FAIL SAFE

If embedding model missing:

download automatically.

If LLM missing:

download automatically.

If repo index missing:

create automatically.

Everything must be automatic.

⸻

INSTALLATION

pip install repo-chat

Then run:

repochat

No other setup required.

⸻

FUTURE EXTENSIONS (not required now)

language specific AST parsing
better symbol extraction
reranking
file tree navigation
IDE plugins
git history context

Do not implement these now.

Keep v1 extremely simple.

⸻

FINAL DESIGN PRINCIPLES

Prefer clarity over cleverness.

Avoid large frameworks.

Prefer simple Python and numpy.

Total codebase should remain small enough that a developer can read the entire project in one sitting.

End of specification.
