# Repo Chat

**Chat with your codebase locally.** Zero cloud, minimal setup. Perfect for small codebases.

## Why repo-chat?

- **100% Local**: Everything runs on your machine. No data leaves your computer. No API keys needed.
- **Lightweight**: Minimal dependencies, fast indexing, small resource footprint (~2GB RAM).
- **Code-aware**: Understands functions, classes, and how they connect through graph-augmented retrieval.
- **Simple**: Just run `repochat` in any repository and start chatting.

## Installation

```bash
git clone https://github.com/ktapasya/repo-chat.git
cd repo-chat
pip install -e .
```

## Usage

```bash
cd your-repo
repochat
```

Then open http://localhost:7331 in your browser.

### Options

```bash
repochat --reindex     # Force rebuild index
repochat --port 8080    # Use different port
```

## How it works

1. **Scans** your repository (ignoring `.git`, `node_modules`, etc.)
2. **Chunks** code by functions, classes, and blocks
3. **Embeds** chunks using a local model (`bge-small-en-v1.5`)
4. **Builds** a code graph (function calls, references, contains)
5. **Searches** using vector similarity + graph expansion
6. **Answers** using a local LLM (`LFM2.5-1.2B` Q4_K_M, ~1GB)

## Requirements

- Python 3.10+
- ~2GB RAM
- First run downloads models (~1.5GB total)

## Architecture

Minimal dependencies (~1900 lines):
- `fastapi` - Web server
- `uvicorn` - ASGI server
- `sentence-transformers` - Embeddings
- `numpy` - Vector operations
- `llama-cpp-python` - LLM inference
- `huggingface-hub` - Model downloads

## License

MIT
