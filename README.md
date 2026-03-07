# repo-chat

Chat with your codebase locally. Zero cloud, minimal setup.

## Features

- **Local-first**: Everything runs on your machine. No data leaves your computer.
- **Zero configuration**: Just run `repochat` in any repository.
- **Fast**: Indexes 5000 files in under 30 seconds.
- **Code-aware**: Understands functions, classes, and code structure.
- **Web interface**: Simple chat UI at http://localhost:7331

## Installation

```bash
pip install repo-chat
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
2. **Chunks** code by functions, classes, and text blocks
3. **Embeds** chunks using a local model (nomic-embed-text-v1.5)
4. **Searches** using vector similarity for your questions
5. **Answers** using a local LLM (Qwen2.5)

## Requirements

- Python 3.8+
- ~2GB RAM
- First run downloads models (~500MB)

## Architecture

Minimal dependencies:
- `fastapi` - Web server
- `uvicorn` - ASGI server
- `sentence-transformers` - Embeddings
- `numpy` - Vector operations
- `llama-cpp-python` - LLM inference
- `huggingface-hub` - Model downloads

Total codebase: ~1000 lines

## License

MIT
