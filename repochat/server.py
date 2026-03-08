"""FastAPI server for repo-chat web interface."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .chat import Chat
from .llm import LocalLLM


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    question: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str
    sources: list[str]


def create_app(repo_root: Optional[str] = None, llm_model: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        repo_root: Path to the repository root. Defaults to current directory.
        llm_model: Optional model name for local LLM. If None, uses LocalLLM default.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="repo-chat", version="0.1.0")

    # Use current directory if not specified
    if repo_root is None:
        repo_root = str(Path.cwd())

    # Frontend directory
    frontend_dir = Path(__file__).parent.parent / "frontend"

    # Initialize LLM
    if llm_model is None:
        llm = LocalLLM()
    else:
        llm = LocalLLM(model_name=llm_model)

    # Initialize chat interface
    chat = Chat(repo_root, llm)

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the static frontend."""
        index_file = frontend_dir / "index.html"
        if not index_file.exists():
            return HTMLResponse("<h1>Frontend not found</h1>")

        return HTMLResponse(index_file.read_text())

    @app.get("/style.css")
    async def style_css():
        """Serve CSS file."""
        style_file = frontend_dir / "style.css"
        if not style_file.exists():
            raise HTTPException(status_code=404, detail="CSS not found")
        return FileResponse(style_file, media_type="text/css")

    @app.get("/app.js")
    async def app_js():
        """Serve JavaScript file."""
        js_file = frontend_dir / "app.js"
        if not js_file.exists():
            raise HTTPException(status_code=404, detail="JS not found")
        return FileResponse(js_file, media_type="application/javascript")

    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """Process a chat question.

        Args:
            request: Chat request with question.

        Returns:
            Chat response with answer and sources.

        Raises:
            HTTPException: If processing fails.
        """
        try:
            response = chat.ask(request.question)
            return ChatResponse(**response)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process question: {str(e)}"
            )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    return app


def run_server(repo_root: Optional[str] = None, port: int = 7331):
    """Run the FastAPI server.

    Args:
        repo_root: Path to the repository root.
        port: Port to run the server on.
    """
    import uvicorn

    app = create_app(repo_root)

    print(f"Starting repo-chat server on http://localhost:{port}")
    print(f"Repository: {repo_root or Path.cwd()}")
    print("Press Ctrl+C to stop")

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run_server()
