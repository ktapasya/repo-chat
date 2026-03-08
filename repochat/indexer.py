"""Index a Python repository by walking files and orchestrating parsing/chunking."""

from pathlib import Path
from typing import List

from .chunker import Chunker
from .parser import Parser
from .models import Node, Edge, Chunk


class Indexer:
    """Orchestrate repository indexing by walking files and running parsers."""

    def __init__(self, repo_root: str):
        """Initialize the indexer.

        Args:
            repo_root: Path to the repository root.
        """
        self.repo_root = Path(repo_root).resolve()
        self.parser = Parser()
        self.chunker = Chunker()

    def index_repo(self) -> dict:
        """Index all Python files in the repository.

        Returns:
            Summary dict with counts of indexed items.
        """
        all_nodes: List[Node] = []
        all_edges: List[Edge] = []
        all_chunks: List[Chunk] = []
        files_indexed = 0

        # Find all Python files
        python_files = list(self.repo_root.rglob("*.py"))

        # Filter out specific directories
        # Don't use .startswith(".") filter as it's too aggressive for subdirs
        IGNORE_DIRS = {
            "__pycache__", "venv", ".venv", "site-packages",
            ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache",
            ".repochat", "node_modules", ".idea", ".vscode"
        }

        python_files = [
            f for f in python_files
            if not any(part in IGNORE_DIRS for part in f.parts)
        ]

        for file_path in python_files:
            try:
                # Read file content once
                content = file_path.read_text(encoding="utf-8")

                # Use relative path for storage (portable)
                rel_path = str(file_path.relative_to(self.repo_root))

                # Parse structure (pass content to avoid re-reading)
                nodes, edges = self.parser.parse_file(rel_path, content)

                # Create chunks
                chunks = self.chunker.chunk_file(rel_path, content, nodes)

                all_nodes.extend(nodes)
                all_edges.extend(edges)
                all_chunks.extend(chunks)
                files_indexed += 1

            except Exception as e:
                # Log error but continue processing other files
                print(f"Error indexing {file_path}: {e}")
                continue

        return {
            "files_indexed": files_indexed,
            "nodes": all_nodes,
            "edges": all_edges,
            "chunks": all_chunks,
        }

    def index_file(self, file_path: str) -> dict:
        """Index a single file.

        Args:
            file_path: Path to the file to index.

        Returns:
            Dict with nodes, edges, and chunks from the file.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            return {"nodes": [], "edges": [], "chunks": []}

        try:
            # Read file content once
            content = path.read_text(encoding="utf-8")

            # Use relative path for storage (portable)
            rel_path = str(path.relative_to(self.repo_root))

            # Parse structure (pass content to avoid re-reading)
            nodes, edges = self.parser.parse_file(rel_path, content)

            # Create chunks
            chunks = self.chunker.chunk_file(rel_path, content, nodes)

            return {
                "nodes": nodes,
                "edges": edges,
                "chunks": chunks,
            }

        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            return {"nodes": [], "edges": [], "chunks": []}
