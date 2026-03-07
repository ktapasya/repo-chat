"""
Code indexer for repo-chat.

Uses lightweight chunker to split files into semantic chunks.
"""

from pathlib import Path
from typing import List, Set
from repochat.chunker import chunk_file, CodeChunk
from repochat.storage import Storage


# ------------------------
# CONFIGURATION
# ------------------------

MAX_FILE_SIZE = 1024 * 1024  # 1MB
IGNORE_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.repochat'}


class Indexer:
    """Index code repository by splitting files into chunks."""

    def __init__(self, repo_root: str):
        """Initialize the indexer.

        Args:
            repo_root: Root directory of the repository.
        """
        self.repo_root = Path(repo_root)
        self.storage = Storage(repo_root)

    def index_repository(self) -> int:
        """Index all files in the repository.

        Returns:
            Number of files indexed.
        """
        files_indexed = 0

        for file_path in self._scan_files():
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = chunk_file(text, str(file_path))

                if chunks:
                    # Convert CodeChunk dataclass to storage format
                    from repochat.models import CodeChunk as StorageChunk
                    storage_chunks = [
                        StorageChunk(
                            id=None,
                            file_path=chunk.file_path,
                            line_start=chunk.line_start,
                            line_end=chunk.line_end,
                            content=chunk.content,
                            embedding=None,
                            is_doc=(chunk.kind == "document"),
                            symbol_name=chunk.symbol,
                            kind=chunk.kind
                        )
                        for chunk in chunks
                    ]
                    self.storage.add_chunks(storage_chunks)
                    files_indexed += 1
            except Exception:
                # Skip files that cause errors during parsing
                continue

        return files_indexed

    def _scan_files(self) -> List[Path]:
        """Scan repository for files to index.

        Returns:
            List of file paths to index.
        """
        files = []

        for file_path in self.repo_root.rglob("*"):
            # Skip ignored directories
            if any(ignored in file_path.parts for ignored in IGNORE_DIRS):
                continue

            # Skip directories and non-files
            if not file_path.is_file():
                continue

            # Skip files that are too large
            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue

            # Skip binary files (simple heuristic)
            if not self._is_text_file(file_path):
                continue

            files.append(file_path)

        return files

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if a file is text-based.

        Args:
            file_path: Path to check.

        Returns:
            True if text file, False otherwise.
        """
        # Try to read first 8KB and check for null bytes
        try:
            content = file_path.read_bytes()[:8192]
            return b'\x00' not in content
        except Exception:
            return False
