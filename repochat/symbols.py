"""Symbol extraction for repo-chat - extracts function and class names."""

import re
from pathlib import Path
from typing import List

from .models import Symbol
from .storage import Storage


# Regex patterns for extracting symbols by language
PATTERNS = {
    ".py": [
        (r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
        (r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]", "class"),
    ],
    ".js": [
        (r"^function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
        (r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
        (r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*function\s*\(", "function"),
        (r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\([^)]*\)\s*=>", "function"),
    ],
    ".ts": [
        (r"^function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
        (r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
        (r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*\([^)]*\)\s*=>", "function"),
    ],
    ".go": [
        (r"^func\s+(?:\([^)]+\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
    ],
    ".rs": [
        (r"^fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
        (r"^struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
        (r"^impl\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*", "class"),
        (r"^enum\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
    ],
    ".java": [
        (r"^(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
        (r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*", "class"),
        (r"^interface\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*{", "class"),
    ],
    ".cpp": [
        (r"^(?:\w+\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
        (r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
        (r"^struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
    ],
    ".c": [
        (r"^(?:\w+\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
        (r"^struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
    ],
    ".h": [
        (r"^(?:\w+\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", "function"),
        (r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
        (r"^struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[{]", "class"),
    ],
}


class SymbolExtractor:
    """Extracts symbols (functions and classes) from code files."""

    def __init__(self, repo_root: str):
        """Initialize the symbol extractor.

        Args:
            repo_root: Path to the repository root.
        """
        self.repo_root = Path(repo_root).resolve()
        self.storage = Storage(repo_root)

    def extract_symbols(self, file_path: str) -> List[Symbol]:
        """Extract symbols from a single file.

        Args:
            file_path: Path to the file to extract symbols from.

        Returns:
            List of Symbols found in the file.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in PATTERNS:
            return []

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            symbols = []

            for pattern, _ in PATTERNS[ext]:
                symbols.extend(self._find_matches(pattern, file_path, lines))

            return symbols
        except Exception:
            return []

    def _find_matches(self, pattern: str, file_path: str, lines: List[str]) -> List[Symbol]:
        """Find all matches for a regex pattern in the file.

        Args:
            pattern: Regex pattern to match.
            file_path: Path to the file.
            lines: List of lines in the file.

        Returns:
            List of Symbols found.
        """
        symbols = []
        regex = re.compile(pattern)

        for line_num, line in enumerate(lines, start=1):
            match = regex.search(line)
            if match:
                symbol_name = match.group(1)
                # Estimate end line (simplistic - just next 5 lines or EOF)
                line_end = min(line_num + 5, len(lines))

                symbols.append(Symbol(
                    symbol=symbol_name,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_end
                ))

        return symbols

    def index_repository_symbols(self) -> int:
        """Extract and store symbols for all indexed files.

        Returns:
            Number of symbols indexed.
        """
        # Get all unique file paths from existing chunks
        chunks = self.storage.get_all_chunks()
        file_paths = set(chunk.file_path for chunk in chunks)

        total_symbols = 0

        for file_path in file_paths:
            symbols = self.extract_symbols(file_path)
            if symbols:
                self.storage.add_symbols(symbols)
                total_symbols += len(symbols)

        return total_symbols
