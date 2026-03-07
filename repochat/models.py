"""Data models for repo-chat."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CodeChunk:
    """A chunk of code from a file."""

    id: Optional[int]
    file_path: str
    line_start: int
    line_end: int
    content: str
    embedding: Optional[bytes] = None
    is_doc: bool = False  # Flag for documentation chunks
    symbol_name: Optional[str] = None  # Function/class/constant name
    kind: Optional[str] = None  # Type: constant, function, class, document


@dataclass
class Symbol:
    """A symbol (function/class) definition in the codebase."""

    symbol: str
    file_path: str
    line_start: int
    line_end: int
