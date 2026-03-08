"""Core data models for repo-chat."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Chunk:
    """A chunk of code from a file.

    Attributes:
        id: Unique identifier (database ID or hash).
        file_path: Path to the source file.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        content: The actual code/content text.
        embedding: Vector embedding as bytes (optional).
    """
    id: Optional[int] = None
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    embedding: Optional[bytes] = None


@dataclass
class Node:
    """A node in the code graph (function, class, file, constant, etc.).

    Attributes:
        id: Unique identifier.
        type: Node type (function, class, file, constant, variable, etc.).
        name: Symbol name.
        file: Path to the source file.
        line: Line number where defined.
    """
    id: Optional[int] = None
    type: str = ""  # function, class, file, constant, variable, etc.
    name: str = ""
    file: str = ""
    line: int = 0
    end_line: int = 0  # For functions, classes, etc.


@dataclass
class Edge:
    """A relationship edge between two nodes.

    Attributes:
        source: Source node ID (None if not yet resolved).
        target: Target node ID (None if not yet resolved).
        type: Relationship type (CALLS, IMPORTS, CONTAINS, INHERITS, SIMILAR, etc.).
        source_name: Source node name (for unresolved edges).
        target_name: Target node name (for unresolved edges).
    """
    source: Optional[int] = None
    target: Optional[int] = None
    type: str = ""  # CALLS, IMPORTS, CONTAINS, INHERITS, SIMILAR, etc.
    source_name: Optional[str] = None
    target_name: Optional[str] = None


@dataclass
class SearchResult:
    """A search result with relevance score.

    Attributes:
        chunk: The matched code chunk.
        score: Relevance score (0-1, higher is better).
    """
    chunk: Chunk
    score: float
