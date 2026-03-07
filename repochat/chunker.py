"""
Code chunker for repo-chat.

Lightweight, multi-language code chunking with:
- AST-based extraction for Python
- Regex-based extraction for other languages
- Size cap enforcement
- Deduplication
- Safe structural splitting

Dependencies: ast, re, dataclasses (all stdlib)
"""

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

# ------------------------
# CONFIGURATION
# ------------------------

MAX_CHUNK = 800
MIN_CHUNK = 120

CODE_EXTENSIONS = {'.py', '.js', '.ts', '.go', '.rs', '.c', '.cpp', '.h', '.hpp', '.java'}
DOC_EXTENSIONS = {'.md', '.txt', '.rst', '.org'}


@dataclass
class CodeChunk:
    """A chunk of code from a file."""
    file_path: str
    content: str
    line_start: int
    line_end: int
    symbol: Optional[str] = None
    kind: Optional[str] = None


# ------------------------
# PYTHON AST PARSER
# ------------------------

def _find_with_decorators(node: ast.AST, lines: List[str]) -> Tuple[int, int]:
    """Find the actual start line including decorators.

    Args:
        node: AST node.
        lines: File lines.

    Returns:
        Tuple of (start_line, end_line).
    """
    start = node.lineno
    if hasattr(node, 'decorator_list') and node.decorator_list:
        # Start from first decorator
        start = min(dec.lineno for dec in node.decorator_list)

    end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start
    return (start, end)


def python_symbols(text: str, file_path: str) -> List[CodeChunk]:
    """Extract symbols (classes, functions, assignments, imports) from Python code using AST.

    Args:
        text: Python file content.
        file_path: Path to the file.

    Returns:
        List of CodeChunks for all top-level Python constructs.
    """
    chunks = []

    try:
        tree = ast.parse(text)
    except Exception:
        return chunks

    lines = text.splitlines()

    # Only traverse top-level nodes to avoid nested duplicates
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _find_with_decorators(node, lines)
            content = "\n".join(lines[start - 1:end])

            chunks.append(CodeChunk(
                file_path=file_path,
                content=content,
                line_start=start,
                line_end=end,
                symbol=node.name,
                kind="function"
            ))

        elif isinstance(node, ast.ClassDef):
            start, end = _find_with_decorators(node, lines)
            content = "\n".join(lines[start - 1:end])

            chunks.append(CodeChunk(
                file_path=file_path,
                content=content,
                line_start=start,
                line_end=end,
                symbol=node.name,
                kind="class"
            ))

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Extract top-level assignments (constants, variables)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]

            for target in targets:
                if isinstance(target, ast.Name):
                    symbol_name = target.id
                    start = node.lineno
                    end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start
                    content = "\n".join(lines[start - 1:end])

                    chunks.append(CodeChunk(
                        file_path=file_path,
                        content=content,
                        line_start=start,
                        line_end=end,
                        symbol=symbol_name,
                        kind="constant"
                    ))

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # Extract imports
            start = node.lineno
            end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start
            content = "\n".join(lines[start - 1:end])

            # Create a symbol name from the import
            if isinstance(node, ast.Import):
                symbol_name = node.names[0].name if node.names else "import"
            else:  # ImportFrom
                symbol_name = node.module or "from_import"

            chunks.append(CodeChunk(
                file_path=file_path,
                content=content,
                line_start=start,
                line_end=end,
                symbol=symbol_name,
                kind="import"
            ))

    return chunks


# ------------------------
# REGEX-BASED EXTRACTION (for non-Python)
# ------------------------

FUNC_PATTERNS = [
    r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",  # JavaScript function
    r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",   # Python fallback
    r"fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",    # Rust function
    r"func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",   # Go function
]

CLASS_PATTERNS = [
    r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\{?",  # JavaScript/TypeScript class
]

CONST_PATTERN = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*=", re.MULTILINE)


def _count_brackets(text: str, open_char: str, close_char: str) -> int:
    """Count bracket depth in text.

    Args:
        text: Text to analyze.
        open_char: Opening bracket character.
        close_char: Closing bracket character.

    Returns:
        Net bracket count (open - close).
    """
    return text.count(open_char) - text.count(close_char)


def extract_constants_safe(text: str, file_path: str) -> List[CodeChunk]:
    """Extract constant definitions using safe bracket counting.

    Args:
        text: File content.
        file_path: Path to the file.

    Returns:
        List of CodeChunks for constants.
    """
    chunks = []
    lines = text.splitlines()

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        match = CONST_PATTERN.match(line_stripped)

        if match:
            symbol_name = match.group(1)
            content = line_stripped
            line_end = i

            # Check if it's a multi-line assignment by counting brackets
            if any(c in line_stripped for c in ['(', '[', '{']):
                bracket_count = _count_brackets(line_stripped, '(', ')')
                bracket_count += _count_brackets(line_stripped, '[', ']')
                bracket_count += _count_brackets(line_stripped, '{', '}')

                # Find closing brackets
                for j in range(i, len(lines)):
                    bracket_count += _count_brackets(lines[j], '(', ')')
                    bracket_count += _count_brackets(lines[j], '[', ']')
                    bracket_count += _count_brackets(lines[j], '{', '}')

                    if bracket_count == 0:
                        line_end = j + 1
                        content = "\n".join(lines[i - 1:line_end])
                        break

            chunks.append(CodeChunk(
                file_path=file_path,
                content=content,
                line_start=i,
                line_end=line_end,
                symbol=symbol_name,
                kind="constant"
            ))

    return chunks


def regex_symbols_safe(text: str, file_path: str) -> List[CodeChunk]:
    """Extract symbols using regex with deduplication by line ranges.

    Args:
        text: Code content.
        file_path: Path to the file.

    Returns:
        List of CodeChunks for functions and classes.
    """
    chunks = []
    lines = text.splitlines()
    seen_ranges: Set[Tuple[int, int]] = set()

    # Extract functions
    for pattern in FUNC_PATTERNS:
        for m in re.finditer(pattern, text, re.MULTILINE):
            start_line = text[:m.start()].count("\n") + 1
            end_line = min(start_line + 40, len(lines))

            # Skip if this range overlaps with already extracted
            if any(start_line <= existing_end and end_line >= existing_start
                   for existing_start, existing_end in seen_ranges):
                continue

            seen_ranges.add((start_line, end_line))
            content = "\n".join(lines[start_line - 1:end_line])

            chunks.append(CodeChunk(
                file_path=file_path,
                content=content,
                line_start=start_line,
                line_end=end_line,
                symbol=m.group(1),
                kind="function"
            ))

    # Extract classes
    for pattern in CLASS_PATTERNS:
        for m in re.finditer(pattern, text, re.MULTILINE):
            start_line = text[:m.start()].count("\n") + 1
            end_line = min(start_line + 100, len(lines))

            # Skip if this range overlaps with already extracted
            if any(start_line <= existing_end and end_line >= existing_start
                   for existing_start, existing_end in seen_ranges):
                continue

            seen_ranges.add((start_line, end_line))
            content = "\n".join(lines[start_line - 1:end_line])

            chunks.append(CodeChunk(
                file_path=file_path,
                content=content,
                line_start=start_line,
                line_end=end_line,
                symbol=m.group(1),
                kind="class"
            ))

    return chunks


# ------------------------
# DOCUMENT CHUNKING
# ------------------------

def chunk_document(text: str, file_path: str) -> List[CodeChunk]:
    """Chunk documentation by paragraphs with correct line numbers.

    Args:
        text: Document content.
        file_path: Path to the file.

    Returns:
        List of CodeChunks for document paragraphs.
    """
    chunks = []
    lines = text.splitlines()
    current_para = []
    para_start = 1

    for i, line in enumerate(lines, 1):
        if line.strip() == "":
            # Empty line - end of paragraph
            if current_para:
                content = "\n".join(current_para)
                if len(content) >= MIN_CHUNK:
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        content=content,
                        line_start=para_start,
                        line_end=i - 1,
                        kind="document"
                    ))
                current_para = []
                para_start = i + 1
        else:
            current_para.append(line)

    # Don't forget the last paragraph
    if current_para:
        content = "\n".join(current_para)
        if len(content) >= MIN_CHUNK:
            chunks.append(CodeChunk(
                file_path=file_path,
                content=content,
                line_start=para_start,
                line_end=len(lines),
                kind="document"
            ))

    return chunks


# ------------------------
# SIZE ENFORCEMENT
# ------------------------

def enforce_size_safe(chunk: CodeChunk) -> List[CodeChunk]:
    """Enforce size cap on chunks by splitting at line boundaries.

    Args:
        chunk: Input chunk to potentially split.

    Returns:
        List of CodeChunks (may be 1 or more).
    """
    if len(chunk.content) <= MAX_CHUNK:
        return [chunk]

    lines = chunk.content.splitlines()
    result = []
    current_chunk = []
    current_size = 0
    line_start = chunk.line_start

    for i, line in enumerate(lines):
        line_size = len(line) + 1  # +1 for newline
        if current_size + line_size > MAX_CHUNK and current_chunk:
            # Save current chunk and start new one
            result.append(CodeChunk(
                file_path=chunk.file_path,
                content="\n".join(current_chunk),
                line_start=line_start,
                line_end=line_start + len(current_chunk) - 1,
                symbol=chunk.symbol,
                kind=chunk.kind
            ))
            current_chunk = [line]
            line_start = chunk.line_start + i
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    # Don't forget remaining lines
    if current_chunk:
        result.append(CodeChunk(
            file_path=chunk.file_path,
            content="\n".join(current_chunk),
            line_start=line_start,
            line_end=line_start + len(current_chunk) - 1,
            symbol=chunk.symbol,
            kind=chunk.kind
        ))

    return result


# ------------------------
# MAIN CHUNKING ENTRY
# ------------------------

def chunk_file(text: str, file_path: str) -> List[CodeChunk]:
    """Chunk a file based on its type.

    Args:
        text: File content.
        file_path: Path to the file.

    Returns:
        List of deduplicated, size-enforced CodeChunks.
    """
    chunks = []

    # Python: use AST for everything (constants, functions, classes, imports)
    if file_path.endswith(".py"):
        chunks.extend(python_symbols(text, file_path))

    # Other code files: use regex for constants + functions/classes
    elif any(file_path.endswith(ext) for ext in CODE_EXTENSIONS):
        chunks.extend(extract_constants_safe(text, file_path))
        chunks.extend(regex_symbols_safe(text, file_path))

    # Documentation: use paragraph-based chunking with correct line numbers
    elif any(file_path.endswith(ext) for ext in DOC_EXTENSIONS):
        chunks.extend(chunk_document(text, file_path))

    # Filter out tiny chunks
    chunks = [c for c in chunks if len(c.content) >= MIN_CHUNK]

    # Deduplicate by (file_path, line_start, line_end)
    seen = set()
    deduped = []
    for chunk in chunks:
        key = (chunk.file_path, chunk.line_start, chunk.line_end)
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    # Enforce size cap with safe line-based splitting
    final = []
    for chunk in deduped:
        final.extend(enforce_size_safe(chunk))

    return final
