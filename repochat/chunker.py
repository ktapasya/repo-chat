"""Chunk code into retrieval units based on parsed structure."""

from typing import List

from .models import Chunk, Node


class Chunker:
    """Split code files into chunks for embedding and retrieval."""

    def chunk_file(
        self,
        file_path: str,
        content: str,
        nodes: List[Node]
    ) -> List[Chunk]:
        """Split a file into chunks based on parsed structure.

        Args:
            file_path: Path to the source file.
            content: Full file content as string.
            nodes: List of parsed nodes from the file (from parser.py).

        Returns:
            List of Chunk objects with line ranges and content.
        """
        lines = content.splitlines()
        chunks: List[Chunk] = []

        # Filter out file node
        # Chunk: methods (dotted names), standalone functions, constants
        # Do NOT chunk full classes (methods are more useful for search)
        code_nodes = [
            n for n in nodes
            if n.type in ["function", "constant"]  # function includes methods
        ]

        if not code_nodes:
            # Fallback: create single chunk for entire file
            chunks.append(Chunk(
                id=None,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                content=content
            ))
            return chunks

        # Sort nodes by line number
        code_nodes.sort(key=lambda n: n.line)

        # Create chunks for each node
        for node in code_nodes:
            # Determine end line
            if node.end_line > 0:
                end_line = node.end_line
            else:
                # Fallback: estimate end line from next node or EOF
                end_line = self._find_end_line(node, code_nodes, len(lines))

            # Extract content
            start_line = node.line
            if start_line <= end_line and start_line <= len(lines):
                chunk_content = "\n".join(lines[start_line-1:end_line])

                chunks.append(Chunk(
                    id=None,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content
                ))

        # Create fallback chunks for uncovered regions
        chunks.extend(self._create_fallback_chunks(
            file_path, lines, code_nodes
        ))

        return chunks

    def _find_end_line(
        self,
        node: Node,
        all_nodes: List[Node],
        file_length: int
    ) -> int:
        """Find the end line for a node.

        Args:
            node: The node to find end line for.
            all_nodes: All sorted nodes in the file.
            file_length: Total lines in the file.

        Returns:
            Estimated end line number.
        """
        # If node has explicit end_line, use it
        if node.end_line > 0:
            return node.end_line

        # Otherwise, estimate from next node's start
        node_idx = all_nodes.index(node)
        if node_idx + 1 < len(all_nodes):
            # End before next node starts (minus 1 for gap)
            next_node = all_nodes[node_idx + 1]
            return max(node.line, next_node.line - 1)

        # Last node: go to end of file
        return file_length

    def _create_fallback_chunks(
        self,
        file_path: str,
        lines: List[str],
        code_nodes: List[Node]
    ) -> List[Chunk]:
        """Create chunks for code not covered by any node.

        Args:
            file_path: Path to the source file.
            lines: File lines.
            code_nodes: Sorted list of code nodes.

        Returns:
            List of fallback chunks for uncovered regions.
        """
        if not code_nodes:
            return []

        chunks = []
        covered_ranges = []

        # Build covered line ranges
        for node in code_nodes:
            end_line = node.end_line if node.end_line > 0 else node.line
            covered_ranges.append((node.line, end_line))

        # Find gaps
        current_line = 1
        for start, end in sorted(covered_ranges):
            if current_line < start:
                # Gap before this node
                gap_content = "\n".join(lines[current_line-1:start-1])
                if gap_content.strip():  # Only non-empty gaps
                    chunks.append(Chunk(
                        id=None,
                        file_path=file_path,
                        start_line=current_line,
                        end_line=start - 1,
                        content=gap_content
                    ))
            current_line = max(current_line, end + 1)

        # Check for trailing gap
        if current_line <= len(lines):
            trailing_content = "\n".join(lines[current_line-1:])
            if trailing_content.strip():
                chunks.append(Chunk(
                    id=None,
                    file_path=file_path,
                    start_line=current_line,
                    end_line=len(lines),
                    content=trailing_content
                ))

        return chunks
