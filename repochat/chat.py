"""Chat interface for repo-chat."""

from typing import List, Any

from .search import Search
from .models import SearchResult


class Chat:
    """Handle question answering over the codebase."""

    def __init__(self, repo_root: str, llm: Any):
        """Initialize the chat interface.

        Args:
            repo_root: Path to the repository root.
            llm: LLM instance with a generate(prompt) method.
        """
        self.search = Search(repo_root)
        self.llm = llm

    def ask(self, question: str, top_k: int = 6) -> dict:
        """Answer a question about the repository.

        Args:
            question: User's question about the codebase.
            top_k: Number of relevant chunks to retrieve.

        Returns:
            Dict with 'answer' (str) and 'sources' (List[str]).
        """
        # Retrieve relevant code chunks
        results = self.search.search(question, top_k)

        # Build context from retrieved chunks
        context = self._build_context(results)

        print("----- CONTEXT -----")
        print(context)
        print("-------------------")

        # Build prompt with context and question
        prompt = self._build_prompt(question, context)

        # Generate answer using LLM
        answer = self.llm.generate(prompt)

        # Format source references
        sources = [
            f"{r.chunk.file_path}:{r.chunk.start_line}-{r.chunk.end_line}"
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
        }

    def _build_context(self, results: List[SearchResult]) -> str:
        """Build context string from search results.

        Args:
            results: List of search results with chunks and scores.

        Returns:
            Formatted context string with code chunks.
        """
        # Step 1: Merge adjacent/overlapping chunks from same file
        merged_chunks = self._merge_nearby_chunks(results)

        # Step 2: Format merged chunks
        parts = []
        for chunk in merged_chunks:
            parts.append(f"{'#'*40}\n### {chunk['file_path']}\n{'#'*40}\n{chunk['content']}")

        return "\n\n".join(parts)

    def _merge_nearby_chunks(self, results: List[SearchResult]) -> List[dict]:
        """Merge chunks by file: split by lines, dedupe by line number, sort, join.

        Args:
            results: List of search results.

        Returns:
            List of merged chunk dicts with keys: file_path, content.
        """
        if not results:
            return []

        from collections import defaultdict

        # Group chunks by file
        chunks_by_file = defaultdict(list)
        for result in results:
            chunks_by_file[result.chunk.file_path].append(result.chunk)

        merged = []

        # Process each file's chunks
        for file_path, chunks in chunks_by_file.items():
            # Split all chunks into (line_number, line_content) pairs
            all_lines = []
            for chunk in chunks:
                lines = chunk.content.split('\n')
                for i, line in enumerate(lines, start=chunk.start_line):
                    all_lines.append((i, line))

            # Sort by line number
            all_lines.sort(key=lambda x: x[0])

            # Dedupe by line number (keep first occurrence)
            seen = set()
            unique_lines = []
            for line_num, line in all_lines:
                if line_num not in seen:
                    seen.add(line_num)
                    unique_lines.append((line_num, line))

            # Build content with ... for gaps
            content_parts = []
            for i, (line_num, line) in enumerate(unique_lines):
                # Add ellipsis if there's a gap from previous line
                if i > 0 and line_num > unique_lines[i-1][0] + 1:
                    content_parts.append("...")
                # Add line number
                content_parts.append(f"{line_num}: {line}")

            merged.append({
                'file_path': file_path,
                'content': '\n'.join(content_parts)
            })

        return merged

    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for the LLM.

        Args:
            question: User's question.
            context: Retrieved code context.

        Returns:
            Complete prompt string.
        """
        return f"""You answer questions about a codebase.

You can take help from the provided code context. Understand it because the answer usually lives in it.

If the answer is a specific value, return ONLY the value.

If the question asks how something works, explain the steps in the code.

If the answer is not present, say:
I cannot find this in the indexed code.

If a statement comes from the context, cite the corresponding number
Example:

Question: What database is used?

Answer:
The system uses the postgres database [1].

Context:
[1] config.py:10-12
DB_ENGINE = “postgres”

Now answer the real question.

Code context:
{context}

Question:
{question}

Answer:
"""

    def close(self):
        """Close the search connection."""
        self.search.close()
