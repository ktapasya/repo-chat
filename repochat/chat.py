"""LLM integration and chat logic for repo-chat."""

import os
from pathlib import Path
from typing import Optional

from .search import Retriever
from .storage import Storage


class Chat:
    """Handles chat interactions with the codebase using a local LLM."""

    # Recommended models for local inference
    DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
    ALTERNATIVE_MODEL = "unsloth/Qwen3.5-2B-GGUF"

    SYSTEM_PROMPT = (
        "You answer questions about a codebase.\n"
        "Rules:\n"
        "- Use ONLY the provided code context.\n"
        "- If the answer appears in the code, return the exact value.\n"
        "- Do NOT explain reasoning.\n"
        "- Do NOT say \"to determine\".\n"
        "- If the answer is not visible in the code context, say: I cannot find this in the indexed code.\n"
    )

    def __init__(self, repo_root: Path):
        """Initialize the chat interface.

        Args:
            repo_root: Path to the repository root.
        """
        self.repo_root = repo_root
        self.storage = Storage(repo_root)
        self.retriever = Retriever(repo_root)
        self._llm = None

    def ask(self, question: str, top_k: int = 6) -> dict:
        """Ask a question about the codebase.

        Args:
            question: User's question.
            top_k: Number of code chunks to retrieve for context.

        Returns:
            Dictionary with 'answer' and 'sources' keys.
        """
        # Retrieve relevant context (limit to 3 chunks to avoid overflow)
        context = self.retriever.get_context(question, 4)

        # Get sources (file references)
        sources = self._extract_sources(question, 4)

        # Generate answer
        answer = self._generate_answer(question, context)

        return {
            "answer": answer,
            "sources": sources
        }

    def _extract_sources(self, question: str, top_k: int) -> list:
        """Extract file references from retrieved chunks.

        Args:
            question: User's question.
            top_k: Number of chunks to retrieve.

        Returns:
            List of file reference strings like "auth.py:42-70".
        """
        chunks = self.retriever.retrieve(question, top_k)
        print("Inside chat.py")
        for c in chunks:
            print("----")
            print(c.file_path, c.line_start, c.line_end)
            print(c.content[:400])

        sources = []
        seen = set()  # Deduplicate

        for chunk in chunks:
            ref = f"{chunk.file_path}:{chunk.line_start}-{chunk.line_end}"
            if ref not in seen:
                sources.append(ref)
                seen.add(ref)

        return sources

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM.

        Args:
            question: User's question.
            context: Retrieved code context.

        Returns:
            Generated answer.
        """
        try:
            # Try using llama-cpp-python for GGUF models
            return self._generate_with_llama_cpp(question, context)
        except Exception as e:
            # Log the error for debugging
            import sys
            print(f"LLM generation error: {type(e).__name__}: {e}", file=sys.stderr)

            # Fallback to a simple rule-based response
            return self._generate_fallback(question, context)

    def _generate_with_llama_cpp(self, question: str, context: str) -> str:
        """Generate answer using llama-cpp-python.

        Args:
            question: User's question.
            context: Retrieved code context.

        Returns:
            Generated answer.
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for LLM inference. "
                "Install with: pip install llama-cpp-python"
            )

        # Lazy load the model
        # if self._llm is None:
        self._llm = self._load_llm()

        # Assemble prompt
        prompt = self._assemble_prompt(question, context)

        print("PROMPT TOKENS:", len(prompt.split()))

        # Generate response
        try:
            response = self._llm(
                prompt,
                max_tokens=512,
                stop=["<|im_end|>", "<|endoftext|>", "\n\n\n"],
                echo=False
            )

            # Handle different response formats from llama-cpp-python
            if isinstance(response, str):
                answer = response.strip()
            elif isinstance(response, dict):
                if "choices" in response:
                    answer = response["choices"][0]["text"].strip()
                else:
                    answer = str(response).strip()
            else:
                answer = str(response).strip()

            return answer
        except Exception as e:
            # If LLM generation fails, raise to trigger fallback
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def _load_llm(self):
        """Load the LLM model with auto-download.

        Returns:
            Loaded llama-cpp-python Llama instance.
        """
        model_name = self.DEFAULT_MODEL
        cache_dir = os.path.expanduser("~/.cache/repochat/models")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Use the model name directly as the repo ID (keep -GGUF suffix)
        hf_repo_id = model_name

        try:
            from huggingface_hub import hf_hub_download

            # Download GGUF model - determine filename based on model
            # Most GGUF models use standardized naming
            if "LFM" in hf_repo_id:
                filename = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
            elif "Qwen3.5" in hf_repo_id:
                filename = hf_repo_id.split("/")[-1].replace("-GGUF", "-Q4_K_M.gguf").lower()
            else:
                filename = hf_repo_id.split("/")[-1].replace("-GGUF", "-q4_k_m.gguf").lower()

            # Download GGUF model
            model_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=filename,
                cache_dir=cache_dir,
                resume_download=True
            )

            from llama_cpp import Llama
            llm = Llama(
                model_path=model_path,
                n_ctx=8192,  # Increased from 4096
                n_threads=4,
                verbose=False,
                temperature=0.0  # Deterministic output
            )

            return llm

        except Exception as e:
            raise RuntimeError(
                f"Failed to load LLM model: {e}. "
                "The model will be downloaded automatically on first run."
            ) from e

    def _assemble_prompt(self, question: str, context: str) -> str:
        """Assemble the prompt for the LLM.

        Uses plain text format.

        Args:
            question: User's question.
            context: Retrieved code context.

        Returns:
            Formatted prompt string.
        """
        # Plain text format (not ChatML)
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            "Code context:\n"
        )

        if context:
            prompt += f"{context}\n\n"
            prompt += f"Question:\n{question}\n\n"
        else:
            prompt += f"Question:\n{question}\n\n"

        prompt += "Answer:\n"

        return prompt

    def _generate_fallback(self, question: str, context: str) -> str:
        """Generate a simple answer when LLM is not available.

        Args:
            question: User's question.
            context: Retrieved code context.

        Returns:
            Simple rule-based answer.
        """
        if not context:
            return "I couldn't find relevant code for your question. The repository may not be indexed yet."

        # Simple extraction: return context directly
        return f"Based on the codebase, here's what I found:\n\n{context[:500]}..."
