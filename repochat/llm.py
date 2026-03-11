"""Local LLM integration using llama-cpp-python with GGUF models."""

import os
from pathlib import Path


# Recommended models for local inference
DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
ALTERNATIVE_MODEL = "unsloth/Qwen3.5-2B-GGUF"


class LocalLLM:
    """Local LLM using llama-cpp-python with GGUF models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        n_ctx: int = 8192,
        n_threads: int = 4,
        temperature: float = 0.0
    ):
        """Initialize the local LLM.

        Args:
            model_name: HuggingFace model ID (GGUF format).
            n_ctx: Context window size.
            n_threads: Number of CPU threads for inference.
            temperature: Sampling temperature (0.0 = deterministic).
        """
        self.model_name = model_name
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self._llm = None

    def generate_stream(self, prompt: str, max_tokens: int = 1024):
        """Generate text from a prompt with streaming.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.

        Yields:
            Text chunks as they are generated.
        """
        # Lazy-load the model on first use
        if self._llm is None:
            self._llm = self._load_llm()

        try:
            self._llm.reset()

            stream = self._llm.create_completion(
                prompt,
                max_tokens=max_tokens,
                stop=["</s>", "###"],
                echo=False,
                temperature=self.temperature,
                stream=True
            )

            for chunk in stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield text

        except Exception as e:
            raise RuntimeError(f"LLM streaming failed: {e}") from e

    def _load_llm(self):
        """Load the LLM model with auto-download.

        Returns:
            Loaded llama-cpp-python Llama instance.
        """
        # Use the model name directly as the repo ID
        hf_repo_id = self.model_name

        try:
            from huggingface_hub import hf_hub_download

            # Download GGUF model - determine filename based on model
            if "LFM" in hf_repo_id:
                filename = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
            elif "Qwen3.5" in hf_repo_id:
                filename = hf_repo_id.split("/")[-1].replace("-GGUF", "-Q4_K_M.gguf").lower()
            else:
                filename = hf_repo_id.split("/")[-1].replace("-GGUF", "-q4_k_m.gguf").lower()

            # Download GGUF model
            print(f"Downloading model {hf_repo_id}...")
            model_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=filename,
                resume_download=True
            )
            print(f"Model loaded from {model_path}")

            from llama_cpp import Llama

            # Detect GPU support
            n_gpu_layers = 0
            try:
                # Try to offload 1 layer to test GPU support
                test_llm = Llama(
                    model_path=model_path,
                    n_ctx=512,
                    n_gpu_layers=1,
                    verbose=False
                )
                n_gpu_layers = -1  # GPU works, offload all layers
                print("🚀 GPU acceleration enabled")
                del test_llm
            except Exception:
                n_gpu_layers = 0  # Use CPU
                print("⚠️ GPU not available, using CPU for inference")

            llm = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )

            return llm

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import llama-cpp-python: {e}. "
                "Install it with: pip install llama-cpp-python huggingface-hub"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LLM model: {e}. "
                "The model will be downloaded automatically on first run."
            ) from e
