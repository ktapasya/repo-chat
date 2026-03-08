"""CLI entry point for repo-chat."""

import sys
from pathlib import Path

import click

from .indexer import Indexer
from .embed import Embedder
from .storage import Storage
from .server import run_server


@click.command()
@click.option(
    "--reindex",
    is_flag=True,
    help="Force reindexing even if index exists"
)
@click.option(
    "--port",
    default=7331,
    help="Port to run the server on (default: 7331)"
)
def main(reindex: bool, port: int):
    """Chat with your codebase locally.

    Run this command inside any repository to start the chat server.
    """
    repo_root = str(Path.cwd())

    # Initialize storage
    storage = Storage(repo_root)

    # Check if index exists
    chunks = storage.get_all_chunks()
    needs_index = reindex or len(chunks) == 0

    if needs_index:
        print("Indexing repository...")
        print(f"Root: {repo_root}")

        # Clear existing index for reindex
        if reindex:
            storage.clear()
            print("✓ Cleared existing index")

        # Step 1: Index files
        indexer = Indexer(repo_root)
        result = indexer.index_repo()

        # Store parsed data
        storage.insert_nodes(result["nodes"])
        storage.insert_edges(result["edges"])
        storage.insert_chunks(result["chunks"])
        print(f"✓ Indexed {result['files_indexed']} files")

        # Step 3: Generate embeddings
        embedder = Embedder()
        print(f"✓ Loading embedding model: {embedder.model_name}")

        # Get chunks without embeddings
        chunks_to_embed = [
            chunk for chunk in storage.get_all_chunks()
            if chunk.embedding is None
        ]

        if chunks_to_embed:
            print(f"✓ Generating embeddings for {len(chunks_to_embed)} chunks...")
            # Prepend filename for better embedding context
            texts = [
                f"FILE: {Path(chunk.file_path).name}\n\n{chunk.content}"
                for chunk in chunks_to_embed
            ]
            embeddings = embedder.embed_batch(texts)

            # Update chunks with embeddings
            for chunk, embedding in zip(chunks_to_embed, embeddings):
                storage.update_chunk_embedding(chunk.id, embedding)

        print("✓ Indexing complete!")
    else:
        print(f"✓ Using existing index ({len(chunks)} chunks)")
        print(f"  Run with --reindex to rebuild")

    # Start server
    run_server(repo_root, port)


if __name__ == "__main__":
    main()
