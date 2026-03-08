#!/usr/bin/env python3
"""Full pipeline test: index → embed → search."""

from repochat.indexer import Indexer
from repochat.embed import Embedder
from repochat.storage import Storage
from repochat.search import Search


def main():
    print("=" * 60)
    print("REPO-CHAT FULL PIPELINE TEST")
    print("=" * 60)
    print()

    # Step 1: Index repository
    print("Step 1: Indexing repository...")
    print("-" * 60)

    indexer = Indexer(".")
    result = indexer.index_repo()

    print(f"✓ Indexed {result['files_indexed']} files")
    print(f"  - {len(result['nodes'])} nodes")
    print(f"  - {len(result['edges'])} edges")
    print(f"  - {len(result['chunks'])} chunks")
    print()

    # Step 2: Store in database
    print("Step 2: Storing in database...")
    print("-" * 60)

    storage = Storage(".")
    storage.clear()

    storage.insert_nodes(result["nodes"])
    storage.insert_edges(result["edges"])
    storage.insert_chunks(result["chunks"])

    print("✓ Stored nodes, edges, and chunks")
    print()

    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    print("-" * 60)

    embedder = Embedder()
    chunks = storage.get_all_chunks()

    print(f"Generating embeddings for {len(chunks)} chunks...")

    # Extract chunk contents
    texts = [chunk.content for chunk in chunks]

    # Batch embed
    embeddings = embedder.embed_batch(texts)

    # Update chunks with embeddings
    for chunk, embedding in zip(chunks, embeddings):
        storage.update_chunk_embedding(chunk.id, embedding)

    print(f"✓ Generated and stored {len(embeddings)} embeddings")
    print()

    # Verify embeddings
    sample_embedding = embeddings[0]
    import numpy as np
    vec = np.frombuffer(sample_embedding, dtype=np.float32)
    print(f"  Embedding dimensions: {vec.shape[0]}")
    print(f"  Embedding size: {len(sample_embedding)} bytes")
    print()

    # Step 4: Test search
    print("Step 4: Testing semantic search...")
    print("-" * 60)

    search = Search(".")

    test_queries = [
        "parse Python files",
        "create embeddings",
        "store data in database",
        "search code",
    ]

    for query in test_queries:
        print(f"\nQuery: {query!r}")
        print("-" * 40)

        results = search.search(query, top_k=3)

        if not results:
            print("  No results found")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n  [{i}] Score: {result.score:.4f}")
                print(f"      File: {result.chunk.file_path}")
                print(f"      Lines: {result.chunk.start_line}-{result.chunk.end_line}")

                # Show first 2 lines of content
                lines = result.chunk.content.split('\n')[:2]
                preview = '\n      '.join(lines)
                print(f"      Preview:\n      {preview}...")

    print()
    print("=" * 60)
    print("✓ FULL PIPELINE TEST COMPLETE")
    print("=" * 60)

    # Cleanup
    search.close()
    storage.close()


if __name__ == "__main__":
    main()
