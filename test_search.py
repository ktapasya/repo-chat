#!/usr/bin/env python3
"""Test search.py implementation."""

from repochat.search import Search


def main():
    # Initialize search
    search = Search(".")

    # Test queries
    queries = [
        "parse Python files",
        "create embeddings",
        "store data in database",
        "chunk code",
    ]

    print("Testing search functionality\n")

    for query in queries:
        print(f"Query: {query!r}")
        print("-" * 60)

        results = search.search(query, top_k=3)

        if not results:
            print("  No results found")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n  [{i}] Score: {result.score:.4f}")
                print(f"      File: {result.chunk.file_path}")
                print(f"      Lines: {result.chunk.start_line}-{result.chunk.end_line}")
                # Show first line of content
                first_line = result.chunk.content.split('\n')[0]
                print(f"      Preview: {first_line[:80]}...")

        print("\n")

    search.close()


if __name__ == "__main__":
    main()
