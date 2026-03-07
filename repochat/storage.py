"""Storage layer for repo-chat using SQLite."""

import json
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import CodeChunk, Symbol


class Storage:
    """Handles all database operations for repo-chat."""

    def __init__(self, repo_root: str):
        """Initialize storage for a repository.

        Args:
            repo_root: Path to the repository root directory.
        """
        self.repo_root = Path(repo_root).resolve()
        self.repochat_dir = self.repo_root / ".repochat"
        self.db_path = self.repochat_dir / "index.sqlite"
        self.config_path = self.repochat_dir / "config.json"

        self._ensure_directories()
        self._ensure_config()
        self._init_db()

    def _ensure_directories(self) -> None:
        """Create .repochat directory if it doesn't exist."""
        self.repochat_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_config(self) -> None:
        """Create default config.json if it doesn't exist."""
        if not self.config_path.exists():
            default_config = {
                "embedding_model": "nomic-embed-text-v1.5",
                "llm_model": "Qwen2.5-3B-Instruct",
                "embedding_dim": 768,
            }
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                is_doc BOOLEAN DEFAULT 0,
                symbol_name TEXT,
                kind TEXT
            )
        """)

        # Create symbols table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                symbol TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                PRIMARY KEY (symbol, file_path)
            )
        """)

        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_file_path
            ON chunks(file_path)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbols_name
            ON symbols(symbol)
        """)

        conn.commit()
        conn.close()

    def add_chunk(self, chunk: CodeChunk) -> int:
        """Add a code chunk to the database.

        Args:
            chunk: CodeChunk to add.

        Returns:
            The ID of the inserted chunk.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO chunks (file_path, line_start, line_end, content, embedding, is_doc, symbol_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (chunk.file_path, chunk.line_start, chunk.line_end,
              chunk.content, chunk.embedding, 1 if chunk.is_doc else 0, chunk.symbol_name))

        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return chunk_id

    def update_chunk_embedding(self, chunk_id: int, embedding: bytes) -> None:
        """Update embedding for an existing chunk.

        Args:
            chunk_id: ID of the chunk to update.
            embedding: Embedding bytes to store.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE chunks
            SET embedding = ?
            WHERE id = ?
        """, (embedding, chunk_id))

        conn.commit()
        conn.close()

    def add_chunks(self, chunks: List[CodeChunk]) -> List[int]:
        """Add multiple code chunks efficiently.

        Args:
            chunks: List of CodeChunks to add.

        Returns:
            List of IDs of inserted chunks.
        """
        if not chunks:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        chunk_ids = []
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO chunks (file_path, line_start, line_end, content, embedding, is_doc, symbol_name, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (chunk.file_path, chunk.line_start, chunk.line_end,
                  chunk.content, chunk.embedding, 1 if chunk.is_doc else 0, chunk.symbol_name, chunk.kind))
            chunk_ids.append(cursor.lastrowid)

        conn.commit()
        conn.close()

        return chunk_ids

    def get_chunk(self, chunk_id: int) -> Optional[CodeChunk]:
        """Get a chunk by ID.

        Args:
            chunk_id: ID of the chunk.

        Returns:
            CodeChunk if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, file_path, line_start, line_end, content, embedding, is_doc, symbol_name, kind
            FROM chunks WHERE id = ?
        """, (chunk_id,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return CodeChunk(
            id=row[0],
            file_path=row[1],
            line_start=row[2],
            line_end=row[3],
            content=row[4],
            embedding=row[5],
            is_doc=bool(row[6]),
            symbol_name=row[7],
            kind=row[8]
        )

    def get_all_chunks(self) -> List[CodeChunk]:
        """Get all chunks from the database.

        Returns:
            List of all CodeChunks.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, file_path, line_start, line_end, content, embedding, is_doc, symbol_name, kind
            FROM chunks
        """)

        chunks = []
        for row in cursor.fetchall():
            chunks.append(CodeChunk(
                id=row[0],
                file_path=row[1],
                line_start=row[2],
                line_end=row[3],
                content=row[4],
                embedding=row[5],
                is_doc=bool(row[6]),
                symbol_name=row[7],
                kind=row[8]
            ))

        conn.close()
        return chunks

    def add_symbol(self, symbol: Symbol) -> None:
        """Add a symbol to the database.

        Args:
            symbol: Symbol to add.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO symbols (symbol, file_path, line_start, line_end)
            VALUES (?, ?, ?, ?)
        """, (symbol.symbol, symbol.file_path, symbol.line_start, symbol.line_end))

        conn.commit()
        conn.close()

    def add_symbols(self, symbols: List[Symbol]) -> None:
        """Add multiple symbols efficiently.

        Args:
            symbols: List of Symbols to add.
        """
        if not symbols:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for symbol in symbols:
            cursor.execute("""
                INSERT OR REPLACE INTO symbols (symbol, file_path, line_start, line_end)
                VALUES (?, ?, ?, ?)
            """, (symbol.symbol, symbol.file_path, symbol.line_start, symbol.line_end))

        conn.commit()
        conn.close()

    def get_symbol(self, symbol_name: str) -> Optional[Symbol]:
        """Get a symbol by name.

        Args:
            symbol_name: Name of the symbol.

        Returns:
            Symbol if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT symbol, file_path, line_start, line_end
            FROM symbols WHERE symbol = ?
        """, (symbol_name,))

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return Symbol(
            symbol=row[0],
            file_path=row[1],
            line_start=row[2],
            line_end=row[3],
        )

    def clear_index(self) -> None:
        """Clear all indexed data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM symbols")

        conn.commit()
        conn.close()

    @property
    def config(self) -> dict:
        """Get the current configuration."""
        with open(self.config_path) as f:
            return json.load(f)

    def update_config(self, **kwargs) -> None:
        """Update configuration values.

        Args:
            **kwargs: Configuration key-value pairs to update.
        """
        config = self.config
        config.update(kwargs)

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
