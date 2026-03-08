"""SQLite storage layer for repo-chat graph and chunks."""

import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import Node, Edge, Chunk


class Storage:
    """Persist nodes, edges, and chunks in SQLite."""

    def __init__(self, repo_root: str):
        """Initialize storage for a repository.

        Args:
            repo_root: Path to the repository root.
        """
        db_dir = Path(repo_root) / ".repochat"
        db_dir.mkdir(exist_ok=True)

        self.db_path = db_dir / "index.db"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cur = self.conn.cursor()

        # Nodes table: stores code entities (classes, functions, etc.)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            name TEXT,
            file TEXT,
            line INTEGER,
            end_line INTEGER
        )
        """)

        # Edges table: stores relationships between nodes
        cur.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source INTEGER,
            target INTEGER,
            type TEXT,
            source_name TEXT,
            target_name TEXT
        )
        """)

        # Chunks table: stores retrievable code chunks
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            start_line INTEGER,
            end_line INTEGER,
            content TEXT,
            embedding BLOB
        )
        """)

        # Create indexes for faster lookups
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)")

        self.conn.commit()

    # -----------------------
    # NODE STORAGE
    # -----------------------

    def insert_nodes(self, nodes: List[Node]) -> None:
        """Insert nodes and assign IDs.

        Args:
            nodes: List of nodes to insert. Their id field will be updated.
        """
        cur = self.conn.cursor()

        for node in nodes:
            cur.execute(
                "INSERT INTO nodes (type, name, file, line, end_line) VALUES (?, ?, ?, ?, ?)",
                (node.type, node.name, node.file, node.line, node.end_line)
            )
            node.id = cur.lastrowid

        self.conn.commit()

    def get_node_by_name(self, name: str) -> Optional[Node]:
        """Get a node by its name.

        Args:
            name: Node name to look up.

        Returns:
            Node if found, None otherwise.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM nodes WHERE name = ?", (name,))
        row = cur.fetchone()

        if row is None:
            return None

        return Node(
            id=row["id"],
            type=row["type"],
            name=row["name"],
            file=row["file"],
            line=row["line"],
            end_line=row["end_line"]
        )

    def get_node_id_by_name(self, name: str) -> Optional[int]:
        """Get a node ID by its name.

        Args:
            name: Node name to look up.

        Returns:
            Node ID if found, None otherwise.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM nodes WHERE name = ?", (name,))
        row = cur.fetchone()
        return row["id"] if row else None

    # -----------------------
    # EDGE STORAGE
    # -----------------------

    def insert_edges(self, edges: List[Edge]) -> None:
        """Insert edges, resolving source/target names to IDs.

        Args:
            edges: List of edges to insert.
        """
        cur = self.conn.cursor()

        for edge in edges:
            # If edge has names but no IDs, resolve them
            if edge.source is None and edge.source_name:
                edge.source = self._resolve_edge_target(edge.source_name, edge)

            if edge.target is None and edge.target_name:
                edge.target = self._resolve_edge_target(edge.target_name, edge)

            # Only insert if both ends resolved
            if edge.source is not None and edge.target is not None:
                cur.execute(
                    "INSERT INTO edges (source, target, type, source_name, target_name) VALUES (?, ?, ?, ?, ?)",
                    (edge.source, edge.target, edge.type, edge.source_name, edge.target_name)
                )

        self.conn.commit()

    def _resolve_edge_target(self, name: str, edge: Edge) -> Optional[int]:
        """Resolve a node name to ID, preferring nodes in the same file.

        Args:
            name: Node name to resolve.
            edge: Edge being resolved (used for context).

        Returns:
            Node ID if found, None otherwise.
        """
        # First, try to find the source node to get its file
        source_file = None
        if edge.source is not None:
            cur = self.conn.cursor()
            cur.execute("SELECT file FROM nodes WHERE id = ?", (edge.source,))
            row = cur.fetchone()
            if row:
                source_file = row[0]

        # Try to find a node with the same name in the same file
        if source_file:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id FROM nodes WHERE name = ? AND file = ? LIMIT 1",
                (name, source_file)
            )
            row = cur.fetchone()
            if row:
                return row[0]

        # Fallback to any node with that name
        return self.get_node_id_by_name(name)

    # -----------------------
    # CHUNK STORAGE
    # -----------------------

    def insert_chunks(self, chunks: List[Chunk]) -> None:
        """Insert chunks and assign IDs.

        Args:
            chunks: List of chunks to insert. Their id field will be updated.
        """
        cur = self.conn.cursor()

        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO chunks (file_path, start_line, end_line, content, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk.file_path, chunk.start_line, chunk.end_line, chunk.content, chunk.embedding)
            )
            chunk.id = cur.lastrowid

        self.conn.commit()

    def update_chunk_embedding(self, chunk_id: int, embedding: bytes) -> None:
        """Update embedding for an existing chunk.

        Args:
            chunk_id: ID of the chunk to update.
            embedding: Embedding bytes to store.
        """
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE chunks SET embedding = ? WHERE id = ?",
            (embedding, chunk_id)
        )
        self.conn.commit()

    # -----------------------
    # RETRIEVAL HELPERS
    # -----------------------

    def get_all_nodes(self) -> List[Node]:
        """Get all nodes from storage.

        Returns:
            List of all nodes.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM nodes")
        rows = cur.fetchall()

        return [
            Node(
                id=row["id"],
                type=row["type"],
                name=row["name"],
                file=row["file"],
                line=row["line"],
                end_line=row["end_line"]
            )
            for row in rows
        ]

    def get_all_chunks(self) -> List[Chunk]:
        """Get all chunks from storage.

        Returns:
            List of all chunks.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chunks")
        rows = cur.fetchall()

        return [
            Chunk(
                id=row["id"],
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                content=row["content"],
                embedding=row["embedding"]
            )
            for row in rows
        ]

    def get_chunks_with_embeddings(self) -> List[Chunk]:
        """Get only chunks that have embeddings.

        Returns:
            List of chunks with embeddings.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chunks WHERE embedding IS NOT NULL")
        rows = cur.fetchall()

        return [
            Chunk(
                id=row["id"],
                file_path=row["file_path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                content=row["content"],
                embedding=row["embedding"]
            )
            for row in rows
        ]

    def clear(self):
        """Clear all data from tables."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM nodes")
        cur.execute("DELETE FROM edges")
        cur.execute("DELETE FROM chunks")
        self.conn.commit()

    # -----------------------
    # GRAPH QUERY HELPERS
    # -----------------------

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get all node IDs connected to the given node.

        Args:
            node_id: ID of the node to get neighbors for.

        Returns:
            List of neighboring node IDs.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT target FROM edges WHERE source = ?
            UNION
            SELECT source FROM edges WHERE target = ?
        """, (node_id, node_id))

        return [row[0] for row in cur.fetchall()]

    def get_nodes_in_chunk(self, file_path: str, start_line: int, end_line: int) -> List[Node]:
        """Get all nodes that are within a chunk's line range.

        Args:
            file_path: Path to the file.
            start_line: Start line of the chunk.
            end_line: End line of the chunk.

        Returns:
            List of nodes within the chunk.
        """
        from pathlib import Path

        # Normalize: try both relative and absolute paths
        abs_path = str(Path(file_path).resolve())

        cur = self.conn.cursor()
        cur.execute("""
            SELECT * FROM nodes
            WHERE (file = ? OR file = ?)
            AND line BETWEEN ? AND ?
        """, (file_path, abs_path, start_line, end_line))

        rows = cur.fetchall()

        return [
            Node(
                id=row["id"],
                type=row["type"],
                name=row["name"],
                file=row["file"],
                line=row["line"],
                end_line=row["end_line"]
            )
            for row in rows
        ]

    def get_chunks_for_nodes(self, node_ids: List[int]) -> List[Chunk]:
        """Get chunks that contain the given nodes.

        Args:
            node_ids: List of node IDs to find chunks for.

        Returns:
            List of chunks containing those nodes.
        """
        if not node_ids:
            return []

        # Get all chunks and nodes, then match in Python
        # (needed because of path mismatch between chunks and nodes)
        chunks = self.get_all_chunks()

        # Get all nodes we're looking for
        cur = self.conn.cursor()
        placeholders = ','.join('?' * len(node_ids))
        cur.execute(f"SELECT * FROM nodes WHERE id IN ({placeholders})", node_ids)
        nodes = cur.fetchall()

        # Build a set of (file_path, line) for all target nodes
        from pathlib import Path
        target_locations = set()
        for node in nodes:
            # Nodes can have absolute paths, chunks have relative
            # Try both for matching
            target_locations.add((node["file"], node["line"]))
            # Also try to get relative path
            try:
                rel_path = str(Path(node["file"]).relative_to(Path.cwd()))
                target_locations.add((rel_path, node["line"]))
            except:
                pass

        # Find chunks containing these locations
        matching_chunks = []
        for chunk in chunks:
            for file_path, line in target_locations:
                if chunk.file_path == file_path:
                    if chunk.start_line <= line <= chunk.end_line:
                        matching_chunks.append(chunk)
                        break

        return matching_chunks

    def close(self):
        """Close the database connection."""
        self.conn.close()
