"""Parse Python files into nodes and edges using AST."""

import ast
from pathlib import Path
from typing import List, Dict, Set

from .models import Node, Edge


class Parser:
    """Parse Python files to extract code structure (nodes and edges).

    Note: Node IDs are assigned by the storage layer, not the parser.
    The parser returns nodes with id=None, and the indexer/storage assigns persistent IDs.
    """

    def parse_file(self, file_path: str, content: str = None) -> tuple[List[Node], List[Edge]]:
        """Parse a Python file and extract nodes and edges.

        Args:
            file_path: Path to the Python file.
            content: Optional file content (if already read). If None, will read from file.

        Returns:
            Tuple of (nodes, edges) extracted from the file.
            Nodes have id=None; IDs are assigned later by storage.
            Edges have source/target=None; resolved later by storage.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            return [], []

        # Read content if not provided
        if content is None:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                return [], []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return [], []

        nodes: List[Node] = []
        edges: List[Edge] = []

        # Create file node first
        file_node = Node(
            id=None,  # Assigned by storage
            type="file",
            name=path.name,
            file=str(path),
            line=0
        )
        nodes.append(file_node)

        # First pass: extract all nodes
        for ast_node in tree.body:
            if isinstance(ast_node, ast.ClassDef):
                class_node = Node(
                    id=None,
                    type="class",
                    name=ast_node.name,
                    file=str(path),
                    line=ast_node.lineno,
                    end_line=ast_node.end_lineno if hasattr(ast_node, 'end_lineno') else 0
                )
                nodes.append(class_node)

                # Extract inheritance edges
                for base in ast_node.bases:
                    if isinstance(base, ast.Name):
                        edges.append(Edge(
                            source=None,
                            target=None,
                            type="INHERITS",
                            source_name=ast_node.name,
                            target_name=base.id
                        ))

                # Extract methods
                for item in ast_node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_node = Node(
                            id=None,
                            type="function",
                            name=f"{ast_node.name}.{item.name}",
                            file=str(path),
                            line=item.lineno,
                            end_line=item.end_lineno if hasattr(item, 'end_lineno') else 0
                        )
                        nodes.append(method_node)

                        # Class CONTAINS method
                        edges.append(Edge(
                            source=None,
                            target=None,
                            type="CONTAINS",
                            source_name=ast_node.name,
                            target_name=f"{ast_node.name}.{item.name}"
                        ))

            elif isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_node = Node(
                    id=None,
                    type="function",
                    name=ast_node.name,
                    file=str(path),
                    line=ast_node.lineno,
                    end_line=ast_node.end_lineno if hasattr(ast_node, 'end_lineno') else 0
                )
                nodes.append(func_node)

            elif isinstance(ast_node, (ast.Assign, ast.AnnAssign)):
                # Extract top-level assignments (only UPPER_CASE constants)
                targets = ast_node.targets if isinstance(ast_node, ast.Assign) else [ast_node.target]

                for target in targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        const_node = Node(
                            id=None,
                            type="constant",
                            name=target.id,
                            file=str(path),
                            line=ast_node.lineno
                        )
                        nodes.append(const_node)

            elif isinstance(ast_node, ast.Import):
                # Create IMPORTS edges
                for alias in ast_node.names:
                    edges.append(Edge(
                        source=None,
                        target=None,
                        type="IMPORTS",
                        source_name=path.name,
                        target_name=alias.name
                    ))

            elif isinstance(ast_node, ast.ImportFrom):
                # Create IMPORTS edges for from imports
                module_name = ast_node.module or "unknown"
                for alias in ast_node.names:
                    edges.append(Edge(
                        source=None,
                        target=None,
                        type="IMPORTS",
                        source_name=path.name,
                        target_name=f"{module_name}.{alias.name}"
                    ))

        # Add file CONTAINS edges
        for node in nodes:
            if node.type in ["class", "function", "constant"]:
                edges.append(Edge(
                    source=None,
                    target=None,
                    type="CONTAINS",
                    source_name=path.name,
                    target_name=node.name
                ))

        # Build a set of known constant names for reference tracking
        known_constants = {node.name for node in nodes if node.type == "constant"}
        known_functions = {node.name for node in nodes if node.type == "function"}

        # Second pass: extract function calls and variable references
        class CallVisitor(ast.NodeVisitor):
            """AST visitor to extract function calls and variable references."""

            def __init__(self, edges: List[Edge], known_constants: Set[str], known_functions: Set[str]):
                self.edges = edges
                self.current_function = None
                self.current_class = None
                self.known_constants = known_constants
                self.known_functions = known_functions

            def visit_ClassDef(self, node: ast.ClassDef):
                """Visit class definition."""
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node: ast.FunctionDef):
                """Visit function definition."""
                old_function = self.current_function
                full_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
                self.current_function = full_name
                self.generic_visit(node)
                self.current_function = old_function

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                """Visit async function definition."""
                self.visit_FunctionDef(node)

            def visit_Name(self, node: ast.Name):
                """Visit variable/constant reference."""
                if self.current_function:
                    # Track references to known constants
                    if node.id in self.known_constants:
                        edges.append(Edge(
                            source=None,
                            target=None,
                            type="REFERENCES",
                            source_name=self.current_function,
                            target_name=node.id
                        ))

                self.generic_visit(node)

            def visit_Call(self, node: ast.Call):
                """Visit function call."""
                if self.current_function:
                    # Try to extract the function name being called
                    call_name = None

                    if isinstance(node.func, ast.Name):
                        call_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        # For method calls like obj.method(), get the method name
                        call_name = node.func.attr

                    if call_name:
                        # Create CALLS edge (resolved later by storage)
                        edges.append(Edge(
                            source=None,
                            target=None,
                            type="CALLS",
                            source_name=self.current_function,
                            target_name=call_name
                        ))

                self.generic_visit(node)

        # Run call visitor
        visitor = CallVisitor(edges, known_constants, known_functions)
        visitor.visit(tree)

        return nodes, edges
