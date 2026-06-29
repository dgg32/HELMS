import os

from .base import GraphBackend

_BACKENDS = ("ladybug", "neo4j")


def get_backend(
    name: str,
    db_path: str,
    nodes: dict,
    rels: list[dict],
    read_only: bool = False,
    setup: bool = True,
) -> GraphBackend:
    if name == "ladybug":
        from .ladybug_backend import LadybugBackend
        backend = LadybugBackend(db_path, read_only=read_only)
        if setup:
            backend.setup(nodes, rels)
        return backend
    if name == "neo4j":
        from .neo4j_backend import Neo4jBackend
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise SystemExit("NEO4J_PASSWORD not found in environment — required for neo4j backend.")
        uri = db_path or os.environ.get("NEO4J_URI")
        if not uri:
            raise SystemExit("No Neo4j URI: pass --db or set NEO4J_URI in .env")
        backend = Neo4jBackend(uri, username, password, read_only=read_only)
        if setup:
            backend.setup(nodes, rels)
        return backend
    raise ValueError(f"Unknown backend: {name!r}. Available: {', '.join(_BACKENDS)}")
