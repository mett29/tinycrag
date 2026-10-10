"""Top-down Julia-1 navigation of a PageIndex document tree."""

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .models import Passage
from .retrieval import JuliaRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TreeNode:
    """One section of a PageIndex tree.

    Attributes:
        doc_name: Name of the indexed PDF, used in citation tags.
        node_id: PageIndex node identifier, unique within the document.
        title: Section title.
        start_page: One-based first page covered by the section and its children.
        end_page: One-based last page covered by the section and its children.
        summary: PageIndex summary of the whole subtree, or an empty string.
        text: The section's own text, excluding text held by its children.
        children: Subsections in document order.
    """

    doc_name: str
    node_id: str
    title: str
    start_page: int
    end_page: int
    summary: str
    text: str
    children: tuple[TreeNode, ...] = ()

    @classmethod
    def from_pageindex(cls, node: Mapping[str, Any], doc_name: str) -> TreeNode:
        """Convert one node of ``PageIndexClient.get_tree`` output, recursively."""
        return cls(
            doc_name=doc_name,
            node_id=str(node.get("node_id") or ""),
            title=str(node.get("title") or "").strip(),
            start_page=int(node.get("start_index") or 0),
            end_page=int(node.get("end_index") or node.get("start_index") or 0),
            summary=str(node.get("summary") or "").strip(),
            text=str(node.get("text") or "").strip(),
            children=tuple(
                cls.from_pageindex(child, doc_name) for child in node.get("nodes") or ()
            ),
        )

    @property
    def tag(self) -> str:
        """Citation tag in ``doc_name:node_id`` form."""
        return f"{self.doc_name}:{self.node_id}"

    @property
    def pages(self) -> str:
        """Page range as ``page N`` or ``pages N-M``."""
        if self.start_page == self.end_page:
            return f"page {self.start_page}"
        return f"pages {self.start_page}-{self.end_page}"

    def walk(self) -> Iterable[TreeNode]:
        """Yield this node and all descendants in depth-first document order."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class NodeDecision:
    """Julia's decision for one visited tree node.

    Attributes:
        node: The visited node.
        depth: Zero-based depth in the tree.
        choice: Winning model label, either ``relevant`` or ``irrelevant``.
        relevance_score: Softmax probability assigned to ``relevant``.
        selected: Whether the score meets the threshold. Selected nodes are
            expanded, and their own text becomes answer context.
    """

    node: TreeNode
    depth: int
    choice: str
    relevance_score: float
    selected: bool


@dataclass(frozen=True)
class TreeRetrievalResult:
    """Outcome of navigating one or more document trees for a query.

    Attributes:
        query: User query used for every decision.
        threshold: Inclusive minimum relevance score for selecting a node.
        roots: Top-level nodes of the navigated trees.
        decisions: Every visited node in visit order (breadth-first by level).
            Children of rejected nodes are never visited.
        seconds: Wall-clock time spent navigating, including model loading.
    """

    query: str
    threshold: float
    roots: tuple[TreeNode, ...]
    decisions: tuple[NodeDecision, ...]
    seconds: float

    @property
    def selected(self) -> tuple[NodeDecision, ...]:
        """Selected nodes with their own text, in document order."""
        order = {id(node): index for index, node in enumerate(self._all_nodes())}
        return tuple(
            sorted(
                (d for d in self.decisions if d.selected and d.node.text),
                key=lambda d: order[id(d.node)],
            )
        )

    def passages(self) -> list[Passage]:
        """Return the selected nodes' own text as answer context."""
        return [
            Passage(d.node.tag, f"{d.node.title}, {d.node.pages}", d.node.text)
            for d in self.selected
        ]

    def _all_nodes(self) -> Iterable[TreeNode]:
        for root in self.roots:
            yield from root.walk()


class JuliaTreeNavigator:
    """Navigate a document tree top-down, letting Julia-1 decide where to descend.

    Each level is scored independently: a node is shown to Julia as its title,
    page range, and PageIndex summary (which describes its whole subtree). Selected
    nodes are expanded into their children; rejected nodes prune their subtree.
    The own text of every selected node becomes context for the answer, so a
    relevant parent contributes its introduction while its children are judged on
    their own.
    """

    def __init__(
        self, retriever: JuliaRetriever | None = None, *, fallback_words: int = 400
    ) -> None:
        """Configure the scorer.

        Args:
            retriever: Julia scorer. Defaults to the local ``Julia-1`` folder on CPU.
            fallback_words: Words of a node's own text shown to Julia when the node
                has no summary.
        """
        self.retriever = retriever if retriever is not None else JuliaRetriever()
        self.fallback_words = fallback_words

    def describe(self, node: TreeNode) -> str:
        """Return the text Julia judges for a node."""
        body = node.summary or " ".join(node.text.split()[: self.fallback_words])
        return f"Title: {node.title}\nLocation: {node.pages}\nSummary: {body}"

    def navigate(
        self, roots: Sequence[TreeNode], query: str, *, threshold: float = 0.5
    ) -> TreeRetrievalResult:
        """Score nodes level by level, descending only into selected nodes.

        Args:
            roots: Top-level nodes of one or more document trees.
            query: Nonempty user question.
            threshold: Inclusive minimum relevance probability for selection.

        Returns:
            Every decision taken and the selected context.

        Raises:
            ValueError: If the query or threshold is invalid, or loading fails.
            RuntimeError: If a node cannot be scored.
        """
        self.retriever.validate_request(query, threshold)
        started = perf_counter()
        decisions: list[NodeDecision] = []
        frontier = [(node, 0) for node in roots]
        while frontier:
            next_frontier = []
            for node, depth in frontier:
                try:
                    choice, score = self.retriever.score(
                        query, self.describe(node), kind="section"
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Classification failed for {node.tag} ({node.title}): {exc}"
                    ) from exc
                selected = score >= threshold
                decisions.append(NodeDecision(node, depth, choice, score, selected))
                if selected:
                    next_frontier.extend((child, depth + 1) for child in node.children)
            logger.info(
                "Scored %s nodes at depth %s; descending into %s",
                len(frontier),
                frontier[0][1],
                len(next_frontier),
            )
            frontier = next_frontier
        return TreeRetrievalResult(
            query=query,
            threshold=threshold,
            roots=tuple(roots),
            decisions=tuple(decisions),
            seconds=perf_counter() - started,
        )
