"""Local PageIndex tree indexing and PageIndex's own agentic chat."""

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .tree import TreeNode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """One tool call the PageIndex chat agent made while answering.

    Attributes:
        name: Tool name, e.g. ``get_page_content``.
        arguments: Raw arguments as the agent sent them.
    """

    name: str
    arguments: str


@dataclass(frozen=True)
class PageIndexAnswer:
    """PageIndex chat's answer and the retrieval steps behind it.

    Attributes:
        answer: Final answer text.
        tool_calls: Tool calls in the order the agent made them.
        model: Chat model that produced the answer.
        seconds: Wall-clock time for the whole chat run.
    """

    answer: str
    tool_calls: tuple[ToolCall, ...]
    model: str
    seconds: float


class PageIndexDocuments:
    """Build and reuse PageIndex trees stored on local disk.

    Indexing uses PageIndex's local mode: the tree comes from the PDF layout, and
    ``index_model`` (via ``OPENAI_API_KEY``) writes node summaries and refines the
    tree. Indexed documents are stored under ``storage_path`` and reused by name,
    so each PDF is indexed once. The PageIndex client is created lazily.
    """

    def __init__(
        self,
        storage_path: str | Path = ".pageindex",
        *,
        index_model: str = "gpt-6-luna",
        chat_model: str = "gpt-6-luna",
    ) -> None:
        """Configure local storage and models.

        Args:
            storage_path: Directory holding indexed documents.
            index_model: LiteLLM model name for summaries and tree refinement.
                Only used when a PDF is first indexed.
            chat_model: LiteLLM model name for ``chat``.
        """
        self.storage_path = Path(storage_path)
        self.index_model = index_model
        self.chat_model = chat_model
        self._client = None

    @property
    def client(self) -> Any:
        """The underlying ``pageindex.PageIndexLocalClient``."""
        if self._client is None:
            from pageindex import PageIndexLocalClient

            self._client = PageIndexLocalClient(
                storage_path=str(self.storage_path),
                index_model=self.index_model,
                chat_model=self.chat_model,
            )
        return self._client

    def find(self, pdf: str | Path) -> str | None:
        """Return the doc ID of an already indexed PDF with this file name."""
        from pageindex import PageIndexAPIError

        try:
            return self.client.get_document_id(Path(pdf).name)
        except PageIndexAPIError:
            return None

    def index(self, pdf: str | Path) -> str:
        """Index a PDF unless a document with the same file name already exists.

        Args:
            pdf: Path to a PDF file.

        Returns:
            The PageIndex doc ID.

        Raises:
            pageindex.PageIndexAPIError: If indexing fails, e.g. no API key.
        """
        doc_id = self.find(pdf)
        if doc_id is None:
            logger.info("Indexing %s with PageIndex", pdf)
            doc_id = self.client.submit_document(str(pdf))["doc_id"]
        return doc_id

    def tree(self, doc_id: str) -> tuple[TreeNode, ...]:
        """Load a document's tree with summaries and each node's own text."""
        name = self.client.get_document(doc_id).get("name") or doc_id
        result = self.client.get_tree(doc_id, node_summary=True, include_text=True)
        return tuple(
            TreeNode.from_pageindex(node, name) for node in result["result"] or ()
        )

    def chat(self, query: str, doc_id: str) -> PageIndexAnswer:
        """Answer with PageIndex's standard agentic chat over the tree.

        Args:
            query: User question.
            doc_id: Document to search.

        Returns:
            The answer, the agent's tool calls, and the elapsed time.
        """
        started = perf_counter()
        stream = self.client.chat(query, doc_id=doc_id, stream=True)
        parts: list[str] = []
        calls: list[ToolCall] = []
        for event in stream.events:
            if event["type"] == "answer":
                parts.append(event["delta"])
            elif event["type"] == "tool_call":
                calls.append(ToolCall(event["name"], str(event.get("arguments", ""))))
        return PageIndexAnswer(
            answer="".join(parts),
            tool_calls=tuple(calls),
            model=str(self.client.chat_model),
            seconds=perf_counter() - started,
        )
