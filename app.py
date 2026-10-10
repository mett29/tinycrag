"""Streamlit UI comparing PageIndex chat with Julia-1 tree navigation.

Run with ``uv run --group ui streamlit run app.py``.
"""

from pathlib import Path
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv

from tinycrag import (
    JuliaRetriever,
    JuliaTreeNavigator,
    OpenAIAnswerer,
    PageIndexDocuments,
    TreeNode,
    TreeRetrievalResult,
)

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="tinycrag: PageIndex vs Julia-1", layout="wide")


@st.cache_resource
def get_navigator() -> JuliaTreeNavigator:
    """Keep one Julia-1 engine resident across reruns."""
    return JuliaTreeNavigator(JuliaRetriever(ROOT / "Julia-1"))


@st.cache_resource
def get_documents(model: str) -> PageIndexDocuments:
    """Keep one PageIndex client per model."""
    return PageIndexDocuments(ROOT / ".pageindex", index_model=model, chat_model=model)


@st.cache_data(show_spinner=False)
def load_tree(_documents: PageIndexDocuments, doc_id: str) -> tuple[TreeNode, ...]:
    """Load a stored tree once per document."""
    return _documents.tree(doc_id)


def render_tree(roots: tuple[TreeNode, ...], result: TreeRetrievalResult | None) -> str:
    """Render the tree as Markdown, marking Julia's decisions when available."""
    decisions = {d.node.tag: d for d in result.decisions} if result else {}
    lines = []

    def visit(node: TreeNode, depth: int) -> None:
        decision = decisions.get(node.tag)
        if result is None:
            mark = ""
        elif decision is None:
            mark = "⚪ "
        elif decision.selected:
            mark = f"🟢 `{decision.relevance_score:.2f}` "
        else:
            mark = f"🔴 `{decision.relevance_score:.2f}` "
        lines.append(
            f"{'    ' * depth}- {mark}**{node.title or '(untitled)'}** "
            f"<small>{node.node_id} · {node.pages}</small>"
        )
        for child in node.children:
            visit(child, depth + 1)

    for root in roots:
        visit(root, 0)
    return "\n".join(lines)


with st.sidebar:
    st.header("Document")
    upload = st.file_uploader("Add a PDF to sources/", type="pdf")
    if upload is not None:
        SOURCES.mkdir(exist_ok=True)
        (SOURCES / upload.name).write_bytes(upload.getvalue())
    pdfs = sorted(p.name for p in SOURCES.glob("*") if p.suffix.lower() == ".pdf")
    pdf_name = st.selectbox("PDF", pdfs, index=None, placeholder="Pick a PDF")

    st.header("Settings")
    model = st.text_input(
        "OpenAI model",
        "gpt-6-luna",
        help="Builds new trees, runs PageIndex chat, and writes the Julia answer.",
    ).strip()
    threshold = st.slider("Julia relevance threshold", 0.0, 1.0, 0.5, 0.05)

st.title("PageIndex chat vs Julia-1 tree navigation")

if pdf_name is None:
    st.info("Pick or upload a PDF in the sidebar.")
    st.stop()

documents = get_documents(model)
doc_id = documents.find(SOURCES / pdf_name)
if doc_id is None:
    st.warning(
        f"**{pdf_name}** is not indexed yet. Indexing builds the PageIndex tree "
        "once and stores it in `.pageindex/`; it needs `OPENAI_API_KEY`."
    )
    if not st.button("Build tree", type="primary"):
        st.stop()
    with st.spinner("Building the PageIndex tree…"):
        doc_id = documents.index(SOURCES / pdf_name)

roots = load_tree(documents, doc_id)
query = st.text_input("Question", placeholder="Ask something about the document")
run = st.button("Run both", type="primary", disabled=not query.strip())

if run:
    st.session_state.pop("results", None)
    pageindex_col, julia_col = st.columns(2)
    with pageindex_col, st.spinner("PageIndex chat is searching the tree…"):
        try:
            pageindex_result = documents.chat(query, doc_id)
        except Exception as exc:
            pageindex_result = exc
    with julia_col, st.spinner("Julia-1 is navigating the tree…"):
        try:
            navigation = get_navigator().navigate(roots, query, threshold=threshold)
            started = perf_counter()
            julia_answer = OpenAIAnswerer(model).answer(query, navigation.passages())
            julia_result = (navigation, julia_answer, perf_counter() - started)
        except Exception as exc:
            julia_result = exc
    st.session_state["results"] = (query, doc_id, pageindex_result, julia_result)

results = st.session_state.get("results")
if results is None or results[:2] != (query, doc_id):
    with st.expander("Document tree", expanded=True):
        st.markdown(render_tree(roots, None), unsafe_allow_html=True)
    st.stop()

_, _, pageindex_result, julia_result = results
pageindex_col, julia_col = st.columns(2)

with pageindex_col:
    st.subheader("PageIndex chat")
    if isinstance(pageindex_result, Exception):
        st.error(f"{type(pageindex_result).__name__}: {pageindex_result}")
    else:
        st.caption(
            f"{pageindex_result.model} · {pageindex_result.seconds:.1f}s · "
            f"{len(pageindex_result.tool_calls)} tool calls"
        )
        st.markdown(pageindex_result.answer)
        with st.expander("Tool calls"):
            for call in pageindex_result.tool_calls:
                st.code(f"{call.name}({call.arguments})", language=None)

with julia_col:
    st.subheader("Julia-1 tree navigation")
    if isinstance(julia_result, Exception):
        st.error(f"{type(julia_result).__name__}: {julia_result}")
    else:
        navigation, julia_answer, answer_seconds = julia_result
        total = sum(1 for root in roots for _ in root.walk())
        st.caption(
            f"{model} · {navigation.seconds + answer_seconds:.1f}s "
            f"(navigation {navigation.seconds:.1f}s, answer {answer_seconds:.1f}s) · "
            f"{len(navigation.decisions)}/{total} nodes scored · "
            f"{len(navigation.selected)} sections sent"
        )
        st.markdown(julia_answer)
        with st.expander("Navigation (🟢 selected · 🔴 pruned · ⚪ not visited)"):
            st.markdown(render_tree(roots, navigation), unsafe_allow_html=True)
        with st.expander("Context sent to the model"):
            for passage in navigation.passages():
                st.markdown(f"**[{passage.tag}]** {passage.location}")
                st.text(passage.text)
