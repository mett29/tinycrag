"""Answer generation from retrieved chunks with Gemini via the Google GenAI SDK."""

from .models import ChunkClassification, RetrievalResult

DEFAULT_SYSTEM_INSTRUCTION = (
    "You answer questions using only the context passages provided, each labelled "
    "with a bracketed source:chunk_index tag. Cite the passages you rely on by "
    "copying that tag exactly, e.g. [report.pdf:12]. If the passages do not contain the "
    "answer, say so instead of guessing."
)


class GeminiAnswerer:
    """Answer a query from retrieved chunks using Gemini.

    By default the client talks to Gemini on Vertex AI (``enterprise=True``) using
    Application Default Credentials, with project and location taken from the
    arguments or the ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``
    environment variables. Pass ``enterprise=False`` to use the Gemini API with an
    API key instead. The client is created lazily on first use, so importing this
    module or constructing the class needs no credentials.
    """

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        *,
        enterprise: bool = True,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        temperature: float = 0.0,
    ) -> None:
        """Configure the backend and generation settings.

        Args:
            model: Gemini model name.
            enterprise: ``True`` (default) for Vertex AI, ``False`` for the Gemini API.
            api_key: Gemini API key. Used only when ``enterprise`` is false; if
                omitted, the SDK reads ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.
            project: Google Cloud project for Vertex AI. Ignored for the Gemini API.
            location: Vertex AI region. Ignored for the Gemini API.
            system_instruction: System prompt guiding grounded answering.
            temperature: Sampling temperature.

        Raises:
            ValueError: If Vertex AI is selected together with an API key, or the
                Gemini API is selected together with a project or location.
        """
        if enterprise and api_key is not None:
            raise ValueError("api_key cannot be used with enterprise=True (Vertex AI)")
        if not enterprise and (project is not None or location is not None):
            raise ValueError("project and location require enterprise=True")
        self.model = model
        self.enterprise = enterprise
        self.system_instruction = system_instruction
        self.temperature = temperature
        self._api_key = api_key
        self._project = project
        self._location = location
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            if self.enterprise:
                self._client = genai.Client(
                    enterprise=True, project=self._project, location=self._location
                )
            else:
                self._client = genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def build_prompt(query: str, matches: tuple[ChunkClassification, ...]) -> str:
        """Format the query and retrieved chunks into a single prompt.

        Args:
            query: The user's question.
            matches: Selected chunks, in the order they should be presented.

        Returns:
            Context passages tagged ``[source:chunk_index]`` with their page, then the question.
        """
        passages = "\n\n".join(
            f"[{item.chunk.source}:{item.chunk.chunk_index}] "
            f"(page {item.chunk.page})\n{item.chunk.text}"
            for item in matches
        )
        return f"Context:\n{passages}\n\nQuestion: {query}"

    def answer(self, result: RetrievalResult, *, max_chunks: int | None = None) -> str:
        """Answer the retrieval query using the selected chunks as context.

        Args:
            result: Retrieval report; its ``matches`` (ranked by relevance) are used.
            max_chunks: Optional cap on how many top-ranked chunks to send.

        Returns:
            The model's answer text. If nothing was retrieved, a fixed message is
            returned without calling the model.

        Raises:
            ValueError: If ``max_chunks`` is not a positive integer.
            RuntimeError: If the model returns no text.
            Exception: If the client or the API call fails.
        """
        if max_chunks is not None and max_chunks < 1:
            raise ValueError("max_chunks must be a positive integer")
        matches = result.matches[:max_chunks]
        if not matches:
            return "No relevant passages were retrieved, so I cannot answer."

        from google.genai import types

        response = self._get_client().models.generate_content(
            model=self.model,
            contents=self.build_prompt(result.query, matches),
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=self.temperature,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no text")
        return response.text
