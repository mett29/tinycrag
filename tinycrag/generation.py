"""Grounded answer generation from retrieved passages with OpenAI or Gemini."""

from collections.abc import Sequence

from .models import Passage

DEFAULT_SYSTEM_INSTRUCTION = (
    "You answer questions using only the context passages provided, each labelled "
    "with a bracketed source:id tag. Cite the passages you rely on by "
    "copying that tag exactly, e.g. [report.pdf:0007]. If the passages do not contain the "
    "answer, say so instead of guessing."
)


NO_CONTEXT_ANSWER = "No relevant passages were retrieved, so I cannot answer."


def build_prompt(query: str, passages: Sequence[Passage]) -> str:
    """Format the query and context passages into a single prompt.

    Args:
        query: The user's question.
        passages: Context in the order it should be presented.

    Returns:
        Passages tagged ``[source:id]`` with their location, then the question.
    """
    context = "\n\n".join(
        f"[{item.tag}] ({item.location})\n{item.text}" for item in passages
    )
    return f"Context:\n{context}\n\nQuestion: {query}"


class OpenAIAnswerer:
    """Answer a query from retrieved passages with an OpenAI model.

    Uses the Responses API with ``OPENAI_API_KEY`` from the environment unless an
    ``api_key`` is given. The client is created lazily on first use.
    """

    def __init__(
        self,
        model: str = "gpt-6-luna",
        *,
        api_key: str | None = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    ) -> None:
        """Configure the model and prompt.

        Args:
            model: OpenAI model name.
            api_key: OpenAI API key; defaults to ``OPENAI_API_KEY``.
            system_instruction: System prompt guiding grounded answering.
        """
        self.model = model
        self.system_instruction = system_instruction
        self._api_key = api_key
        self._client = None

    def answer(self, query: str, passages: Sequence[Passage]) -> str:
        """Answer a query from tagged passages, e.g. selected document tree nodes.

        Args:
            query: The user's question.
            passages: Context passages; the model cites them by their ``tag``.

        Returns:
            The model's answer text, or a fixed message without calling the model
            when ``passages`` is empty.

        Raises:
            RuntimeError: If the model returns no text.
            Exception: If the client or the API call fails.
        """
        if not passages:
            return NO_CONTEXT_ANSWER
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        response = self._client.responses.create(
            model=self.model,
            instructions=self.system_instruction,
            input=build_prompt(query, passages),
        )
        if not response.output_text:
            raise RuntimeError("OpenAI returned no text")
        return response.output_text


class GeminiAnswerer:
    """Answer a query from retrieved passages using Gemini.

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

    def answer(self, query: str, passages: Sequence[Passage]) -> str:
        """Answer a query from tagged passages, e.g. selected document tree nodes.

        Args:
            query: The user's question.
            passages: Context passages; the model cites them by their ``tag``.

        Returns:
            The model's answer text, or a fixed message without calling the model
            when ``passages`` is empty.

        Raises:
            RuntimeError: If the model returns no text.
            Exception: If the client or the API call fails.
        """
        if not passages:
            return NO_CONTEXT_ANSWER

        from google.genai import types

        response = self._get_client().models.generate_content(
            model=self.model,
            contents=build_prompt(query, passages),
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=self.temperature,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no text")
        return response.text
