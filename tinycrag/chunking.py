"""Simple text chunking without model or parser dependencies."""


class WordChunker:
    """Split text into non-overlapping groups of whitespace-separated words.

    Chunking normalizes whitespace and may split sentences or tables. Calling
    ``split`` separately for each page prevents chunks from crossing page boundaries.
    """

    def __init__(self, chunk_size: int = 300) -> None:
        """Configure the maximum number of words per chunk.

        Args:
            chunk_size: Positive integer word limit. The final chunk may be shorter.

        Raises:
            ValueError: If the limit is not a positive integer.
        """
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size

    def split(self, text: str) -> list[str]:
        """Return normalized chunks while retaining every word exactly once.

        Args:
            text: Plain text extracted from a page or supplied by the caller.

        Returns:
            Chunks in input order. Empty or whitespace-only text returns an empty list.
        """
        words = text.split()
        return [
            " ".join(words[index : index + self.chunk_size])
            for index in range(0, len(words), self.chunk_size)
        ]
