"""Typed data objects shared across tinycrag."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """A piece of context sent to an answering model.

    Attributes:
        tag: Citation tag the model should use, e.g. ``report.pdf:0007``.
        location: Human-readable location, e.g. ``Training, pages 4-6``.
        text: The passage content.
    """

    tag: str
    location: str
    text: str
