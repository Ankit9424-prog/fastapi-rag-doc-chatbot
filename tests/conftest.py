"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_text() -> str:
    """Return a sample text for testing chunking strategies."""
    return (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to natural intelligence displayed by animals including humans. "
        "AI research has been defined as the field of study of intelligent agents, "
        "which refers to any system that perceives its environment and takes actions "
        "that maximize its chance of achieving its goals.\n\n"
        "The term 'artificial intelligence' had previously been used to describe "
        "machines that mimic and display human cognitive skills. AI applications "
        "include advanced web search engines, recommendation systems, understanding "
        "human speech, self-driving cars, automated decision-making, and competing "
        "at the highest level in strategic game systems.\n\n"
        "As machines become increasingly capable, tasks considered to require "
        "intelligence are often removed from the definition of AI, a phenomenon "
        "known as the AI effect. For instance, optical character recognition is "
        "frequently excluded from things considered to be AI, having become a "
        "routine technology.\n\n"
        "Modern artificial intelligence techniques are pervasive and are too "
        "numerous to list here. Frequently, when a technique reaches mainstream "
        "use, it is no longer considered artificial intelligence; this phenomenon "
        "is described as the AI effect."
    )


@pytest.fixture
def short_text() -> str:
    """Return a short text for edge case testing."""
    return "Hello world. This is a test."


@pytest.fixture
def empty_text() -> str:
    """Return empty text for edge case testing."""
    return ""
