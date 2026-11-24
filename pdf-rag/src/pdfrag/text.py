"""Small text helpers shared by retrieval, reranking and judging: tokens, stems, sentences."""

from __future__ import annotations

import re

# Keeps model codes, part numbers and addresses whole: "h-200", "hs-112", "192.168.10.40", "3.0".
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
CITATION_RE = re.compile(r"\[([a-z0-9-]+) p\.(\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = frozenset(
    """a an and are as at be been before by can do does for from has have how i if in into is it its
    me my no not of on or our should so than that the their then there these this to too was we
    what when where which while who why will with would you your after all also any about each
    must use used using up out over per""".split()
)


def stem(word: str) -> str:
    """A tiny suffix stripper: enough to match "running" with "run", "damaged" with "damages", "prime" with "priming"."""
    if any(c.isdigit() for c in word) or len(word) <= 3:
        return word
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if suffix in ("ing", "ed") and len(word) > 3 and word[-1] == word[-2] and word[-1] not in "ls":
                word = word[:-1]  # runn -> run, stopp -> stop
            break
    else:
        if word.endswith("e") and len(word) > 4:
            word = word[:-1]  # prime -> prim, matching priming -> prim
    return word


def tokens(text: str) -> list[str]:
    """Lower-case word tokens, stop words kept."""
    return TOKEN_RE.findall(text.lower())


def terms(text: str) -> list[str]:
    """Content terms: tokens without stop words, stemmed. Used by BM25, the reranker and the judge."""
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    """Split on new lines first (table rows are one line each), then on sentence ends."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.extend(s.strip() for s in SENTENCE_RE.split(line) if s.strip())
    return out


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def citations(text: str) -> list[tuple[str, int]]:
    return [(doc, int(page)) for doc, page in CITATION_RE.findall(text)]
