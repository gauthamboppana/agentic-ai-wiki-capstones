"""Structured research assistant: article in, validated JSON out."""

from .pipeline import ResearchError, ResearchResult, analyse
from .schema import Entity, Findings, strict_json_schema

__all__ = ["Entity", "Findings", "ResearchError", "ResearchResult", "analyse", "strict_json_schema"]
