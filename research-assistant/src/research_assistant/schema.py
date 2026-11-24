"""M1: the typed output schema and its strict JSON Schema twin."""

from __future__ import annotations

import copy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EntityType = Literal["person", "organisation", "place", "date", "other"]


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    type: EntityType


class Findings(BaseModel):
    """What the assistant must return for every article."""

    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(min_length=1)
    key_findings: list[str] = Field(min_length=1, max_length=5)
    entities: list[Entity] = Field(max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)


def strict_json_schema(model: type[BaseModel] = Findings) -> dict:
    """Pydantic's schema, tightened for strict structured output.

    Strict mode on most hosts wants every property listed in `required`,
    `additionalProperties: false` on every object, and no `default` or `title`
    noise. We walk the whole tree, including `$defs`, and fix each object.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def tighten(node: object) -> None:
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for item in node:
                tighten(item)

    tighten(schema)
    return {"name": "findings", "strict": True, "schema": schema}
