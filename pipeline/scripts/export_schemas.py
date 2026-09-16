#!/usr/bin/env python
"""Dump the JSON Schema of every artifact model into ``site/src/types/``.

The site's TypeScript types are generated from this file, so the schema in
``schemas.py`` stays the single source of truth.
"""

from __future__ import annotations

import json

from examprep import schemas
from examprep.config import REPO_ROOT

EXPORTED = [
    schemas.QuestionSet,
    schemas.Crosswalk,
    schemas.Course,
    schemas.Transcript,
    schemas.Chunk,
    schemas.Extract,
    schemas.Answer,
    schemas.DataIndex,
]

OUT_PATH = REPO_ROOT / "site" / "src" / "types" / "schemas.json"


def _require_every_property(definition: object) -> None:
    """Mark all properties required.

    Pydantic keeps a field with a default out of ``required``, but the pipeline
    serializes every field, including the ones it leaves empty or null. Saying so
    spares the site a pile of optional-chaining on data that is always there.
    """

    if not isinstance(definition, dict):
        return
    properties = definition.get("properties")
    if isinstance(properties, dict) and properties:
        definition["required"] = list(properties)


def main() -> None:
    definitions: dict[str, object] = {}
    properties: dict[str, object] = {}

    for model in EXPORTED:
        # Serialization mode: fields with defaults are always present in the
        # files the pipeline writes, so the site should see them as required.
        schema = model.model_json_schema(ref_template="#/$defs/{model}", mode="serialization")
        definitions.update(schema.pop("$defs", {}))
        definitions[model.__name__] = schema
        properties[model.__name__] = {"$ref": f"#/$defs/{model.__name__}"}

    for definition in definitions.values():
        _require_every_property(definition)

    bundle = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ExamprepData",
        "type": "object",
        "properties": properties,
        "$defs": definitions,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(EXPORTED)} models)")


if __name__ == "__main__":
    main()
