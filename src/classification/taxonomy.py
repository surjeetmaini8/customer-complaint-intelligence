"""
Centralized complaint taxonomy access.

Every module that needs category/subcategory strings should import from
here rather than hardcoding literals, so the taxonomy stays a single
source of truth (see config.yaml: paths.taxonomy_file).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List

from src.config import settings


@lru_cache(maxsize=1)
def load_taxonomy() -> Dict[str, List[str]]:
    path = settings.path("paths.taxonomy_file")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def categories() -> List[str]:
    return list(load_taxonomy().keys())


def subcategories(category: str) -> List[str]:
    return load_taxonomy().get(category, [])


def all_subcategories() -> List[str]:
    result = []
    for subs in load_taxonomy().values():
        result.extend(subs)
    return result


def is_valid_category(category: str) -> bool:
    return category in load_taxonomy()


def is_valid_subcategory(category: str, subcategory: str) -> bool:
    return subcategory in subcategories(category)
