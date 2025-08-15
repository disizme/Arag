
from __future__ import annotations
import re
from typing import List

_whitespace_re = re.compile(r"\s+")
_number_re = re.compile(r"\d+[\d,\.]*")

def normalize(text: str) -> str:
    text = text.strip()
    text = _whitespace_re.sub(" ", text)
    return text

def split_sentences(text: str) -> List[str]:
    text = normalize(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s]

def extract_numbers(text: str) -> List[str]:
    return _number_re.findall(text)
