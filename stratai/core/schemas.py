"""Pydantic models for every agent input and output. Later stages add their models here."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Source(BaseModel):
    url: AnyHttpUrl
    title: str
    retrieved_at: datetime = Field(default_factory=_utcnow)
    snippet: str = ""


class Claim(BaseModel):
    text: str
    sources: list[Source] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]
