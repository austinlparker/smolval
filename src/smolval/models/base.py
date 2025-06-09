"""Base models and common types for smolval."""

from datetime import datetime

from pydantic import BaseModel, Field


class BaseMessage(BaseModel):
    """Base class for all message types."""

    type: str
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseContent(BaseModel):
    """Base class for message content."""

    type: str
