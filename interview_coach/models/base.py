"""Base model classes for the Interview Coach System."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, List
from uuid import uuid4

from pydantic import BaseModel as PydanticBaseModel, Field


class BaseModel(PydanticBaseModel):
    """Base model with common functionality."""

    class Config:
        """Pydantic configuration."""
        use_enum_values = False
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        # Ensure enums are properly handled during serialization/deserialization
        validate_assignment = True


class IdentifiableModel(BaseModel):
    """Base model with ID field."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")


class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""

    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()


class ValidatableModel(BaseModel):
    """Base model with validation capabilities."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate the model data."""
        pass

    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        pass
