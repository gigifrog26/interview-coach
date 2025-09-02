"""Enumeration types for the Interview Coach System."""

from enum import Enum, auto
from typing import List


class CustomerTier(Enum):
    """Customer subscription tiers."""

    STANDARD = auto()
    MVP = auto()

    @property
    def max_questions(self) -> int:
        """Get maximum questions allowed for this tier."""
        return 3 if self == CustomerTier.STANDARD else 20
    
    @classmethod
    def _missing_(cls, value):
        """Handle string and integer values during deserialization."""
        if isinstance(value, str):
            # Handle cases like "CustomerTier.MVP" or "MVP"
            if value.startswith("CustomerTier."):
                value = value.split(".", 1)[1]
            try:
                return cls[value]
            except KeyError:
                pass
        elif isinstance(value, int):
            # Handle integer values (0=STANDARD, 1=MVP)
            enum_values = list(cls)
            if 0 <= value < len(enum_values):
                return enum_values[value]
        return None


class DifficultyLevel(Enum):
    """Question difficulty levels."""

    EASY = auto()
    MEDIUM = auto()
    HARD = auto()

    @property
    def score_multiplier(self) -> float:
        """Get numerical score for this difficulty level."""
        return {DifficultyLevel.EASY: 0.0, DifficultyLevel.MEDIUM: 1.0, DifficultyLevel.HARD: 2.0}[self]
    
    @classmethod
    def _missing_(cls, value):
        """Handle string and integer values during deserialization."""
        if isinstance(value, str):
            # Handle cases like "DifficultyLevel.EASY" or "EASY"
            if value.startswith("DifficultyLevel."):
                value = value.split(".", 1)[1]
            try:
                return cls[value]
            except KeyError:
                pass
        elif isinstance(value, int):
            # Handle integer values (0=EASY, 1=MEDIUM, 2=HARD)
            enum_values = list(cls)
            if 0 <= value < len(enum_values):
                return enum_values[value]
        return None


class SessionStatus(Enum):
    """Interview session status."""

    CREATED = auto()
    IN_PROGRESS = auto()
    PAUSED = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    
    @classmethod
    def _missing_(cls, value):
        """Handle string values during deserialization."""
        if isinstance(value, str):
            # Handle cases like "SessionStatus.COMPLETED" or "COMPLETED"
            if value.startswith("SessionStatus."):
                value = value.split(".", 1)[1]
            try:
                return cls[value]
            except KeyError:
                pass
        return None


class SkillLevel(Enum):
    """Skill proficiency levels."""

    BEGINNER = auto()
    INTERMEDIATE = auto()
    ADVANCED = auto()
    EXPERT = auto()

    @property
    def score_range(self) -> tuple[float, float]:
        """Get score range for this skill level."""
        ranges = {
            SkillLevel.BEGINNER: (0.0, 0.25),
            SkillLevel.INTERMEDIATE: (0.25, 0.5),
            SkillLevel.ADVANCED: (0.5, 0.75),
            SkillLevel.EXPERT: (0.75, 1.0),
        }
        return ranges[self]
    
    @classmethod
    def _missing_(cls, value):
        """Handle string and integer values during deserialization."""
        if isinstance(value, str):
            # Handle cases like "SkillLevel.ADVANCED" or "ADVANCED"
            if value.startswith("SkillLevel."):
                value = value.split(".", 1)[1]
            try:
                return cls[value]
            except KeyError:
                pass
        elif isinstance(value, int):
            # Handle integer values (0=BEGINNER, 1=INTERMEDIATE, 2=ADVANCED, 3=EXPERT)
            enum_values = list(cls)
            if 0 <= value < len(enum_values):
                return enum_values[value]
        return None


class SkillPriority(Enum):
    """Skill priority levels for job requirements."""

    REQUIRED = auto()
    PREFERRED = auto()
    NICE_TO_HAVE = auto()

    @property
    def weight(self) -> float:
        """Get weight for this priority level."""
        return {SkillPriority.REQUIRED: 1.0, SkillPriority.PREFERRED: 0.7, SkillPriority.NICE_TO_HAVE: 0.3}[self]
    
    @classmethod
    def _missing_(cls, value):
        """Handle string and integer values during deserialization."""
        if isinstance(value, str):
            # Handle cases like "SkillPriority.REQUIRED" or "REQUIRED"
            if value.startswith("SkillPriority."):
                value = value.split(".", 1)[1]
            try:
                return cls[value]
            except KeyError:
                pass
        elif isinstance(value, int):
            # Handle integer values (0=REQUIRED, 1=PREFERRED, 2=NICE_TO_HAVE)
            enum_values = list(cls)
            if 0 <= value < len(enum_values):
                return enum_values[value]
        return None
