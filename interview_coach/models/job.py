"""Job description models for the Interview Coach System."""

from typing import List, Optional
from pydantic import Field

from .base import BaseModel, TimestampedModel
from .enums import SkillPriority


class JobDescription(TimestampedModel):
    """Job requirements and description."""

    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    required_skills: List[str] = Field(default_factory=list, description="Required technical skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred technical skills")
    experience_years: Optional[float] = Field(None, description="Required years of experience")
    responsibilities: List[str] = Field(default_factory=list, description="Job responsibilities")
    technical_stack: List[str] = Field(default_factory=list, description="Technical stack used")
    industry: Optional[str] = Field(None, description="Industry sector")
    employment_type: Optional[str] = Field(None, description="Employment type (Full-time, Contract, etc.)")
    salary_range: Optional[str] = Field(None, description="Salary range if available")
    benefits: List[str] = Field(default_factory=list, description="Benefits offered")
    remote_policy: Optional[str] = Field(None, description="Remote work policy")

    def get_skill_priority(self, skill: str) -> SkillPriority:
        """Get priority level for a specific skill."""
        skill_lower = skill.lower()
        
        if skill_lower in [s.lower() for s in self.required_skills]:
            return SkillPriority.REQUIRED
        elif skill_lower in [s.lower() for s in self.preferred_skills]:
            return SkillPriority.PREFERRED
        else:
            return SkillPriority.NICE_TO_HAVE

    def calculate_skill_match(self, candidate_skills: List[str]) -> float:
        """Calculate skill match percentage with candidate skills."""
        if not self.required_skills:
            return 0.0
        
        required_skills = set(skill.lower() for skill in self.required_skills)
        candidate_skills_set = set(skill.lower() for skill in candidate_skills)
        
        if not required_skills:
            return 0.0
        
        matched_skills = required_skills.intersection(candidate_skills_set)
        base_match = len(matched_skills) / len(required_skills)
        
        # Bonus for preferred skills
        preferred_skills = set(skill.lower() for skill in self.preferred_skills)
        preferred_matches = preferred_skills.intersection(candidate_skills_set)
        preferred_bonus = len(preferred_matches) * 0.1  # 10% bonus per preferred skill
        
        return min(1.0, base_match + preferred_bonus)

    def get_technical_focus_areas(self) -> List[str]:
        """Get technical focus areas for interview questions."""
        focus_areas = []
        
        # Combine required and preferred skills
        all_skills = self.required_skills + self.preferred_skills
        
        # Add technical stack items
        focus_areas.extend(self.technical_stack)
        
        # Add skills that are not in technical stack
        for skill in all_skills:
            if skill.lower() not in [tech.lower() for tech in self.technical_stack]:
                focus_areas.append(skill)
        
        return list(set(focus_areas))  # Remove duplicates

    def get_experience_requirements(self) -> dict:
        """Get experience requirements breakdown."""
        return {
            "total_years": self.experience_years,
            "senior_level": self.experience_years >= 5,
            "mid_level": 2 <= self.experience_years < 5,
            "junior_level": self.experience_years < 2,
        }
