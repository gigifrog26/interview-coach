"""Resume data models for the Interview Coach System."""

from datetime import date
from typing import List, Optional
from pydantic import Field

from .base import BaseModel, TimestampedModel


class WorkExperience(BaseModel):
    """Work experience information."""

    company: str = Field(..., description="Company name")
    position: str = Field(..., description="Job position/title")
    start_date: date = Field(..., description="Start date")
    end_date: Optional[date] = Field(None, description="End date (None if current)")
    description: str = Field(..., description="Job description")
    skills_used: List[str] = Field(default_factory=list, description="Skills used in this role")
    achievements: List[str] = Field(default_factory=list, description="Key achievements")

    @property
    def duration_months(self) -> int:
        """Calculate duration in months."""
        end = self.end_date or date.today()
        return (end.year - self.start_date.year) * 12 + (end.month - self.start_date.month)


class Education(BaseModel):
    """Education information."""

    institution: str = Field(..., description="Educational institution")
    degree: str = Field(..., description="Degree obtained")
    field_of_study: str = Field(..., description="Field of study")
    start_date: date = Field(..., description="Start date")
    end_date: Optional[date] = Field(None, description="End date")
    gpa: Optional[float] = Field(None, description="GPA if available")
    relevant_courses: List[str] = Field(default_factory=list, description="Relevant courses taken")


class ResumeData(TimestampedModel):
    """Resume data for a candidate."""

    candidate_name: Optional[str] = Field(None, description="Candidate's full name")
    email: Optional[str] = Field(None, description="Candidate's email address")
    phone: Optional[str] = Field(None, description="Candidate's phone number")
    experience_years: Optional[float] = Field(None, description="Total years of experience")
    skills: List[str] = Field(default_factory=list, description="Technical skills")
    work_history: List[WorkExperience] = Field(default_factory=list, description="Work experience")
    education: List[Education] = Field(default_factory=list, description="Education history")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")
    summary: Optional[str] = Field(None, description="Professional summary")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")

    def get_skill_level(self, skill: str) -> str:
        """Get proficiency level for a specific skill based on experience."""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated logic
        if skill.lower() in [s.lower() for s in self.skills]:
            if self.experience_years >= 5:
                return "Expert"
            elif self.experience_years >= 3:
                return "Advanced"
            elif self.experience_years >= 1:
                return "Intermediate"
            else:
                return "Beginner"
        return "Not Listed"

    def get_relevant_experience(self, job_requirements: "JobDescription") -> List[WorkExperience]:
        """Get work experience relevant to job requirements."""
        relevant_experience = []
        required_skills = [skill.lower() for skill in job_requirements.required_skills]
        
        for experience in self.work_history:
            experience_skills = [skill.lower() for skill in experience.skills_used]
            if any(skill in experience_skills for skill in required_skills):
                relevant_experience.append(experience)
        
        return relevant_experience

    def calculate_skill_match(self, job_requirements: "JobDescription") -> float:
        """Calculate skill match percentage with job requirements."""
        if not job_requirements.required_skills:
            return 0.0
        
        required_skills = set(skill.lower() for skill in job_requirements.required_skills)
        candidate_skills = set(skill.lower() for skill in self.skills)
        
        if not required_skills:
            return 0.0
        
        matched_skills = required_skills.intersection(candidate_skills)
        return len(matched_skills) / len(required_skills)
