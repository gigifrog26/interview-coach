"""Interview session models for the Interview Coach System."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pydantic import Field, computed_field

from .base import BaseModel, TimestampedModel, IdentifiableModel
from .enums import CustomerTier, DifficultyLevel, SessionStatus
from .resume import ResumeData
from .job import JobDescription


class Question(BaseModel):
    """Represents an interview question."""

    question_id: str = Field(..., description="Unique question identifier")
    content: str = Field(..., description="Question content")
    topic: str = Field(..., description="Question topic")
    difficulty: DifficultyLevel = Field(..., description="Question difficulty level")
    context: Dict[str, Any] = Field(default_factory=dict, description="Question context")
    generated_by: str = Field(..., description="Source of question generation")

    expected_answer_points: List[str] = Field(default_factory=list, description="Expected answer points")
    time_limit_minutes: Optional[int] = Field(None, description="Time limit for answering")



    def get_difficulty_score(self) -> float:
        """Get numerical score for difficulty level."""
        return self.difficulty.score_multiplier


class Evaluation(BaseModel):
    """Represents evaluation of a candidate response."""

    question_id: str = Field(..., description="Question identifier")
    response: str = Field(..., description="Candidate's response")
    technical_score: int = Field(..., ge=0, le=10, description="Technical accuracy score")
    problem_solving_score: int = Field(..., ge=0, le=10, description="Problem-solving approach score")
    communication_score: int = Field(..., ge=0, le=10, description="Communication clarity score")
    code_quality_score: int = Field(..., ge=0, le=10, description="Code quality score")
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Overall performance score")
    feedback: str = Field(..., description="Detailed feedback")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    response_time_seconds: Optional[float] = Field(None, description="Time taken to respond")


    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "technical": 0.4,
            "problem_solving": 0.3,
            "communication": 0.2,
            "code_quality": 0.1,
        }
        calculated_score = (
            self.technical_score * weights["technical"]
            + self.problem_solving_score * weights["problem_solving"]
            + self.communication_score * weights["communication"]
            + self.code_quality_score * weights["code_quality"]
        )
        return round(calculated_score, 3)

    def get_performance_level(self) -> str:
        """Get performance level based on overall score."""
        if self.overall_score >= 0.8:
            return "Excellent"
        elif self.overall_score >= 0.6:
            return "Good"
        elif self.overall_score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"


class SkillAssessment(BaseModel):
    """Assessment of a specific skill."""

    skill_name: str = Field(..., description="Skill name")
    proficiency_level: str = Field(..., description="Proficiency level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in assessment")
    evidence_from_interview: List[str] = Field(default_factory=list, description="Evidence from interview")


class SkillGap(BaseModel):
    """Gap analysis for a skill."""

    skill_name: str = Field(..., description="Skill name")
    current_level: str = Field(..., description="Current proficiency level")
    required_level: str = Field(..., description="Required proficiency level")
    gap_size: float = Field(..., ge=0.0, le=1.0, description="Gap size")
    impact_on_role: str = Field(..., description="Impact on job role")
    learning_resources: List[str] = Field(default_factory=list, description="Learning resources")


class SkillAnalysis(BaseModel):
    """Analysis of candidate skills."""

    strong_skills: List[SkillAssessment] = Field(default_factory=list, description="Strong skills")
    weak_skills: List[SkillAssessment] = Field(default_factory=list, description="Weak skills")
    skill_gaps: List[SkillGap] = Field(default_factory=list, description="Skill gaps")
    learning_priorities: List[str] = Field(default_factory=list, description="Learning priorities")
    estimated_improvement_time: Dict[str, str] = Field(default_factory=dict, description="Estimated improvement time")


class ReportFormat(BaseModel):
    """Format configuration for reports."""

    format_type: str = Field(default="JSON", description="Report format type")
    include_charts: bool = Field(default=True, description="Include charts in report")
    include_code_samples: bool = Field(default=True, description="Include code samples")
    anonymize_data: bool = Field(default=False, description="Anonymize sensitive data")


class InterviewSessionReport(TimestampedModel):
    """Comprehensive interview session report."""

    report_id: str = Field(..., description="Unique report identifier")
    session_id: str = Field(..., description="Session identifier")
    customer_id: str = Field(..., description="Customer identifier")
    customer_tier: CustomerTier = Field(..., description="Customer tier")
    question_evaluations: List[Evaluation] = Field(default_factory=list, description="Question evaluations")
    improvement_recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="Report generation time")
    report_format: ReportFormat = Field(default_factory=ReportFormat, description="Report format configuration")

    # EvaluationSummary
    total_evaluations: int = Field(..., description="Total number of evaluations")
    average_scores: Optional[Dict[str, float]] = Field(None, description="Average scores for each category")
    top_improvement_areas: Optional[List[str]] = Field(default_factory=list, description="Top areas for improvement")
    score_distribution: Optional[Dict[str, int]] = Field(None, description="Distribution of scores")

    # InterviewSummary
    total_questions: int = Field(..., description="Total questions asked")
    questions_answered: int = Field(..., description="Questions answered")
    session_duration: timedelta = Field(..., description="Total session duration")
    session_status: SessionStatus = Field(..., description="Final session status")

    def get_performance_summary(self) -> str:
        """Get human-readable performance summary."""
        score = self.overall_scores.overall_performance
        if score >= 0.8:
            return "Excellent performance with strong technical skills"
        elif score >= 0.6:
            return "Good performance with room for improvement"
        elif score >= 0.4:
            return "Fair performance, significant improvement needed"
        else:
            return "Below expectations, extensive improvement required"

    def get_top_improvement_areas(self, limit: int = 3) -> List[str]:
        """Get top areas for improvement."""
        if not self.improvement_recommendations:
            return []
        return self.improvement_recommendations[:limit]


class InterviewSession(TimestampedModel):
    """Represents an interview session."""

    session_id: str = Field(..., description="Unique session identifier")
    customer_id: str = Field(..., description="Customer identifier")
    customer_tier: CustomerTier = Field(..., description="Customer subscription tier")
    resume_data: ResumeData = Field(..., description="Candidate resume data")
    job_description: JobDescription = Field(..., description="Job requirements")
    current_topic: str = Field(default="", description="Current interview topic")
    current_difficulty: DifficultyLevel = Field(default=DifficultyLevel.EASY, description="Current difficulty level")
    questions_asked: int = Field(default=0, description="Number of questions asked")
    max_questions: int = Field(..., description="Maximum questions allowed")
    start_time: datetime = Field(default_factory=datetime.now, description="Session start time")
    status: SessionStatus = Field(default=SessionStatus.CREATED, description="Session status")
    last_save_point: datetime = Field(default_factory=datetime.now, description="Last auto-save time")
    save_frequency: int = Field(default=3, description="Auto-save every N questions")
    recovery_data: Dict[str, Any] = Field(default_factory=dict, description="Session state for recovery")
    questions: List[Question] = Field(default_factory=list, description="Questions in this session")
    evaluations: List[Evaluation] = Field(default_factory=list, description="Question evaluations")
    notes: Optional[str] = Field(None, description="Session notes")
    final_report: Optional[InterviewSessionReport] = Field(None, description="Final report")

    def can_ask_more_questions(self) -> bool:
        """Check if more questions can be asked."""
        return self.questions_asked < self.max_questions

    def should_auto_save(self) -> bool:
        """Check if auto-save should be triggered."""
        return self.questions_asked % self.save_frequency == 0

    def session_duration(self) -> timedelta:
        """Calculate session duration."""
        end_time = self.updated_at or datetime.now()
        return end_time - self.start_time

    def completion_percentage(self) -> float:
        """Calculate session completion percentage."""
        return (self.questions_asked / self.max_questions) * 100

    def add_question(self, question: Question) -> None:
        """Add a question to the session."""
        self.questions.append(question)
        self.questions_asked += 1
        self.update_timestamp()

    def add_evaluation(self, evaluation: Evaluation) -> None:
        """Add an evaluation to the session."""
        self.evaluations.append(evaluation)
        self.update_timestamp()

    def get_average_score(self) -> float:
        """Calculate average score across all evaluations."""
        if not self.evaluations:
            return 0.0
        total_score = sum(eval.overall_score for eval in self.evaluations)
        return round(total_score / len(self.evaluations), 3)

    def get_topic_coverage(self) -> Dict[str, int]:
        """Get coverage of topics in the session."""
        topic_counts = {}
        for question in self.questions:
            topic_counts[question.topic] = topic_counts.get(question.topic, 0) + 1
        return topic_counts
