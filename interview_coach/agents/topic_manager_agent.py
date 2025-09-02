"""Topic Manager Agent for managing interview topics and progression."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..models.interview import InterviewSession, Question, Evaluation
from ..models.resume import ResumeData
from ..models.job import JobDescription
from ..models.enums import DifficultyLevel, SkillLevel
from ..agents.base_agent import BaseAgent
from ..services.llm_manager import LLMProviderManager
from ..utils.exceptions import AgentError, QuestionGenerationError


@dataclass
class TopicTransition:
    """Represents a topic transition recommendation."""
    
    from_topic: str
    to_topic: str
    reason: str
    confidence: float
    suggested_difficulty: DifficultyLevel
    estimated_duration: int  # minutes


@dataclass
class TopicCoverage:
    """Represents coverage statistics for a specific topic."""
    
    topic: str
    questions_asked: int
    average_score: float
    difficulty_progression: List[DifficultyLevel]
    time_spent: int  # minutes
    skill_gaps: List[str]
    strengths: List[str]
    coverage_percentage: float


class TopicManagerAgent(BaseAgent):
    """Manages interview topics, tracks coverage, and suggests transitions."""
    
    def __init__(self, llm_manager: LLMProviderManager):
        """Initialize the TopicManagerAgent.
        
        Args:
            llm_manager: Manager for LLM provider interactions.
        """
        super().__init__("TopicManagerAgent")
        self.llm_manager = llm_manager
        self._topic_hierarchy: Dict[str, List[str]] = {}
        self._skill_topic_mapping: Dict[str, List[str]] = {}
        self._topic_difficulty_progression: Dict[str, List[DifficultyLevel]] = {}
        
    def _initialize_resources(self) -> None:
        """Initialize agent-specific resources."""
        # Initialize LLM manager if needed
        self.llm_manager.initialize_providers()
        self.logger.info("TopicManagerAgent resources initialized")
        
    async def _cleanup_resources(self) -> None:
        """Cleanup agent-specific resources."""
        # Cleanup any agent-specific resources
        self.logger.info("TopicManagerAgent resources cleaned up")
        
    def initialize(self) -> None:
        """Initialize the agent and load topic configurations."""
        super().initialize()
        
        # Load topic configurations
        self._load_topic_configurations()
        
        self.log_operation("TopicManagerAgent initialized successfully")
        
    def _load_topic_configurations(self) -> None:
        """Load topic hierarchy and skill mappings."""
        # Define core technical topics
        self._topic_hierarchy = {
            "fundamentals": ["data_structures", "algorithms", "programming_concepts"],
            "data_structures": ["arrays", "linked_lists", "stacks", "queues", "trees", "graphs", "hash_tables"],
            "algorithms": ["sorting", "searching", "recursion", "dynamic_programming", "greedy", "backtracking"],
            "programming_concepts": ["oop", "functional_programming", "design_patterns", "clean_code"],
            "system_design": ["scalability", "distributed_systems", "databases", "caching", "load_balancing"],
            "backend": ["apis", "authentication", "databases", "caching", "message_queues"],
            "frontend": ["javascript", "react", "css", "accessibility", "performance"],
            "devops": ["ci_cd", "containerization", "monitoring", "logging", "security"],
            "testing": ["unit_testing", "integration_testing", "test_driven_development", "mocking"],
            "databases": ["sql", "nosql", "database_design", "query_optimization", "transactions"]
        }
        
        # Map skills to relevant topics
        self._skill_topic_mapping = {
            "python": ["programming_concepts", "data_structures", "algorithms"],
            "java": ["programming_concepts", "data_structures", "algorithms", "oop"],
            "javascript": ["programming_concepts", "frontend", "backend"],
            "react": ["frontend", "programming_concepts"],
            "node.js": ["backend", "programming_concepts"],
            "sql": ["databases", "data_structures"],
            "docker": ["devops", "containerization"],
            "kubernetes": ["devops", "containerization", "distributed_systems"],
            "aws": ["devops", "system_design", "distributed_systems"],
            "machine_learning": ["algorithms", "data_structures", "statistics"]
        }
        
        # Define difficulty progression for each topic
        for topic in self._get_all_topics():
            self._topic_difficulty_progression[topic] = [
                DifficultyLevel.EASY,
                DifficultyLevel.MEDIUM,
                DifficultyLevel.HARD
            ]
    
    def _get_all_topics(self) -> Set[str]:
        """Get all available topics."""
        topics = set()
        for main_topic, sub_topics in self._topic_hierarchy.items():
            topics.add(main_topic)
            topics.update(sub_topics)
        return topics
    
    def process(self, session: InterviewSession, **kwargs) -> Dict[str, any]:
        """Process the interview session and provide topic management insights.
        
        Args:
            session: The current interview session.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dictionary containing topic management insights and recommendations.
        """
        try:
            # Analyze current topic coverage
            topic_coverage = self._analyze_topic_coverage(session)
            
            # Check if topic transition is needed
            transition_needed = self._should_transition_topic(session, topic_coverage)
            if transition_needed:
                # Get next topic recommendation
                next_topic = self._get_next_topic_recommendation(session, topic_coverage)
            else:
                next_topic = session.current_topic
            
            next_topic_difficulty = DifficultyLevel.EASY
            if session.questions_asked > 0 and next_topic == session.current_topic:
                next_topic_difficulty = DifficultyLevel.MEDIUM if session.current_difficulty == DifficultyLevel.EASY else DifficultyLevel.HARD
            
            # Generate topic insights
            insights = self._generate_topic_insights(session, topic_coverage)
            
            return {
                "topic_coverage": topic_coverage,
                "next_topic": next_topic,
                "transition_needed": transition_needed,
                "difficulty": next_topic_difficulty,
                "insights": insights,
                "session_progress": self._calculate_session_progress(session)
            }
            
        except Exception as e:
            self.log_error(f"Error processing topic management: {str(e)}")
            raise AgentError(f"Topic management processing failed: {str(e)}")
    
    def _analyze_topic_coverage(self, session: InterviewSession) -> Dict[str, TopicCoverage]:
        """Analyze coverage statistics for all topics in the session.
        
        Args:
            session: The interview session to analyze.
            
        Returns:
            Dictionary mapping topics to their coverage statistics.
        """
        topic_coverage = {}
        
        for topic in self._get_all_topics():
            topic_questions = [q for q in session.questions if q.topic == topic]
            topic_evaluations = [e for e in session.evaluations 
                               if any(q.topic == topic for q in session.questions if q.question_id == e.question_id)]
            
            if not topic_questions:
                continue
                
            # Calculate average score
            scores = [e.calculate_overall_score() for e in topic_evaluations if e.calculate_overall_score() is not None]
            average_score = sum(scores) / len(scores) if scores else 0.0
            
            # Get difficulty progression
            difficulty_progression = [q.difficulty for q in topic_questions]
            
            # Calculate time spent
            time_spent = sum(e.response_time for e in topic_evaluations if e.response_time)
            
            # Identify skill gaps and strengths
            skill_gaps = self._identify_skill_gaps(topic_evaluations, topic)
            strengths = self._identify_strengths(topic_evaluations, topic)
            
            # Calculate coverage percentage
            coverage_percentage = self._calculate_topic_coverage_percentage(
                session, topic, topic_questions
            )
            
            topic_coverage[topic] = TopicCoverage(
                topic=topic,
                questions_asked=len(topic_questions),
                average_score=average_score,
                difficulty_progression=difficulty_progression,
                time_spent=time_spent,
                skill_gaps=skill_gaps,
                strengths=strengths,
                coverage_percentage=coverage_percentage
            )
        
        return topic_coverage
    
    def _identify_skill_gaps(self, evaluations: List[Evaluation], topic: str) -> List[str]:
        """Identify skill gaps based on evaluation scores.
        
        Args:
            evaluations: List of evaluations for the topic.
            topic: The topic being analyzed.
            
        Returns:
            List of identified skill gaps.
        """
        gaps = []
        
        for evaluation in evaluations:
            if evaluation.technical_score and evaluation.technical_score < 0.6:
                gaps.append("technical_knowledge")
            if evaluation.problem_solving_score and evaluation.problem_solving_score < 0.6:
                gaps.append("problem_solving")
            if evaluation.communication_score and evaluation.communication_score < 0.6:
                gaps.append("communication")
            if evaluation.code_quality_score and evaluation.code_quality_score < 0.6:
                gaps.append("code_quality")
        
        return list(set(gaps))
    
    def _identify_strengths(self, evaluations: List[Evaluation], topic: str) -> List[str]:
        """Identify strengths based on evaluation scores.
        
        Args:
            evaluations: List of evaluations for the topic.
            topic: The topic being analyzed.
            
        Returns:
            List of identified strengths.
        """
        strengths = []
        
        for evaluation in evaluations:
            if evaluation.technical_score and evaluation.technical_score > 0.8:
                strengths.append("technical_knowledge")
            if evaluation.problem_solving_score and evaluation.problem_solving_score > 0.8:
                strengths.append("problem_solving")
            if evaluation.communication_score and evaluation.communication_score > 0.8:
                strengths.append("communication")
            if evaluation.code_quality_score and evaluation.code_quality_score > 0.8:
                strengths.append("code_quality")
        
        return list(set(strengths))
    
    def _calculate_topic_coverage_percentage(self, session: InterviewSession, topic: str, 
                                          topic_questions: List[Question]) -> float:
        """Calculate the coverage percentage for a specific topic.
        
        Args:
            session: The interview session.
            topic: The topic to calculate coverage for.
            topic_questions: Questions asked for this topic.
            
        Returns:
            Coverage percentage (0.0 to 1.0).
        """
        if not session.job_description:
            return 0.0
            
        # Get required skills for the topic from job description
        required_skills = self._get_required_skills_for_topic(session.job_description, topic)
        
        if not required_skills:
            return 0.0
            
        # Count covered skills
        covered_skills = 0
        for skill in required_skills:
            if any(q.context and skill.lower() in q.context.lower() for q in topic_questions):
                covered_skills += 1
        
        return covered_skills / len(required_skills) if required_skills else 0.0
    
    def _get_required_skills_for_topic(self, job_description: JobDescription, topic: str) -> List[str]:
        """Get required skills for a specific topic from job description.
        
        Args:
            job_description: The job description.
            topic: The topic to get skills for.
            
        Returns:
            List of required skills for the topic.
        """
        all_skills = []
        all_skills.extend(job_description.required_skills or [])
        all_skills.extend(job_description.preferred_skills or [])
        
        # Filter skills relevant to the topic
        relevant_skills = []
        for skill in all_skills:
            if self._is_skill_relevant_to_topic(skill, topic):
                relevant_skills.append(skill)
        
        return relevant_skills
    
    def _is_skill_relevant_to_topic(self, skill: str, topic: str) -> bool:
        """Check if a skill is relevant to a specific topic.
        
        Args:
            skill: The skill to check.
            topic: The topic to check relevance against.
            
        Returns:
            True if the skill is relevant to the topic.
        """
        skill_lower = skill.lower()
        topic_lower = topic.lower()
        
        # Direct topic match
        if skill_lower == topic_lower:
            return True
            
        # Check skill-topic mapping
        if skill_lower in self._skill_topic_mapping:
            return topic_lower in self._skill_topic_mapping[skill_lower]
            
        # Check if skill is a subtopic
        for main_topic, sub_topics in self._topic_hierarchy.items():
            if topic_lower in sub_topics and skill_lower in sub_topics:
                return True
                
        return False
    
    def _get_next_topic_recommendation(self, session: InterviewSession, 
                                           topic_coverage: Dict[str, TopicCoverage]) -> Optional[str]:
        """Get recommendation for the next topic to cover.
        
        Args:
            session: The current interview session.
            topic_coverage: Current topic coverage statistics.
            
        Returns:
            Recommended next topic or None if no recommendation.
        """
        if not session.job_description:
            return None
            
        # Get job requirements
        required_skills = session.job_description.required_skills or []
        preferred_skills = session.job_description.preferred_skills or []
        
        # Calculate topic priorities
        topic_priorities = {}
        for topic in self._get_all_topics():
            priority = 0.0
            
            # Higher priority for topics with required skills
            for skill in required_skills:
                if self._is_skill_relevant_to_topic(skill, topic):
                    priority += 2.0
                    
            # Medium priority for topics with preferred skills
            for skill in preferred_skills:
                if self._is_skill_relevant_to_topic(skill, topic):
                    priority += 1.0
            
            # Lower priority for topics already covered well
            if topic in topic_coverage:
                coverage = topic_coverage[topic]
                if coverage.average_score > 0.8 and coverage.questions_asked >= 2:
                    priority *= 0.5
                elif coverage.average_score < 0.4:
                    priority *= 1.5  # Higher priority for weak areas
            
            topic_priorities[topic] = priority
        
        # Return topic with highest priority
        if topic_priorities:
            return max(topic_priorities, key=topic_priorities.get)
        
        return None
    
    def _should_transition_topic(self, session: InterviewSession, 
                                     topic_coverage: Dict[str, TopicCoverage]) -> bool:
        """Determine if a topic transition is needed.
        
        Args:
            session: The current interview session.
            topic_coverage: Current topic coverage statistics.
            
        Returns:
            True if topic transition is recommended.
        """
        if not session.current_topic:
            return False
            
        current_coverage = topic_coverage.get(session.current_topic)
        if not current_coverage:
            return False
        
        # Transition if current topic is well covered
        if (current_coverage.questions_asked >= 3 and 
            current_coverage.average_score >= 0.7):
            return True
            
        # Transition if current topic shows consistent weakness
        if (current_coverage.questions_asked >= 2 and 
            current_coverage.average_score <= 0.4):
            return True
            
        # Transition if too much time spent on current topic
        if current_coverage.time_spent > 30:  # 30 minutes
            return True
        
        if session.current_difficulty == DifficultyLevel.HARD:
            return True
            
        return False
    
    def _generate_topic_insights(self, session: InterviewSession, 
                                     topic_coverage: Dict[str, TopicCoverage]) -> Dict[str, any]:
        """Generate insights about topic performance and recommendations.
        
        Args:
            session: The current interview session.
            topic_coverage: Current topic coverage statistics.
            
        Returns:
            Dictionary containing topic insights and recommendations.
        """
        insights = {
            "strong_areas": [],
            "weak_areas": [],
            "recommendations": [],
            "topic_progression": {}
        }
        
        # Identify strong and weak areas
        for topic, coverage in topic_coverage.items():
            if coverage.average_score >= 0.8:
                insights["strong_areas"].append({
                    "topic": topic,
                    "score": coverage.average_score,
                    "questions": coverage.questions_asked
                })
            elif coverage.average_score <= 0.4:
                insights["weak_areas"].append({
                    "topic": topic,
                    "score": coverage.average_score,
                    "questions": coverage.questions_asked,
                    "gaps": coverage.skill_gaps
                })
        
        # Generate recommendations
        for topic, coverage in topic_coverage.items():
            if coverage.average_score <= 0.4:
                insights["recommendations"].append({
                    "topic": topic,
                    "action": "focus_more",
                    "reason": f"Low performance ({coverage.average_score:.2f})",
                    "suggested_questions": 2
                })
            elif coverage.average_score >= 0.8 and coverage.questions_asked < 2:
                insights["recommendations"].append({
                    "topic": topic,
                    "action": "validate_consistency",
                    "reason": f"High performance but limited questions ({coverage.questions_asked})",
                    "suggested_questions": 1
                })
        
        # Track topic progression
        for topic, coverage in topic_coverage.items():
            insights["topic_progression"][topic] = {
                "difficulty_progression": [d.value for d in coverage.difficulty_progression],
                "score_trend": self._calculate_score_trend(coverage),
                "time_efficiency": coverage.average_score / (coverage.time_spent / 60) if coverage.time_spent > 0 else 0
            }
        
        return insights
    
    def _calculate_score_trend(self, coverage: TopicCoverage) -> str:
        """Calculate the score trend for a topic.
        
        Args:
            coverage: Topic coverage data.
            
        Returns:
            Trend description: "improving", "declining", "stable", or "insufficient_data".
        """
        if coverage.questions_asked < 2:
            return "insufficient_data"
            
        # Simple trend calculation based on difficulty progression
        if len(coverage.difficulty_progression) >= 2:
            first_score = coverage.average_score
            last_score = coverage.average_score  # Simplified for now
            
            if last_score > first_score + 0.1:
                return "improving"
            elif last_score < first_score - 0.1:
                return "declining"
            else:
                return "stable"
        
        return "insufficient_data"
    
    def _calculate_session_progress(self, session: InterviewSession) -> Dict[str, any]:
        """Calculate overall session progress metrics.
        
        Args:
            session: The interview session.
            
        Returns:
            Dictionary containing progress metrics.
        """
        total_questions = session.max_questions
        questions_asked = len(session.questions)
        questions_remaining = total_questions - questions_asked
        
        # Calculate time progress
        session_duration = session.session_duration()
        estimated_total_time = total_questions * 15  # 15 minutes per question average
        time_progress = min(session_duration.total_seconds() / (estimated_total_time * 60), 1.0) if estimated_total_time > 0 else 0.0
        
        # Calculate topic coverage progress
        unique_topics = len(set(q.topic for q in session.questions))
        estimated_topics = min(total_questions // 2, 8)  # Estimate 2 questions per topic, max 8 topics
        topic_progress = min(unique_topics / estimated_topics, 1.0) if estimated_topics > 0 else 0.0
        
        return {
            "questions_progress": questions_asked / total_questions if total_questions > 0 else 0.0,
            "questions_remaining": questions_remaining,
            "time_progress": time_progress,
            "topic_progress": topic_progress,
            "unique_topics_covered": unique_topics,
            "estimated_completion": questions_remaining * 15  # minutes
        }
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        await super().cleanup()
        self.log_operation("TopicManagerAgent cleanup completed")
    
    @property
    def health_status(self) -> Dict[str, any]:
        """Get the health status of the agent."""
        return {
            "agent": self.agent_name,
            "status": "healthy" if self._initialized else "initializing",
            "topics_configured": len(self._topic_hierarchy),
            "skill_mappings": len(self._skill_topic_mapping),
            "correlation_id": self._correlation_id
        }
