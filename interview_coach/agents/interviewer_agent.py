"""Interviewer Agent for generating contextual interview questions."""

import asyncio
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..models.interview import Question
from ..models.enums import DifficultyLevel
from ..services.llm_manager import LLMProviderManager

from ..utils.exceptions import QuestionGenerationError
from .base_agent import BaseAgent


class QuestionContext:
    """Context for question generation."""

    def __init__(
        self,
        topic: str,
        difficulty: DifficultyLevel,
        resume_data: Dict[str, Any],
        job_requirements: Dict[str, Any],
        previous_questions: List[Question],
        performance_history: List[float],
    ):
        """Initialize question context.
        
        Args:
            topic: Current interview topic
            difficulty: Desired difficulty level
            resume_data: Candidate resume information
            job_requirements: Job requirements and description
            previous_questions: Previously asked questions
            performance_history: Performance scores from previous questions
        """
        self.topic = topic
        self.difficulty = difficulty
        self.resume_data = resume_data
        self.job_requirements = job_requirements
        self.previous_questions = previous_questions
        self.performance_history = performance_history



class InterviewerAgent(BaseAgent):
    """Generates contextual interview questions."""

    def __init__(self, llm_manager: LLMProviderManager):
        """Initialize the InterviewerAgent.
        
        Args:
            llm_manager: LLM provider manager for question generation
        """
        super().__init__("InterviewerAgent")
        self.llm_manager = llm_manager
        self.question_templates = self._load_question_templates()

    async def process(self, input_data: Any) -> Any:
        """Process input data and return results.
        
        Args:
            input_data: Input data to process (can be QuestionContext or other data)
            
        Returns:
            Processed results (Question object or other results)
            
        Raises:
            QuestionGenerationError: If processing fails
        """
        if isinstance(input_data, QuestionContext):
            return await self.generate_question(input_data)
        else:
            raise QuestionGenerationError(f"Unsupported input data type: {type(input_data)}")

    def _initialize_resources(self) -> None:
        """Initialize agent resources."""
        self.llm_manager.initialize_providers()
        self.logger.info("InterviewerAgent resources initialized")

    async def _cleanup_resources(self) -> None:
        """Cleanup agent resources."""
        # Cleanup any agent-specific resources
        self.logger.info("InterviewerAgent resources cleaned up")

    async def generate_question(self, context: QuestionContext) -> Question:
        """Generate a new question based on context.
        
        Args:
            context: Context for question generation
            
        Returns:
            Generated Question object
            
        Raises:
            QuestionGenerationError: If question generation fails
        """
        self.log_operation("generate_question", {"topic": context.topic, "difficulty": context.difficulty})
        
        try:
            
            # Generate new question using LLM
            question_content = await self._generate_question_content(context)
            
            # Create question object
            question = Question(
                question_id=str(uuid4()),
                content=question_content,
                topic=context.topic,
                difficulty=context.difficulty,
                context=context.resume_data,
                generated_by="LLM",
                expected_answer_points=await self._generate_expected_points(question_content),
                time_limit_minutes=self._get_time_limit(context.difficulty),
            )
            

            
            self.logger.info(f"Generated new question for topic: {context.topic}")
            return question
            
        except Exception as e:
            self.log_error(e, {"context": context.topic, "difficulty": context.difficulty})
            raise QuestionGenerationError(f"Failed to generate question: {e}")

    async def adapt_difficulty(self, question: Question, performance: float) -> Question:
        """Adapt question difficulty based on performance.
        
        Args:
            question: Original question
            performance: Performance score (0.0 to 1.0)
            
        Returns:
            Adapted question with new difficulty
        """
        self.log_operation("adapt_difficulty", {"performance": performance})
        
        # Determine new difficulty based on performance
        if performance >= 0.8:
            new_difficulty = self._increase_difficulty(question.difficulty)
        elif performance <= 0.4:
            new_difficulty = self._decrease_difficulty(question.difficulty)
        else:
            new_difficulty = question.difficulty
        
        if new_difficulty != question.difficulty:
            # Create new question with adapted difficulty
            adapted_question = Question(
                question_id=str(uuid4()),
                content=question.content,
                topic=question.topic,
                difficulty=new_difficulty,
                context=question.context,
                generated_by=f"Adapted from {question.generated_by}",
                expected_answer_points=question.expected_answer_points,
                time_limit_minutes=self._get_time_limit(new_difficulty),
            )
            
            self.logger.info(f"Adapted question difficulty from {question.difficulty} to {new_difficulty}")
            return adapted_question
        
        return question



    async def _generate_question_content(self, context: QuestionContext) -> str:
        """Generate question content using LLM."""
        
        try:
            # Handle both enum and integer difficulty values (Pydantic v2 compatibility)
            difficulty_value = context.difficulty
            if hasattr(context.difficulty, 'name'):
                difficulty_value = context.difficulty.name
            elif hasattr(context.difficulty, 'value'):
                difficulty_value = context.difficulty.value
            elif isinstance(context.difficulty, int):
                difficulty_value = context.difficulty
            
            from ..services.llm_manager import LLMRequest
            request = LLMRequest(
                type="question_generation",
                context= {
                    "topic": context.topic,
                    "difficulty": difficulty_value,
                    "resume_skills": context.resume_data.get("skills", []),
                    "job_requirements": context.job_requirements.get("required_skills", []),
                }
            )
            response = self.llm_manager.make_request(request)
            return response.content
            
        except Exception as e:
            self.logger.error(f"LLM question generation failed: {e}")
            # Fallback to template-based question
            return self._generate_template_question(context)

    async def _generate_expected_points(self, question_content: str) -> List[str]:
        """Generate expected answer points for the question."""
        # This would typically use LLM to generate expected points
        # For now, return a basic template
        return [
            "Clear understanding of the concept",
            "Practical application examples",
            "Problem-solving approach",
            "Code implementation if applicable",
        ]

    def _generate_template_question(self, context: QuestionContext) -> str:
        """Generate a template-based question as fallback."""
        templates = {
            DifficultyLevel.EASY: f"Can you explain the basic concepts of {context.topic}?",
            DifficultyLevel.MEDIUM: f"How would you implement {context.topic} in a real-world scenario?",
            DifficultyLevel.HARD: f"Design a system that demonstrates advanced {context.topic} principles.",
        }
        
        return templates.get(context.difficulty, f"Explain {context.topic}.")

    def _increase_difficulty(self, current_difficulty: DifficultyLevel) -> DifficultyLevel:
        """Increase question difficulty."""
        difficulty_map = {
            DifficultyLevel.EASY: DifficultyLevel.MEDIUM,
            DifficultyLevel.MEDIUM: DifficultyLevel.HARD,
            DifficultyLevel.HARD: DifficultyLevel.HARD,
        }
        return difficulty_map.get(current_difficulty, DifficultyLevel.HARD)

    def _decrease_difficulty(self, current_difficulty: DifficultyLevel) -> DifficultyLevel:
        """Decrease question difficulty."""
        difficulty_map = {
            DifficultyLevel.EASY: DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM: DifficultyLevel.EASY,
            DifficultyLevel.HARD: DifficultyLevel.MEDIUM,
        }
        return difficulty_map.get(current_difficulty, DifficultyLevel.EASY)

    def _get_time_limit(self, difficulty: DifficultyLevel) -> int:
        """Get time limit for question based on difficulty."""
        time_limits = {
            DifficultyLevel.EASY: 5,
            DifficultyLevel.MEDIUM: 10,
            DifficultyLevel.HARD: 15,
        }
        return time_limits.get(difficulty, 10)

    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question templates for fallback from external configuration file."""
        try:
            # Get the config directory path relative to the project root
            config_path = Path(__file__).parent.parent.parent / "config" / "question_templates.yaml"
            
            if not config_path.exists():
                self.logger.warning(f"Question templates file not found at {config_path}, using default templates")
                return self._get_default_templates()
            
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            templates = config_data.get("question_templates", {})
            if not templates:
                self.logger.warning("No question templates found in configuration file, using default templates")
                return self._get_default_templates()
            
            self.logger.info(f"Loaded {len(templates)} question template categories from configuration file")
            return templates
            
        except Exception as e:
            self.logger.error(f"Failed to load question templates from configuration file: {str(e)}")
            self.logger.info("Falling back to default question templates")
            return self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, List[str]]:
        """Get default question templates as fallback."""
        return {
            "Python": [
                "Explain the difference between lists and tuples in Python.",
                "How does Python handle memory management?",
                "What are decorators and how do you use them?",
            ],
            "JavaScript": [
                "Explain closures in JavaScript.",
                "How does the event loop work?",
                "What are the differences between var, let, and const?",
            ],
            "System Design": [
                "Design a URL shortening service.",
                "How would you design a chat application?",
                "Design a recommendation system.",
            ],
        }


