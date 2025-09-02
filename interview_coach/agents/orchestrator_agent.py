"""Orchestrator Agent for coordinating the interview process and managing agent interactions."""

import asyncio
import logging
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid

from ..models.interview import InterviewSession, Question, Evaluation, InterviewSessionReport, ReportFormat
from ..models.resume import ResumeData
from ..models.job import JobDescription
from ..models.enums import DifficultyLevel, SessionStatus, CustomerTier
from ..agents.base_agent import BaseAgent
from ..agents.interviewer_agent import InterviewerAgent, QuestionContext
from ..agents.topic_manager_agent import TopicManagerAgent
from ..agents.evaluator_agent import EvaluatorAgent, EvaluationContext
from ..services.llm_manager import LLMProviderManager
from ..services.storage_manager import StorageManager

from ..utils.exceptions import AgentError, OrchestrationError, SessionError
from ..utils.logging import set_correlation_id


@dataclass
class InterviewStep:
    """Represents a single step in the interview process."""
    
    step_number: int
    action: str
    agent: str
    status: str  # "pending", "in_progress", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class InterviewFlow:
    """Defines the flow of the interview process."""
    
    steps: List[InterviewStep]
    current_step: int
    total_steps: int
    status: str  # "initializing", "running", "paused", "completed", "failed"
    start_time: datetime
    estimated_duration: Optional[int] = None  # minutes


class OrchestratorAgent(BaseAgent):
    """Coordinates all agents and manages the overall interview flow."""
    
    def __init__(self, llm_manager: LLMProviderManager, 
                 storage_manager: StorageManager):
        """Initialize the OrchestratorAgent.
        
        Args:
            llm_manager: Manager for LLM provider interactions.
            storage_manager: Manager for data storage operations.
        """
        super().__init__("OrchestratorAgent")
        self.llm_manager = llm_manager
        self.storage_manager = storage_manager
        
        # Initialize sub-agents
        self.interviewer_agent: Optional[InterviewerAgent] = None
        self.topic_manager_agent: Optional[TopicManagerAgent] = None
        self.evaluator_agent: Optional[EvaluatorAgent] = None
        
        # Interview flow management
        self.interview_flow: Optional[InterviewFlow] = None
        self.session: Optional[InterviewSession] = None
        
        # Performance tracking
        self._step_performance: Dict[int, Dict[str, Any]] = {}
        self._agent_performance: Dict[str, Dict[str, Any]] = {}
        
    def initialize(self) -> None:
        """Initialize the agent and all sub-agents."""
        super().initialize()
    
    def _initialize_resources(self) -> None:
        """Initialize agent-specific resources."""
        try:
            # Initialize sub-agents
            self._initialize_sub_agents()
            
            # Initialize services
            self._initialize_services()
            
            self.log_operation("OrchestratorAgent resources initialized successfully")
            
        except Exception as e:
            self.log_error(f"Failed to initialize OrchestratorAgent resources: {str(e)}")
            raise OrchestrationError(f"Resource initialization failed: {str(e)}")
    
    async def _cleanup_resources(self) -> None:
        """Cleanup agent-specific resources."""
        try:
            # Cleanup sub-agents
            if self.interviewer_agent:
                await self.interviewer_agent.cleanup()
            if self.topic_manager_agent:
                await self.topic_manager_agent.cleanup()
            if self.evaluator_agent:
                await self.evaluator_agent.cleanup()
            
            self.log_operation("OrchestratorAgent resources cleaned up successfully")
            
        except Exception as e:
            self.log_error(f"Failed to cleanup OrchestratorAgent resources: {str(e)}")
    
    def _initialize_sub_agents(self) -> None:
        """Initialize all sub-agents."""
        # Initialize InterviewerAgent
        self.interviewer_agent = InterviewerAgent(self.llm_manager)
        self.interviewer_agent.initialize()
        
        # Initialize TopicManagerAgent
        self.topic_manager_agent = TopicManagerAgent(self.llm_manager)
        self.topic_manager_agent.initialize()
        
        # Initialize EvaluatorAgent
        self.evaluator_agent = EvaluatorAgent(self.llm_manager)
        self.evaluator_agent.initialize()
        
        self.log_operation("All sub-agents initialized successfully")
    
    def _initialize_services(self) -> None:
        """Initialize required services."""
        # Initialize LLM manager
        self.llm_manager.initialize_providers()
        
        # Initialize storage manager
        self.storage_manager.initialize()
        

        
        self.log_operation("All services initialized successfully")
    
    async def process(self, session: InterviewSession, **kwargs) -> Dict[str, Any]:
        """Process the interview session by orchestrating all agents.
        
        Args:
            session: The interview session to process.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dictionary containing orchestration results and session updates.
        """
        try:
            self.session = session
            set_correlation_id(session.session_id)
            
            # Initialize interview flow
            await self._initialize_interview_flow(session)
            
            # Execute interview flow
            results = await self._execute_interview_flow()
            
            # Generate final report if session is complete
            if session.status == SessionStatus.COMPLETED:
                report = await self._generate_final_report(session)
                results["final_report"] = report
            
            # Save session state
            await self._save_session_state(session)
            
            return results
            
        except Exception as e:
            self.log_error(f"Error orchestrating interview: {str(e)}")
            raise OrchestrationError(f"Interview orchestration failed: {str(e)}")
    
    async def start_interview(self, customer_id: str, customer_tier: CustomerTier, 
                            resume_data: dict, job_description: dict) -> InterviewSession:
        """Start a new interview session.
        
        Args:
            customer_id: Customer identifier
            customer_tier: Customer subscription tier
            resume_data: Parsed resume data
            job_description: Parsed job description data
            
        Returns:
            InterviewSession object
        """
        try:
            # Convert parsed dictionaries to model objects
            resume_data_obj = ResumeData(
                candidate_name=resume_data.get("candidate_name"),
                email=resume_data.get("email"),
                experience_years=resume_data.get("experience_years"),
                skills=resume_data.get("skills", []),
                work_history=[],
                education=[],
                certifications=[],
                summary=resume_data.get("content")
            )
            
            job_description_obj = JobDescription(
                title=job_description.get("title"),
                company=job_description.get("company"),
                experience_years=job_description.get("experience_years"),
                required_skills=job_description.get("required_skills", []),
                preferred_skills=[],
                responsibilities=[],
                technical_stack=[],
                benefits=[]
            )
            
            # Create new session
            session = InterviewSession(
                session_id=f"sess_{customer_id}_{int(datetime.now().timestamp())}",
                customer_id=customer_id,
                customer_tier=customer_tier,
                max_questions=20 if customer_tier == CustomerTier.MVP else 3,
                status=SessionStatus.CREATED,
                created_at=datetime.now(),
                resume_data=resume_data_obj,
                job_description=job_description_obj
            )
            
            self.session = session
            self.log_operation("start_interview", {"session_id": session.session_id})
            
            return session
            
        except Exception as e:
            self.log_error(f"Failed to start interview: {str(e)}")
            raise OrchestrationError(f"Failed to start interview: {str(e)}")
    
    async def generate_next_question(self, session: InterviewSession) -> Question:
        """Generate the next question for the interview session.
        
        Args:
            session: Current interview session
            
        Returns:
            Question object
        """
        try:
            if not self.topic_manager_agent:
                raise OrchestrationError("TopicManagerAgent not initialized")
            if not self.interviewer_agent:
                raise OrchestrationError("InterviewerAgent not initialized")
            
            next_topic_context = self.topic_manager_agent.process(session)
            # Create question context
            context = QuestionContext(
                topic=next_topic_context["next_topic"] if next_topic_context["next_topic"] else "Python",  # Default topic for now
                difficulty=next_topic_context["difficulty"] if next_topic_context["difficulty"] else DifficultyLevel.EASY,  # Default difficulty
                resume_data={
                    "skills": session.resume_data.skills,
                    "content": session.resume_data.summary or "",
                    "experience_years": session.resume_data.experience_years or 0
                },
                job_requirements={
                    "required_skills": session.job_description.required_skills,
                    "title": session.job_description.title or "",
                    "company": session.job_description.company or "",
                    "experience_years": session.job_description.experience_years or 0
                },
                previous_questions=session.questions,
                performance_history=[eval.overall_score for eval in session.evaluations]
            )
            
            question = await self.interviewer_agent.generate_question(context)
            session.questions.append(question)
            session.questions_asked += 1
            session.current_topic = question.topic
            session.current_difficulty = question.difficulty
            
            self.log_operation("generate_next_question", {"question_id": question.question_id})
            return question
            
        except Exception as e:
            self.log_error(f"Failed to generate next question: {str(e)}")
            raise OrchestrationError(f"Failed to generate next question: {str(e)}")
    
    async def evaluate_response(self, session: InterviewSession, question: Question, response: str) -> Evaluation:
        """Evaluate a candidate's response to a question.
        
        Args:
            session: Current interview session
            question: The question that was answered
            response: Candidate's response
            
        Returns:
            Evaluation object
        """
        try:
            if not self.evaluator_agent:
                raise OrchestrationError("EvaluatorAgent not initialized")
            
            # Create evaluation context
            context = EvaluationContext(
                question=question,
                candidate_response=response,
                resume_data=session.resume_data,
                job_description=session.job_description,
                response_time=60  # Default response time
            )
            
            evaluation = await self.evaluator_agent.process(context)
            
            # Add evaluation to session instead of trying to set non-existent fields on question
            session.add_evaluation(evaluation)
            
            self.log_operation("evaluate_response", {"question_id": question.question_id, "score": evaluation.overall_score})
            return evaluation
            
        except Exception as e:
            self.log_error(f"Failed to evaluate response: {str(e)}")
            raise OrchestrationError(f"Failed to evaluate response: {str(e)}")
    
    def complete_session(self, session: InterviewSession) -> InterviewSessionReport:
        """Complete the interview session.
        
        Args:
            session: Interview session to complete
        """
        try:
            # Ensure session status is properly set as an enum object
            if not isinstance(session.status, SessionStatus):
                self.logger.warning(f"Session status is not a SessionStatus enum: {session.status} (type: {type(session.status)})")
                # Try to convert it to the proper enum
                if isinstance(session.status, str):
                    if session.status.startswith("SessionStatus."):
                        status_name = session.status.split(".", 1)[1]
                        session.status = SessionStatus[status_name]
                    else:
                        session.status = SessionStatus[session.status]
                elif isinstance(session.status, int):
                    enum_values = list(SessionStatus)
                    if 0 <= session.status < len(enum_values):
                        session.status = enum_values[session.status]
                    else:
                        session.status = SessionStatus.COMPLETED
                else:
                    session.status = SessionStatus.COMPLETED
                self.logger.info(f"Converted session status to: {session.status}")
            
            session.status = SessionStatus.COMPLETED
            session.update_timestamp()  # This will update the updated_at field
            
            # Generate final report
            report = self._generate_final_report(session)
            session.final_report = report
            
            # Save session
            self._save_session_state(session)
            
            self.log_operation("complete_session", {"session_id": session.session_id})
            return report
        except Exception as e:
            self.logger.error(f"Failed to complete session: {str(e)}")
            self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise OrchestrationError(f"Failed to complete session: {str(e)}")
    
    async def _initialize_interview_flow(self, session: InterviewSession) -> None:
        """Initialize the interview flow for the session.
        
        Args:
            session: The interview session.
        """
        steps = []
        
        # Determine number of questions based on customer tier
        max_questions = session.max_questions
        
        # Create steps for each question
        for i in range(max_questions):
            # Topic analysis step
            steps.append(InterviewStep(
                step_number=i * 3 + 1,
                action="analyze_topic_coverage",
                agent="TopicManagerAgent",
                status="pending"
            ))
            
            # Question generation step
            steps.append(InterviewStep(
                step_number=i * 3 + 2,
                action="generate_question",
                agent="InterviewerAgent",
                status="pending"
            ))
            
            # Response evaluation step
            steps.append(InterviewStep(
                step_number=i * 3 + 3,
                action="evaluate_response",
                agent="EvaluatorAgent",
                status="pending"
            ))
        
        # Add final analysis step
        steps.append(InterviewStep(
            step_number=len(steps) + 1,
            action="final_analysis",
            agent="TopicManagerAgent",
            status="pending"
        ))
        
        self.interview_flow = InterviewFlow(
            steps=steps,
            current_step=0,
            total_steps=len(steps),
            status="initializing",
            start_time=datetime.now(),
            estimated_duration=max_questions * 20  # 20 minutes per question average
        )
        
        self.log_operation(f"Interview flow initialized with {len(steps)} steps")
    
    async def _execute_interview_flow(self) -> Dict[str, Any]:
        """Execute the interview flow step by step.
        
        Returns:
            Dictionary containing execution results.
        """
        if not self.interview_flow:
            raise OrchestrationError("Interview flow not initialized")
        
        self.interview_flow.status = "running"
        results = {
            "flow_status": "running",
            "steps_completed": 0,
            "current_step": 0,
            "step_results": {},
            "session_updates": {}
        }
        
        try:
            for i, step in enumerate(self.interview_flow.steps):
                self.interview_flow.current_step = i
                step.status = "in_progress"
                step.start_time = datetime.now()
                
                self.log_operation(f"Executing step {step.step_number}: {step.action}")
                
                # Execute step
                step_result = await self._execute_step(step)
                
                # Update step status
                step.status = "completed"
                step.end_time = datetime.now()
                step.result = step_result
                
                # Track performance
                self._track_step_performance(step, step_result)
                
                # Update results
                results["step_results"][step.step_number] = step_result
                results["steps_completed"] += 1
                results["current_step"] = i + 1
                
                # Check if session should continue
                if not await self._should_continue_session():
                    self.log_operation("Session completion criteria met, stopping flow")
                    break
                
                # Auto-save session every few steps
                if i > 0 and i % 3 == 0:
                    await self._auto_save_session()
                
                # Small delay between steps
                await asyncio.sleep(0.1)
            
            self.interview_flow.status = "completed"
            results["flow_status"] = "completed"
            
        except Exception as e:
            self.log_error(f"Error executing interview flow: {str(e)}")
            self.interview_flow.status = "failed"
            results["flow_status"] = "failed"
            results["error"] = str(e)
            
            # Mark current step as failed
            if self.interview_flow.current_step < len(self.interview_flow.steps):
                current_step = self.interview_flow.steps[self.interview_flow.current_step]
                current_step.status = "failed"
                current_step.error = str(e)
        
        return results
    
    async def _execute_step(self, step: InterviewStep) -> Dict[str, Any]:
        """Execute a single interview step.
        
        Args:
            step: The step to execute.
            
        Returns:
            Dictionary containing step execution results.
        """
        try:
            if step.action == "analyze_topic_coverage":
                return await self._execute_topic_analysis()
            elif step.action == "generate_question":
                return await self._execute_question_generation()
            elif step.action == "evaluate_response":
                return await self._execute_response_evaluation()
            elif step.action == "final_analysis":
                return await self._execute_final_analysis()
            else:
                raise OrchestrationError(f"Unknown step action: {step.action}")
                
        except Exception as e:
            self.log_error(f"Error executing step {step.step_number}: {str(e)}")
            step.status = "failed"
            step.error = str(e)
            raise
    
    async def _execute_topic_analysis(self) -> Dict[str, Any]:
        """Execute topic analysis step.
        
        Returns:
            Dictionary containing topic analysis results.
        """
        if not self.topic_manager_agent or not self.session:
            raise OrchestrationError("Topic manager agent or session not available")
        
        # Get topic management insights
        topic_insights = await self.topic_manager_agent.process(self.session)
        
        # Update session with topic insights
        if "next_topic" in topic_insights and topic_insights["next_topic"]:
            self.session.current_topic = topic_insights["next_topic"]
        
        # Check if topic transition is needed
        if topic_insights.get("transition_needed", False):
            self.log_operation(f"Topic transition recommended to: {topic_insights.get('next_topic')}")
        
        return {
            "action": "topic_analysis",
            "topic_insights": topic_insights,
            "session_updated": True
        }
    
    async def _execute_question_generation(self) -> Dict[str, Any]:
        """Execute question generation step.
        
        Returns:
            Dictionary containing question generation results.
        """
        if not self.interviewer_agent or not self.session:
            raise OrchestrationError("Interviewer agent or session not available")
        
        # Create question context
        from ..agents.interviewer_agent import QuestionContext
        
        context = QuestionContext(
            resume_data=self.session.resume_data.model_dump(),
            job_description=self.session.job_description.model_dump(),
            current_topic=self.session.current_topic,
            current_difficulty=self.session.current_difficulty
        )
        
        # Generate question
        question = await self.interviewer_agent.generate_question(context)
        
        # Add question to session
        self.session.add_question(question)
        
        # Update session topic and difficulty
        self.session.current_topic = question.topic
        self.session.current_difficulty = question.difficulty
        
        return {
            "action": "question_generation",
            "question_generated": True,
            "question_id": question.id,
            "question_topic": question.topic,
            "question_difficulty": question.difficulty.value,
            "session_updated": True
        }
    
    async def _execute_response_evaluation(self) -> Dict[str, Any]:
        """Execute response evaluation step.
        
        Returns:
            Dictionary containing response evaluation results.
        """
        if not self.evaluator_agent or not self.session:
            raise OrchestrationError("Evaluator agent or session not available")
        
        # Get the most recent question
        if not self.session.questions:
            raise OrchestrationError("No questions available for evaluation")
        
        latest_question = self.session.questions[-1]
        
        # Simulate candidate response (in real implementation, this would come from user input)
        candidate_response = await self._get_candidate_response(latest_question)
        
        # Create evaluation context
        context = EvaluationContext(
            question=latest_question,
            candidate_response=candidate_response,
            resume_data=self.session.resume_data,
            job_description=self.session.job_description,
            response_time=30,  # Simulated response time
            difficulty_level=latest_question.difficulty
        )
        
        # Evaluate response
        evaluation = await self.evaluator_agent.process(context)
        
        # Add evaluation to session
        self.session.add_evaluation(evaluation)
        
        # Update session difficulty based on performance
        await self._adapt_difficulty(evaluation)
        
        return {
            "action": "response_evaluation",
            "evaluation_completed": True,
            "evaluation_id": evaluation.id,
            "overall_score": evaluation.overall_score,
            "difficulty_adapted": True,
            "session_updated": True
        }
    
    async def _execute_final_analysis(self) -> Dict[str, Any]:
        """Execute final analysis step.
        
        Returns:
            Dictionary containing final analysis results.
        """
        if not self.topic_manager_agent or not self.session:
            raise OrchestrationError("Topic manager agent or session not available")
        
        # Get comprehensive topic analysis
        final_analysis = await self.topic_manager_agent.process(self.session)
        
        # Mark session as completed
        self.session.status = SessionStatus.COMPLETED
        self.session.end_time = datetime.now()
        
        return {
            "action": "final_analysis",
            "final_analysis_completed": True,
            "session_completed": True,
            "topic_insights": final_analysis,
            "session_updated": True
        }
    
    async def _get_candidate_response(self, question: Question) -> str:
        """Get candidate response for a question.
        
        Args:
            question: The question to get response for.
            
        Returns:
            Simulated candidate response.
        """
        # In a real implementation, this would get input from the user
        # For now, return a simulated response based on question type
        
        if "code" in question.content.lower() or "implement" in question.content.lower():
            return "Here's my solution:\n\ndef solution():\n    # Implementation here\n    pass"
        elif "design" in question.content.lower():
            return "I would design this system with the following components..."
        else:
            return "This is my approach to solving this problem..."
    
    async def _adapt_difficulty(self, evaluation: Evaluation) -> None:
        """Adapt question difficulty based on evaluation performance.
        
        Args:
            evaluation: The evaluation result.
        """
        if not self.session:
            return
        
        overall_score = evaluation.overall_score or 0.0
        
        if overall_score >= 0.8:
            # Increase difficulty
            if self.session.current_difficulty == DifficultyLevel.EASY:
                self.session.current_difficulty = DifficultyLevel.MEDIUM
            elif self.session.current_difficulty == DifficultyLevel.MEDIUM:
                self.session.current_difficulty = DifficultyLevel.HARD
        elif overall_score <= 0.4:
            # Decrease difficulty
            if self.session.current_difficulty == DifficultyLevel.HARD:
                self.session.current_difficulty = DifficultyLevel.MEDIUM
            elif self.session.current_difficulty == DifficultyLevel.MEDIUM:
                self.session.current_difficulty = DifficultyLevel.EASY
        
        self.log_operation(f"Difficulty adapted to: {self.session.current_difficulty.value}")
    
    async def _should_continue_session(self) -> bool:
        """Determine if the session should continue.
        
        Returns:
            True if session should continue, False otherwise.
        """
        if not self.session:
            return False
        
        # Check if max questions reached
        if len(self.session.questions) >= self.session.max_questions:
            return False
        
        # Check if session time limit reached (if applicable)
        if self.session.start_time:
            session_duration = datetime.now() - self.session.start_time
            if session_duration.total_seconds() > 7200:  # 2 hours
                return False
        
        # Check if session was manually stopped
        if self.session.status == SessionStatus.STOPPED:
            return False
        
        return True
    
    async def _auto_save_session(self) -> None:
        """Auto-save the current session state."""
        if not self.session:
            return
        
        try:
            await self.storage_manager.save_session(self.session)
            self.log_operation("Session auto-saved successfully")
        except Exception as e:
            self.log_error(f"Failed to auto-save session: {str(e)}")
    
    def _save_session_state(self, session: InterviewSession) -> None:
        """Save the final session state.
        
        Args:
            session: The session to save.
        """
        try:
            self.storage_manager.save_session(session)
            self.log_operation("Final session state saved successfully")
        except Exception as e:
            self.log_error(f"Failed to save final session state: {str(e)}")
    
    def _generate_final_report(self, session: InterviewSession) -> InterviewSessionReport:
        """Generate the final interview session report.
        
        Args:
            session: The completed interview session.
            
        Returns:
            InterviewSessionReport object.
        """
        try:
            # Get topic analysis for final insights
            topic_insights = self.topic_manager_agent.process(session)
            
            # Create report
            report = InterviewSessionReport(
                report_id=str(uuid.uuid4()),
                session_id=session.session_id,
                customer_id=session.customer_id,
                customer_tier=session.customer_tier,
                resume_data=session.resume_data,
                job_description=session.job_description,
                interview_date=session.start_time,
                total_questions=len(session.questions),
                questions_answered=len(session.evaluations),
                session_duration=session.session_duration(),
                session_status=session.status,
                total_evaluations = len(session.evaluations),
                question_evaluations=session.evaluations,
                generated_at=datetime.now(),
                report_format=ReportFormat(format_type="comprehensive", include_charts=True, include_code_samples=True, anonymize_data=False)
            )
            self.logger.info(f"Report created with session_status: {report.session_status} (type: {type(report.session_status)})")

            # Get evaluation summary
            if session.evaluations and len(session.evaluations) > 0:
                evaluations = session.evaluations              
                avg_scores = {
                    "technical": sum(e.technical_score or 0 for e in evaluations) / len(evaluations),
                    "problem_solving": sum(e.problem_solving_score or 0 for e in evaluations) / len(evaluations),
                    "communication": sum(e.communication_score or 0 for e in evaluations) / len(evaluations),
                    "code_quality": sum(e.code_quality_score or 0 for e in evaluations) / len(evaluations),
                    "overall": sum(e.overall_score or 0 for e in evaluations) / len(evaluations)
                }
                report.average_scores = avg_scores
        
                # Identify common areas for improvement
                improvement_areas = {}
                for evaluation in evaluations:
                    for area in evaluation.improvement_suggestions or []:
                        improvement_areas[area] = improvement_areas.get(area, 0) + 1
                
                # Sort improvement areas by frequency
                sorted_improvements = sorted(
                    improvement_areas.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                report.top_improvement_areas = [item[0] for item in sorted_improvements[:5]]

                # Get score distribution
                distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
                for evaluation in evaluations:
                    score = evaluation.overall_score or 0
                    if score >= 0.8:
                        distribution["excellent"] += 1
                    elif score >= 0.6:
                        distribution["good"] += 1
                    elif score >= 0.4:
                        distribution["fair"] += 1
                    else:
                        distribution["poor"] += 1
                report.score_distribution = distribution

                report.improvement_recommendations.extend(self._generate_recommendations(session, topic_insights, avg_scores))
            
            # Save report
            self.storage_manager.save_report(report)
            
            return report
            
        except Exception as e:
            self.log_error(f"Failed to generate final report: {str(e)}")
            raise OrchestrationError(f"Report generation failed: {str(e)}")
    
    def _generate_recommendations(self, session: InterviewSession, 
                                topic_insights: Dict[str, Any],
                                average_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on interview results.
        
        Args:
            session: The interview session.
            topic_insights: Insights from topic analysis.
            average_scores: Average scores by category.
            
        Returns:
            List of recommendations.
        """
        recommendations = []
        
        # Add recommendations based on weak areas
        weak_areas = topic_insights.get("insights", {}).get("weak_areas", [])
        for area in weak_areas:
            recommendations.append(f"Focus on improving {area['topic']} skills")
        
        # Add recommendations based on average scores
        if average_scores:
            if average_scores.get("technical", 0) < 0.6:
                recommendations.append("Strengthen technical knowledge fundamentals")
            if average_scores.get("communication", 0) < 0.6:
                recommendations.append("Improve communication and explanation skills")
            if average_scores.get("problem_solving", 0) < 0.6:
                recommendations.append("Practice systematic problem-solving approaches")
        
        # Add general recommendations
        if session.get_average_score() or 0 < 0.5:
            recommendations.append("Consider additional preparation before next interview")
        elif session.get_average_score() or 0 > 0.8:
            recommendations.append("Excellent performance! Consider more challenging questions next time")
        
        self.logger.info(recommendations)
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _track_step_performance(self, step: InterviewStep, result: Dict[str, Any]) -> None:
        """Track performance metrics for a step.
        
        Args:
            step: The completed step.
            result: The step execution result.
        """
        if step.start_time and step.end_time:
            duration = (step.end_time - step.start_time).total_seconds()
            
            self._step_performance[step.step_number] = {
                "duration": duration,
                "status": step.status,
                "agent": step.agent,
                "action": step.action,
                "success": step.status == "completed"
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the orchestrator.
        
        Returns:
            Dictionary containing performance metrics.
        """
        if not self.interview_flow:
            return {}
        
        total_steps = len(self.interview_flow.steps)
        completed_steps = sum(1 for step in self.interview_flow.steps if step.status == "completed")
        failed_steps = sum(1 for step in self.interview_flow.steps if step.status == "failed")
        
        # Calculate average step duration
        step_durations = [
            perf["duration"] for perf in self._step_performance.values()
            if "duration" in perf
        ]
        avg_step_duration = sum(step_durations) / len(step_durations) if step_durations else 0
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "success_rate": completed_steps / total_steps if total_steps > 0 else 0,
            "average_step_duration": avg_step_duration,
            "flow_status": self.interview_flow.status if self.interview_flow else "unknown",
            "agent_performance": self._agent_performance
        }
    
    async def cleanup(self) -> None:
        """Clean up agent resources and sub-agents."""
        try:
            # Clean up sub-agents
            if self.interviewer_agent:
                await self.interviewer_agent.cleanup()
            if self.topic_manager_agent:
                await self.topic_manager_agent.cleanup()
            if self.evaluator_agent:
                await self.evaluator_agent.cleanup()
            
            # Clean up services
            if self.storage_manager:
                await self.storage_manager.cleanup()
            
            await super().cleanup()
            self.log_operation("OrchestratorAgent cleanup completed")
            
        except Exception as e:
            self.log_error(f"Error during cleanup: {str(e)}")
    
    @property
    def health_status(self) -> Dict[str, Any]:
        """Get the health status of the agent and all sub-agents."""
        sub_agent_status = {}
        
        if self.interviewer_agent:
            sub_agent_status["interviewer"] = self.interviewer_agent.health_status
        if self.topic_manager_agent:
            sub_agent_status["topic_manager"] = self.topic_manager_agent.health_status
        if self.evaluator_agent:
            sub_agent_status["evaluator"] = self.evaluator_agent.health_status
        
        return {
            "agent": self.agent_name,
            "status": "healthy" if self._initialized else "initializing",
            "sub_agents": sub_agent_status,
            "interview_flow": {
                "status": self.interview_flow.status if self.interview_flow else "none",
                "current_step": self.interview_flow.current_step if self.interview_flow else 0,
                "total_steps": self.interview_flow.total_steps if self.interview_flow else 0
            },
            "correlation_id": self._correlation_id
        }
