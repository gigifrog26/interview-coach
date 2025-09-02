"""CLI interface for the Interview Coach System."""
import traceback
import asyncio
import sys
from pathlib import Path
from typing import Optional
import time
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

from .services.configuration_manager import ConfigurationManager
from .services.storage_manager import StorageManager

from .services.llm_manager import LLMProviderManager
from .services.parsing_service import ParsingService
from .agents.orchestrator_agent import OrchestratorAgent
from .models.enums import CustomerTier
from .models.interview import Question, Evaluation
from .utils.logging import setup_logging, get_logger
from .utils.exceptions import InterviewCoachError


console = Console()
logger = get_logger("cli")


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool):
    """Intelligent Interview Coach System - AI-powered mock technical interviews."""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = "DEBUG" if verbose else "ERROR"
    setup_logging(log_level)
    
    # Initialize configuration
    try:
        config_path = config or "config"
        ctx.obj["config_manager"] = ConfigurationManager(config_path)
        ctx.obj["config_manager"].initialize()
        logger.info("Configuration manager initialized")
        
        # Initialize storage manager
        storage_config = ctx.obj["config_manager"].get_storage_config()
        ctx.obj["storage_manager"] = StorageManager("file", base_path=storage_config.get("base_path", "data"))
        ctx.obj["storage_manager"].initialize()
        logger.info("Storage manager initialized")
        


        # Initialize LLM provider manager
        ctx.obj["llm_manager"] = LLMProviderManager(ctx.obj["config_manager"])
        ctx.obj["llm_manager"].initialize()
        logger.info("LLM manager initialized")

        ctx.obj["parsing_service"] = ParsingService()
        ctx.obj["parsing_service"].initialize()
        logger.info("Parsing service initialized")

        ctx.obj["orchestrator"] = OrchestratorAgent(
            llm_manager=ctx.obj["llm_manager"],
            storage_manager=ctx.obj["storage_manager"]
        )   
        ctx.obj["orchestrator"].initialize()
        logger.info("Orchestrator agent initialized")
        
        logger.info("CLI initialized successfully")
        
    except Exception as e:
        console.print(f"[red]Failed to initialize system: {e}[/red]")
        logger.error(f"CLI initialization failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--resume", "-r", type=click.Path(exists=True), required=True, help="Path to resume file")
@click.option("--job-desc", "-j", type=click.Path(exists=True), required=True, help="Path to job description file")
@click.option("--customer-id", "-c", required=True, help="Customer identifier")
@click.option("--tier", "-t", type=click.Choice(["standard", "mvp"]), default="standard", help="Customer tier")
@click.pass_context
def start_interview(ctx: click.Context, resume: str, job_desc: str, customer_id: str, tier: str):
    """Start a new interview session."""
    try:
        # Parse customer tier
        customer_tier = CustomerTier.MVP if tier == "mvp" else CustomerTier.STANDARD
        
        # Display welcome message
        welcome_content = "\n".join([
            "🎯 Welcome to the Intelligent Interview Coach System!",
            f"Customer: {customer_id} ({tier.upper()})",
            f"Resume: {Path(resume).name}",
            f"Job Description: {Path(job_desc).name}"
        ])
        welcome_panel = Panel(
            welcome_content,
            title="Interview Session Setup",
            border_style="blue"
        )
        console.clear()
        console.print(welcome_panel)
        
        # Run interview asynchronously
        asyncio.run(_run_interview(ctx, resume, job_desc, customer_id, customer_tier))
        
    except Exception as e:
        console.print(f"[red]Failed to start interview: {e}[/red]")
        logger.error(f"Interview start failed: {e}")
        sys.exit(1)



@cli.command()
@click.pass_context
def help(ctx: click.Context):
    """Show detailed help information."""
    help_text = """
    [bold blue]Intelligent Interview Coach System[/bold blue]
    
    This system provides AI-powered mock technical interviews using a multi-agent architecture.
    
    [bold]Commands:[/bold]
    • start-interview: Start a new interview session
    • help: Show this help message
    
    [bold]Customer Tiers:[/bold]
    • standard: Up to 3 questions per session
    • mvp: Up to 20 questions per session
    
    [bold]Example Usage:[/bold]
    interview-coach start-interview -r resume.txt -j job.txt -c user123 -t mvp
    """
    
    help_panel = Panel(help_text, title="Help", border_style="blue")
    console.print(help_panel)


async def _run_interview(ctx: click.Context, resume_path: str, job_desc_path: str, 
                        customer_id: str, customer_tier: CustomerTier):
    """Run the interview session."""
    try:
        # Load resume and job description
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Loading resume and job description...", total=None)
            
            # Parse resume and job description
            resume_data = await _parse_resume(resume_path)
            job_description = await _parse_job_description(job_desc_path)
            
            progress.update(task, description="Initializing interview system...")
            
            # Initialize orchestrator
            orchestrator = ctx.obj["orchestrator"]
            
            progress.update(task, description="Starting interview session...")
            
            # Start interview
            session = await orchestrator.start_interview(
                customer_id=customer_id,
                customer_tier=customer_tier,
                resume_data=resume_data,
                job_description=job_description
            )
            
            progress.update(task, description="Interview session started!")

            time.sleep(2)
            
        # Run the interview loop
        await _interview_loop(orchestrator, session)
        
    except Exception as e:
        console.print(f"[red]Interview failed: {e}[/red]")
        logger.error(f"Interview execution failed: {e}")
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        raise


async def _interview_loop(orchestrator: OrchestratorAgent, session):
    """Main interview loop."""
    try:
        console.print("\n[bold green]🎯 Interview Session Started![/bold green]")
        
        while session.can_ask_more_questions():
            # Generate next question with loading indicator
            question = await _generate_question_with_loading(orchestrator, session)
            
            # Display question
            question_content = f"[bold]Question {session.questions_asked}:[/bold]\n\n{question.content}"
            
            # Handle both enum and integer difficulty values (Pydantic v2 compatibility)
            difficulty_display = question.difficulty
            if hasattr(question.difficulty, 'name'):
                difficulty_display = question.difficulty.name
            elif hasattr(question.difficulty, 'value'):
                difficulty_display = question.difficulty.value
            elif isinstance(question.difficulty, int):
                # Convert integer back to enum name for display
                difficulty_names = {0: "EASY", 1: "MEDIUM", 2: "HARD"}
                difficulty_display = difficulty_names.get(question.difficulty, str(question.difficulty))
            
            question_panel = Panel(
                question_content,
                title=f"Topic: {question.topic} | Difficulty: {difficulty_display} | Progress: {session.questions_asked}/{session.max_questions}",
                border_style="green"
            )
            console.print(question_panel)
            
            # Get candidate response
            response = await _get_candidate_response(question)
            
            if "candidate chose to skip this question." != response.lower():
                # Evaluate response with loading indicator
                evaluation = await _evaluate_response_with_loading(orchestrator, session, question, response)
            
                # Display evaluation
                evaluation_content = f"[bold]Evaluation:[/bold]\n\n{evaluation.feedback}"
                evaluation_panel = Panel(
                    evaluation_content,
                    title=f"Evaluation Result | Score: {evaluation.overall_score:.1f}",
                    border_style="cyan"
                )
                console.print(evaluation_panel)
            else:
                console.print("[bold yellow]Candidate chose to skip this question.[/bold yellow]")
            
            # Check if session should continue
            if not session.can_ask_more_questions():
                break
            
            # Ask if candidate wants to continue
            continue_interview = await _ask_continue_interview()
            if not continue_interview:
                break
        
        # Complete session
        report = orchestrator.complete_session(session)
        console.print(f"\n[bold green]🎉 Interview session completed![/bold green]")

        # Display report
        # Handle different types of session_status
        if hasattr(report.session_status, 'name'):
            session_status_str = report.session_status.name
        elif isinstance(report.session_status, str):
            session_status_str = report.session_status
        elif isinstance(report.session_status, int):
            session_status_str = str(report.session_status)
        else:
            session_status_str = str(report.session_status)
        summary_message = f"Total Questions: {report.total_questions}\nQuestions Answered: {report.questions_answered}\nSession Duration: {report.session_duration}\nSession Status: {session_status_str}"
        if report.average_scores:
            average_scores_message = f"Average Scores: \n- Technical: {report.average_scores['technical']:.1f}/10.0\n- Problem Solving: {report.average_scores['problem_solving']:.1f}/10.0\n- Communication: {report.average_scores['communication']:.1f}/10.0\n- Code Quality: {report.average_scores['code_quality']:.1f}/10.0\n- Overall: {report.average_scores['overall']:.1f}/10.0"
        else:
            average_scores_message = "Average Scores: N/A"
        top_suggestions_message = f"Top Suggestions: \n- {"\n- ".join(report.top_improvement_areas)}"
        if report.improvement_recommendations:
            recommendations_message = "\n- ".join(report.improvement_recommendations)
        else:
            recommendations_message = "N/A"
        report_content = f"[bold]Final Report:[/bold]\n\n{summary_message}\n\n{average_scores_message}\n\nRecommendations:\n- {recommendations_message}\n\n{top_suggestions_message}"
        report_panel = Panel(
            report_content,
            title=f"Final Report",
            border_style="blue"
        )
        console.clear()
        console.print(report_panel)
        
    except Exception as e:
        console.print(f"[red]Interview loop failed: {e}[/red]")
        logger.error(f"Interview loop failed: {e}")
        raise


async def _parse_resume(resume_path: str) -> dict:
    """Parse resume file and extract relevant information."""
    # This is a simplified implementation
    # In a real system, you'd use more sophisticated parsing
    try:
        with open(resume_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic parsing logic (simplified)
        resume_data = {
            "content": content,
            "skills": _extract_skills(content),
            "experience_years": _extract_experience_years(content),
        }
        
        return resume_data
        
    except Exception as e:
        raise InterviewCoachError(f"Failed to parse resume: {e}")


async def _parse_job_description(job_desc_path: str) -> dict:
    """Parse job description file and extract requirements."""
    try:
        with open(job_desc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic parsing logic (simplified)
        job_data = {
            "content": content,
            "required_skills": _extract_required_skills(content),
            "experience_years": _extract_job_experience_years(content),
        }
        
        return job_data
        
    except Exception as e:
        raise InterviewCoachError(f"Failed to parse job description: {e}")


def _extract_skills(content: str) -> list:
    """Extract skills from resume content."""
    # Simplified skill extraction
    common_skills = ["Python", "JavaScript", "Java", "C++", "React", "Node.js", "AWS", "Docker"]
    found_skills = []
    
    for skill in common_skills:
        if skill.lower() in content.lower():
            found_skills.append(skill)
    
    return found_skills


def _extract_experience_years(content: str) -> int:
    """Extract years of experience from resume content."""
    # Simplified experience extraction
    import re
    
    experience_patterns = [
        r"(\d+)\s*years?\s*experience",
        r"experience:\s*(\d+)",
        r"(\d+)\s*years?\s*in",
    ]
    
    for pattern in experience_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 2  # Default fallback


def _extract_required_skills(content: str) -> list:
    """Extract required skills from job description."""
    # Simplified skill extraction
    common_skills = ["Python", "JavaScript", "Java", "C++", "React", "Node.js", "AWS", "Docker"]
    found_skills = []
    
    for skill in common_skills:
        if skill.lower() in content.lower():
            found_skills.append(skill)
    
    return found_skills


def _extract_job_experience_years(content: str) -> int:
    """Extract required years of experience from job description."""
    import re
    
    experience_patterns = [
        r"(\d+)\+?\s*years?\s*experience",
        r"experience:\s*(\d+)",
        r"(\d+)\+?\s*years?\s*in",
    ]
    
    for pattern in experience_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 3  # Default fallback


async def _get_candidate_response(question) -> str:
    """Get candidate response for a question."""
    console.print("\n[bold yellow]Please provide your response (or type 'skip' to skip):[/bold yellow]")
    
    # In a real CLI, you'd use input() or similar
    # For now, we'll simulate with a placeholder
    response = input("Your response: ").strip()
    
    if response.lower() == "skip":
        return "Candidate chose to skip this question."
    
    return response


async def _generate_question_with_loading(orchestrator: OrchestratorAgent, session) -> Question:
    """Generate next question with continuous loading indicator."""
    
    # Create loading messages that will cycle
    loading_messages = [
        "🤖 AI is analyzing your profile and generating a personalized question...",
        "🧠 Processing your experience level and job requirements...",
        "⚡ Crafting the perfect question for your skill level...",
        "🎯 Selecting the most relevant topic for your interview...",
        "💡 Generating a challenging yet fair question...",
        "🔄 Almost ready with your next question..."
    ]
    
    # Create the loading display
    loading_display = Panel(
        Text("🤖 AI is analyzing your profile and generating a personalized question...", style="cyan"),
        title="Generating Question",
        border_style="blue"
    )
    console.clear()
    # Start the question generation task
    question_task = asyncio.create_task(orchestrator.generate_next_question(session))
    
    # Create a task for cycling loading messages
    async def cycle_loading_messages():
        message_index = 1
        while not question_task.done():
            current_message = loading_messages[message_index % len(loading_messages)]
            loading_display = Panel(
                Text(current_message, style="cyan"),
                title="Generating Question",
                border_style="blue"
            )
            live.console.clear()
            live.update(loading_display)
            await asyncio.sleep(1)
            message_index += 1
    
    # Show loading indicator while question is being generated
    with Live(loading_display, console=console, refresh_per_second=1) as live:
        # Run both tasks concurrently
        await asyncio.gather(
            question_task,
            cycle_loading_messages(),
            return_exceptions=True
        )
    
    # Get the result from the completed task
    question = await question_task
    
    # Clear the console to show only the question
    console.clear()
    
    return question


async def _evaluate_response_with_loading(orchestrator: OrchestratorAgent, session, question: Question, response: str) -> Evaluation:
    """Evaluate response with continuous loading indicator."""
    
    # Create loading messages that will cycle
    loading_messages = [
        "🤖 AI is analyzing your response and comparing it to expected answers...",
        "🧠 Evaluating technical accuracy and problem-solving approach...",
        "📊 Calculating scores across multiple evaluation criteria...",
        "💭 Generating detailed feedback and improvement suggestions...",
        "🎯 Assessing communication clarity and code quality...",
        "📝 Preparing comprehensive evaluation report..."
    ]
    
    # Create the loading display
    loading_display = Panel(
        Text("🤖 AI is analyzing your response and comparing it to expected answers...", style="cyan"),
        title="Evaluating Response",
        border_style="blue"
    )
    console.clear()
    # Start the evaluation task
    evaluation_task = asyncio.create_task(orchestrator.evaluate_response(session, question, response))
    
    # Create a task for cycling loading messages
    async def cycle_loading_messages():
        message_index = 1
        while not evaluation_task.done():
            current_message = loading_messages[message_index % len(loading_messages)]
            loading_display = Panel(
                Text(current_message, style="cyan"),
                title="Evaluating Response",
                border_style="blue"
            )
            live.console.clear()
            live.update(loading_display)
            await asyncio.sleep(1)
            message_index += 1
    
    # Show loading indicator while evaluation is being processed
    with Live(loading_display, console=console, refresh_per_second=1) as live:
        # Run both tasks concurrently
        await asyncio.gather(
            evaluation_task,
            cycle_loading_messages(),
            return_exceptions=True
        )
    
    # Get the result from the completed task
    evaluation = await evaluation_task
    
    # Clear the console to show only the evaluation
    console.clear()
    
    return evaluation


async def _ask_continue_interview() -> bool:
    """Ask if candidate wants to continue the interview."""
    console.print("\n[bold blue]Would you like to continue with the next question? (y/n):[/bold blue]")
    
    # In a real CLI, you'd use input() or similar
    # For now, we'll simulate with a placeholder
    response = input("Continue? ").strip().lower()
    
    return response in ["y", "yes", "continue"]


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interview interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logger.error(f"Unexpected CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
