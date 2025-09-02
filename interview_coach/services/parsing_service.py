"""Service for coordinating resume and job description parsing operations."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from ..parsers import ResumeParser, JobParser, ParsedData, FileFormat
from ..models.resume import ResumeData
from ..models.job import JobDescription
from ..utils.logging import get_logger
from ..utils.exceptions import ParsingError, FileValidationError


class ParsingService:
    """Service for coordinating resume and job description parsing operations."""
    
    def __init__(self):
        """Initialize the parsing service."""
        self.logger = get_logger("parsing_service")
        self.resume_parser: Optional[ResumeParser] = None
        self.job_parser: Optional[JobParser] = None
        self._initialized = False
        
        # Parsing statistics
        self.stats = {
            "resumes_parsed": 0,
            "jobs_parsed": 0,
            "total_files_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "average_confidence": 0.0,
            "last_parse_time": None
        }
    
    def initialize(self) -> None:
        """Initialize the parsing service and parsers."""
        try:
            self.logger.info("Initializing parsing service...")
            
            # Initialize parsers
            self.resume_parser = ResumeParser()
            self.job_parser = JobParser()
            
            self.resume_parser.initialize()
            self.job_parser.initialize()
            
            self._initialized = True
            self.logger.info("Parsing service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize parsing service: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up parsing service resources."""
        try:
            if self.resume_parser:
                await self.resume_parser.cleanup()
            if self.job_parser:
                await self.job_parser.cleanup()
            
            self._initialized = False
            self.logger.info("Parsing service cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during parsing service cleanup: {e}")
    
    def is_initialized(self) -> bool:
        """Check if the parsing service is initialized."""
        return self._initialized
    
    async def parse_resume(self, file_path: Union[str, Path]) -> Tuple[ResumeData, float]:
        """Parse a resume file and return structured data.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Tuple of (ResumeData, confidence_score)
            
        Raises:
            ParsingError: If parsing fails
            FileValidationError: If file is invalid
        """
        if not self._initialized:
            raise RuntimeError("Parsing service not initialized")
        
        try:
            self.logger.info(f"Parsing resume file: {file_path}")
            
            # Parse the file
            parsed_data = await self.resume_parser.parse_file(file_path)
            
            # Convert to ResumeData
            resume_data = ResumeData(**parsed_data.structured_data)
            
            # Update statistics
            self._update_stats("resumes_parsed", parsed_data.confidence_score)
            
            self.logger.info(f"Successfully parsed resume with confidence: {parsed_data.confidence_score:.2f}")
            
            return resume_data, parsed_data.confidence_score
            
        except Exception as e:
            self.logger.error(f"Failed to parse resume {file_path}: {e}")
            self._update_stats("failed_parses")
            raise ParsingError(f"Failed to parse resume: {e}")
    
    async def parse_job_description(self, file_path: Union[str, Path]) -> Tuple[JobDescription, float]:
        """Parse a job description file and return structured data.
        
        Args:
            file_path: Path to the job description file
            
        Returns:
            Tuple of (JobDescription, confidence_score)
            
        Raises:
            ParsingError: If parsing fails
            FileValidationError: If file is invalid
        """
        if not self._initialized:
            raise RuntimeError("Parsing service not initialized")
        
        try:
            self.logger.info(f"Parsing job description file: {file_path}")
            
            # Parse the file
            parsed_data = await self.job_parser.parse_file(file_path)
            
            # Convert to JobDescription
            job_data = JobDescription(**parsed_data.structured_data)
            
            # Update statistics
            self._update_stats("jobs_parsed", parsed_data.confidence_score)
            
            self.logger.info(f"Successfully parsed job description with confidence: {parsed_data.confidence_score:.2f}")
            
            return job_data, parsed_data.confidence_score
            
        except Exception as e:
            self.logger.error(f"Failed to parse job description {file_path}: {e}")
            self._update_stats("failed_parses")
            raise ParsingError(f"Failed to parse job description: {e}")
    
    async def parse_resume_text(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> Tuple[ResumeData, float]:
        """Parse resume text and return structured data.
        
        Args:
            text: Raw resume text to parse
            source_info: Optional information about the text source
            
        Returns:
            Tuple of (ResumeData, confidence_score)
            
        Raises:
            ParsingError: If parsing fails
        """
        if not self._initialized:
            raise RuntimeError("Parsing service not initialized")
        
        try:
            self.logger.info("Parsing resume text")
            
            # Parse the text
            parse_result = await self.resume_parser.parse_text(text, source_info)
            
            # Update statistics
            self._update_stats("resumes_parsed", parse_result.confidence_score)
            
            self.logger.info(f"Successfully parsed resume text with confidence: {parse_result.confidence_score:.2f}")
            
            return parse_result.resume_data, parse_result.confidence_score
            
        except Exception as e:
            self.logger.error(f"Failed to parse resume text: {e}")
            self._update_stats("failed_parses")
            raise ParsingError(f"Failed to parse resume text: {e}")
    
    async def parse_job_description_text(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> Tuple[JobDescription, float]:
        """Parse job description text and return structured data.
        
        Args:
            text: Raw job description text to parse
            source_info: Optional information about the text source
            
        Returns:
            Tuple of (JobDescription, confidence_score)
            
        Raises:
            ParsingError: If parsing fails
        """
        if not self._initialized:
            raise RuntimeError("Parsing service not initialized")
        
        try:
            self.logger.info("Parsing job description text")
            
            # Parse the text
            parse_result = await self.job_parser.parse_text(text, source_info)
            
            # Update statistics
            self._update_stats("jobs_parsed", parse_result.confidence_score)
            
            self.logger.info(f"Successfully parsed job description text with confidence: {parse_result.confidence_score:.2f}")
            
            return parse_result.job_data, parse_result.confidence_score
            
        except Exception as e:
            self.logger.error(f"Failed to parse job description text: {e}")
            self._update_stats("failed_parses")
            raise ParsingError(f"Failed to parse job description text: {e}")
    
    async def batch_parse_resumes(self, file_paths: List[Union[str, Path]]) -> List[Tuple[ResumeData, float, str]]:
        """Parse multiple resume files in batch.
        
        Args:
            file_paths: List of resume file paths
            
        Returns:
            List of tuples (ResumeData, confidence_score, file_path)
        """
        if not self._initialized:
            raise RuntimeError("Parsing service not initialized")
        
        results = []
        
        for file_path in file_paths:
            try:
                resume_data, confidence = await self.parse_resume(file_path)
                results.append((resume_data, confidence, str(file_path)))
            except Exception as e:
                self.logger.error(f"Failed to parse resume {file_path}: {e}")
                # Continue with other files
                continue
        
        return results
    
    async def batch_parse_job_descriptions(self, file_paths: List[Union[str, Path]]) -> List[Tuple[JobDescription, float, str]]:
        """Parse multiple job description files in batch.
        
        Args:
            file_paths: List of job description file paths
            
        Returns:
            List of tuples (JobDescription, confidence_score, file_path)
        """
        if not self._initialized:
            raise RuntimeError("Parsing service not initialized")
        
        results = []
        
        for file_path in file_paths:
            try:
                job_data, confidence = await self.parse_job_description(file_path)
                results.append((job_data, confidence, str(file_path)))
            except Exception as e:
                self.logger.error(f"Failed to parse job description {file_path}: {e}")
                # Continue with other files
                continue
        
        return results
    
    async def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate if a file can be parsed.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid for parsing, False otherwise
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                self.logger.warning(f"File does not exist: {file_path}")
                return False
            
            # Check if file is readable
            if not path.is_file():
                self.logger.warning(f"Path is not a file: {file_path}")
                return False
            
            # Check file size (max 10MB)
            if path.stat().st_size > 10 * 1024 * 1024:
                self.logger.warning(f"File too large: {file_path}")
                return False
            
            # Check file extension
            supported_extensions = [fmt.value for fmt in FileFormat]
            if path.suffix.lower() not in supported_extensions:
                self.logger.warning(f"Unsupported file format: {path.suffix}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating file {file_path}: {e}")
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats.
        
        Returns:
            List of supported file format extensions
        """
        return [fmt.value for fmt in FileFormat]
    
    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get parsing service statistics.
        
        Returns:
            Dictionary containing parsing statistics
        """
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset parsing service statistics."""
        self.stats = {
            "resumes_parsed": 0,
            "jobs_parsed": 0,
            "total_files_processed": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "average_confidence": 0.0,
            "last_parse_time": None
        }
        self.logger.info("Parsing service statistics reset")
    
    def _update_stats(self, stat_type: str, confidence_score: Optional[float] = None) -> None:
        """Update parsing service statistics.
        
        Args:
            stat_type: Type of statistic to update
            confidence_score: Optional confidence score for the parse
        """
        self.stats["total_files_processed"] += 1
        self.stats["last_parse_time"] = datetime.now().isoformat()
        
        if stat_type in ["resumes_parsed", "jobs_parsed"]:
            self.stats[stat_type] += 1
            self.stats["successful_parses"] += 1
            
            if confidence_score is not None:
                # Update average confidence
                current_avg = self.stats["average_confidence"]
                total_successful = self.stats["successful_parses"]
                self.stats["average_confidence"] = (
                    (current_avg * (total_successful - 1) + confidence_score) / total_successful
                )
        
        elif stat_type == "failed_parses":
            self.stats["failed_parses"] += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the parsing service.
        
        Returns:
            Dictionary containing health status information
        """
        health_status = {
            "service": "parsing_service",
            "status": "healthy",
            "initialized": self._initialized,
            "parsers_available": {
                "resume_parser": self.resume_parser is not None,
                "job_parser": self.job_parser is not None
            },
            "statistics": self.get_parsing_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if parsers are working
        if self._initialized:
            try:
                # Test resume parser with minimal text
                test_text = "John Doe\nSoftware Engineer\njohn@example.com"
                await self.resume_parser.parse_text(test_text)
                health_status["parser_tests"] = {"resume_parser": "working"}
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["parser_tests"] = {"resume_parser": f"error: {e}"}
        
        return health_status
