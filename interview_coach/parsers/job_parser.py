"""Job description parser for extracting structured information from job postings."""

import asyncio
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from .base_parser import BaseParser, FileFormat, ParsedData
from .file_handlers import FileHandlerRegistry, TextExtractionResult
from ..models.job import JobDescription
from ..utils.logging import get_logger


@dataclass
class JobParseResult:
    """Result of job description parsing operation."""
    job_data: JobDescription
    confidence_score: float
    parse_errors: List[str]
    extraction_metadata: Dict[str, Any]


class JobParser(BaseParser):
    """Parser for extracting structured information from job descriptions."""
    
    def __init__(self):
        """Initialize the job parser."""
        super().__init__("JobParser")
        self.supported_formats = [
            FileFormat.PDF, FileFormat.DOCX, FileFormat.TXT, 
            FileFormat.JSON, FileFormat.HTML, FileFormat.RTF
        ]
        self.file_handler_registry: Optional[FileHandlerRegistry] = None
        
        # Common patterns for job parsing
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Z][A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'website': r'https?://(?:www\.)?[\w-]+\.[a-z]{2,}(?:/[\w-]*)*',
            'salary': r'\$[\d,]+(?:-\$[\d,]+)?(?:\s*(?:per\s+)?(?:year|month|hour|week))?',
            'experience_years': r'\b(\d+)\s*(?:to\s*(\d+))?\s*(?:years?|yrs?)\b',
            'location': r'(?:in|at|based\s+in)\s+([A-Z][a-z]+(?:[\s,]+[A-Z]{2})?(?:[\s,]+[A-Z][a-z]+)*)',
            'remote': r'\b(?:remote|work\s+from\s+home|telecommute|hybrid|on-site|in-office)\b',
            'job_type': r'\b(?:full-time|part-time|contract|temporary|internship|freelance)\b',
            'degree': r'\b(?:Bachelor|Master|PhD|BSc|MSc|MBA|Associate|Diploma|Certificate)\b',
            'date': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'
        }
        
        # Common job requirement keywords
        self.requirement_keywords = {
            'technical_skills': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'sql', 'nosql', 'aws', 'azure', 'docker', 'kubernetes', 'git', 'agile',
                'scrum', 'machine learning', 'ai', 'data science', 'devops', 'ci/cd'
            ],
            'soft_skills': [
                'communication', 'leadership', 'teamwork', 'problem solving', 'analytical',
                'creativity', 'adaptability', 'time management', 'collaboration'
            ],
            'tools': [
                'jira', 'confluence', 'slack', 'teams', 'postman', 'jenkins', 'gitlab',
                'github', 'bitbucket', 'trello', 'asana', 'monday.com'
            ]
        }
    
    def _load_resources(self) -> None:
        """Load parser resources."""
        self.file_handler_registry = FileHandlerRegistry()
        self.file_handler_registry.initialize()
    
    async def parse_file(self, file_path: Union[str, Path]) -> ParsedData:
        """Parse a job description file and return structured data.
        
        Args:
            file_path: Path to the job description file
            
        Returns:
            ParsedData object containing the parsed job information
        """
        try:
            # Validate file
            if not await self.validate_file(file_path):
                raise ValueError(f"File {file_path} is not valid for parsing")
            
            # Extract text from file
            text_result = await self._extract_text_from_file(file_path)
            
            if not text_result.text:
                raise ValueError("No text could be extracted from the file")
            
            # Parse the extracted text
            parse_result = await self.parse_text(text_result.text, {
                "source_file": str(file_path),
                "extraction_metadata": text_result.metadata
            })
            
            # Create ParsedData object
            return ParsedData(
                raw_text=text_result.text,
                structured_data=parse_result.job_data.model_dump(),
                confidence_score=parse_result.confidence_score * text_result.confidence_score,
                parse_errors=parse_result.parse_errors + text_result.extraction_errors,
                metadata={
                    "source_file": str(file_path),
                    "file_format": self._detect_file_format(Path(file_path)),
                    "extraction_metadata": text_result.metadata,
                    "parse_metadata": parse_result.extraction_metadata
                },
                source_file=str(file_path),
                file_format=self._detect_file_format(Path(file_path))
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse job description file {file_path}: {e}")
            raise
    
    async def parse_text(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> JobParseResult:
        """Parse job description text and extract structured information.
        
        Args:
            text: Raw job description text to parse
            source_info: Optional information about the text source
            
        Returns:
            JobParseResult containing parsed job data
        """
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Extract different sections
            sections = self._extract_sections(cleaned_text)
            
            # Parse each section
            company_info = self._parse_company_info(cleaned_text, sections)
            job_details = self._parse_job_details(cleaned_text, sections)
            requirements = self._parse_requirements(cleaned_text, sections)
            responsibilities = self._parse_responsibilities(cleaned_text, sections)
            benefits = self._parse_benefits(cleaned_text, sections)
            
            # Create JobDescription object
            job_data = JobDescription(
                title=job_details.get('title', 'Unknown Position'),
                company=company_info.get('name', 'Unknown Company'),
                location=job_details.get('location', 'Unknown Location'),
                job_type=job_details.get('job_type', 'Full-time'),
                remote_option=job_details.get('remote_option', 'On-site'),
                salary_range=job_details.get('salary_range'),
                experience_level=requirements.get('experience_level', 'Entry'),
                required_skills=requirements.get('required_skills', []),
                preferred_skills=requirements.get('preferred_skills', []),
                responsibilities=responsibilities,
                requirements=requirements.get('requirements', []),
                benefits=benefits,
                company_description=company_info.get('description', ''),
                application_deadline=job_details.get('deadline'),
                contact_info=company_info.get('contact', {}),
                website=company_info.get('website')
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                job_details, requirements, responsibilities, company_info
            )
            
            # Collect parse errors
            parse_errors = self._collect_parse_errors(
                job_details, requirements, responsibilities, company_info
            )
            
            return JobParseResult(
                job_data=job_data,
                confidence_score=confidence_score,
                parse_errors=parse_errors,
                extraction_metadata={
                    "sections_found": list(sections.keys()),
                    "text_length": len(cleaned_text),
                    "parse_timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse job description text: {e}")
            raise
    
    async def _extract_text_from_file(self, file_path: Union[str, Path]) -> TextExtractionResult:
        """Extract text from a file using appropriate handler.
        
        Args:
            file_path: Path to the file
            
        Returns:
            TextExtractionResult with extracted text
        """
        if not self.file_handler_registry:
            raise RuntimeError("File handler registry not initialized")
        
        file_format = self._detect_file_format(Path(file_path))
        handler = self.file_handler_registry.get_handler(file_format)
        
        if not handler:
            raise ValueError(f"No handler available for file format: {file_format}")
        
        return await handler.extract_text(file_path)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize job description text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with parsing
        text = re.sub(r'[^\w\s@.-]', ' ', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from job description text.
        
        Args:
            text: Cleaned job description text
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        # Common section headers
        section_patterns = [
            r'(?i)(?:about\s+us|company|organization)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)',
            r'(?i)(?:job\s+description|position|role)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)',
            r'(?i)(?:requirements|qualifications|skills)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)',
            r'(?i)(?:responsibilities|duties|key\s+responsibilities)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)',
            r'(?i)(?:benefits|perks|compensation)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)',
            r'(?i)(?:how\s+to\s+apply|application|contact)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                section_name = re.search(r'\(?i\)\?\(([^)]+)\)', pattern).group(1).lower()
                sections[section_name] = matches[0].strip()
        
        return sections
    
    def _parse_company_info(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Parse company information from job description text.
        
        Args:
            text: Job description text to parse
            sections: Extracted sections
            
        Returns:
            Dictionary containing company information
        """
        company_info = {}
        
        # Get company section content
        company_text = sections.get('about us', text)
        
        # Extract company name (usually at the beginning or in title)
        company_name_patterns = [
            r'([A-Z][A-Z\s&]+)\s+is\s+(?:looking|seeking|hiring)',
            r'(?:Join|Work\s+at|Position\s+at)\s+([A-Z][A-Z\s&]+)',
            r'([A-Z][A-Z\s&]+)\s+(?:Job|Position|Opening)'
        ]
        
        for pattern in company_name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company_info['name'] = match.group(1).strip()
                break
        
        if 'name' not in company_info:
            # Fallback: look for company-like patterns
            company_match = re.search(r'([A-Z][A-Z\s&]{3,})', text)
            if company_match:
                company_info['name'] = company_match.group(1).strip()
        
        # Extract website
        website_match = re.search(self.patterns['website'], text)
        if website_match:
            company_info['website'] = website_match.group(0)
        
        # Extract contact information
        email_match = re.search(self.patterns['email'], text)
        if email_match:
            company_info['contact'] = {'email': email_match.group(0)}
        
        phone_match = re.search(self.patterns['phone'], text)
        if phone_match:
            if 'contact' not in company_info:
                company_info['contact'] = {}
            company_info['contact']['phone'] = ''.join(phone_match.groups())
        
        # Extract company description
        if company_text and company_text != text:
            company_info['description'] = company_text.strip()
        
        return company_info
    
    def _parse_job_details(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Parse job details from job description text.
        
        Args:
            text: Job description text to parse
            sections: Extracted sections
            
        Returns:
            Dictionary containing job details
        """
        job_details = {}
        
        # Extract job title
        title_patterns = [
            r'(?:We\s+are\s+looking\s+for\s+a\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Position:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:Job\s+Title:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Developer|Engineer|Manager|Analyst|Designer|Consultant)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                job_details['title'] = match.group(1).strip()
                break
        
        if 'title' not in job_details:
            # Look for common job titles
            common_titles = ['Developer', 'Engineer', 'Manager', 'Analyst', 'Designer', 'Consultant']
            for title in common_titles:
                if re.search(rf'\b{title}\b', text, re.IGNORECASE):
                    job_details['title'] = title
                    break
        
        # Extract location
        location_match = re.search(self.patterns['location'], text, re.IGNORECASE)
        if location_match:
            job_details['location'] = location_match.group(1).strip()
        
        # Extract remote option
        remote_match = re.search(self.patterns['remote'], text, re.IGNORECASE)
        if remote_match:
            remote_text = remote_match.group(0).lower()
            if 'remote' in remote_text:
                job_details['remote_option'] = 'Remote'
            elif 'hybrid' in remote_text:
                job_details['remote_option'] = 'Hybrid'
            else:
                job_details['remote_option'] = 'On-site'
        
        # Extract job type
        job_type_match = re.search(self.patterns['job_type'], text, re.IGNORECASE)
        if job_type_match:
            job_details['job_type'] = job_type_match.group(0).title()
        
        # Extract salary range
        salary_match = re.search(self.patterns['salary'], text)
        if salary_match:
            job_details['salary_range'] = salary_match.group(0)
        
        # Extract application deadline
        deadline_patterns = [
            r'(?:Apply\s+by|Deadline|Closing\s+date)\s*:?\s*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:Applications\s+close|Position\s+closes)\s*:?\s*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})'
        ]
        
        for pattern in deadline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                job_details['deadline'] = match.group(1).strip()
                break
        
        return job_details
    
    def _parse_requirements(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Parse job requirements from job description text.
        
        Args:
            text: Job description text to parse
            sections: Extracted sections
            
        Returns:
            Dictionary containing job requirements
        """
        requirements = {}
        
        # Get requirements section content
        req_text = sections.get('requirements', text)
        
        # Extract experience level
        exp_match = re.search(self.patterns['experience_years'], text, re.IGNORECASE)
        if exp_match:
            min_years = int(exp_match.group(1))
            max_years = int(exp_match.group(2)) if exp_match.group(2) else min_years
            
            if min_years <= 2:
                requirements['experience_level'] = 'Entry'
            elif min_years <= 5:
                requirements['experience_level'] = 'Mid-level'
            elif min_years <= 10:
                requirements['experience_level'] = 'Senior'
            else:
                requirements['experience_level'] = 'Expert'
        else:
            requirements['experience_level'] = 'Entry'
        
        # Extract required skills
        required_skills = []
        preferred_skills = []
        
        # Look for required vs preferred skills
        required_patterns = [
            r'(?:Required|Must\s+have|Essential)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)',
            r'(?:Requirements|Qualifications)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)'
        ]
        
        preferred_patterns = [
            r'(?:Preferred|Nice\s+to\s+have|Bonus)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)',
            r'(?:Plus|Additional|Desired)\s*:?\s*(.*?)(?=\n\s*[A-Z][A-Z\s]*:|\Z)'
        ]
        
        # Extract required skills
        for pattern in required_patterns:
            matches = re.findall(pattern, req_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                required_skills.extend(skills)
        
        # Extract preferred skills
        for pattern in preferred_patterns:
            matches = re.findall(pattern, req_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                preferred_skills.extend(skills)
        
        # If no specific required/preferred sections, extract all skills
        if not required_skills and not preferred_skills:
            all_skills = self._extract_skills_from_text(req_text)
            required_skills = all_skills
        
        requirements['required_skills'] = list(set(required_skills))
        requirements['preferred_skills'] = list(set(preferred_skills))
        
        # Extract general requirements
        general_reqs = []
        req_lines = req_text.split('\n')
        
        for line in req_lines:
            line = line.strip()
            if line and not any(skill.lower() in line.lower() for skill in required_skills + preferred_skills):
                if len(line) > 10:  # Filter out very short lines
                    general_reqs.append(line)
        
        requirements['requirements'] = general_reqs
        
        return requirements
    
    def _parse_responsibilities(self, text: str, sections: Dict[str, str]) -> List[str]:
        """Parse job responsibilities from job description text.
        
        Args:
            text: Job description text to parse
            sections: Extracted sections
            
        Returns:
            List of responsibilities
        """
        responsibilities = []
        
        # Get responsibilities section content
        resp_text = sections.get('responsibilities', text)
        
        # Split into individual responsibilities
        resp_patterns = [
            r'•\s*(.*?)(?=\n\s*•|\Z)',
            r'-\s*(.*?)(?=\n\s*-|\Z)',
            r'\d+\.\s*(.*?)(?=\n\s*\d+\.|\Z)',
            r'([A-Z][^.\n]+\.)'
        ]
        
        for pattern in resp_patterns:
            matches = re.findall(pattern, resp_text, re.DOTALL)
            for match in matches:
                responsibility = match.strip()
                if responsibility and len(responsibility) > 10:
                    responsibilities.append(responsibility)
        
        # If no structured format found, split by sentences
        if not responsibilities:
            sentences = re.split(r'[.!?]+', resp_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 20:
                    responsibilities.append(sentence)
        
        return responsibilities[:20]  # Limit to 20 responsibilities
    
    def _parse_benefits(self, text: str, sections: Dict[str, str]) -> List[str]:
        """Parse job benefits from job description text.
        
        Args:
            text: Job description text to parse
            sections: Extracted sections
            
        Returns:
            List of benefits
        """
        benefits = []
        
        # Get benefits section content
        benefits_text = sections.get('benefits', text)
        
        # Common benefit keywords
        benefit_keywords = [
            'health insurance', 'dental insurance', 'vision insurance', '401k', 'retirement',
            'paid time off', 'vacation', 'sick leave', 'flexible hours', 'remote work',
            'professional development', 'training', 'education', 'gym membership',
            'free lunch', 'snacks', 'coffee', 'parking', 'transportation', 'bonus',
            'stock options', 'equity', 'profit sharing', 'life insurance', 'disability'
        ]
        
        # Extract benefits by keywords
        for keyword in benefit_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', benefits_text, re.IGNORECASE):
                benefits.append(keyword.title())
        
        # Extract structured benefits
        benefit_patterns = [
            r'•\s*(.*?)(?=\n\s*•|\Z)',
            r'-\s*(.*?)(?=\n\s*-|\Z)',
            r'\d+\.\s*(.*?)(?=\n\s*\d+\.|\Z)'
        ]
        
        for pattern in benefit_patterns:
            matches = re.findall(pattern, benefits_text, re.DOTALL)
            for match in matches:
                benefit = match.strip()
                if benefit and len(benefit) > 5:
                    benefits.append(benefit)
        
        return list(set(benefits))[:15]  # Limit to 15 benefits
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using keyword matching.
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List of extracted skills
        """
        skills = []
        
        for category, keywords in self.requirement_keywords.items():
            for keyword in keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', text, re.IGNORECASE):
                    skills.append(keyword.title())
        
        return skills
    
    def _calculate_confidence_score(self, job_details: Dict[str, Any], 
                                  requirements: Dict[str, Any], 
                                  responsibilities: List[str], 
                                  company_info: Dict[str, Any]) -> float:
        """Calculate confidence score for the parsing result.
        
        Args:
            job_details: Parsed job details
            requirements: Parsed requirements
            responsibilities: Parsed responsibilities
            company_info: Parsed company information
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        total_weight = 0.0
        
        # Job details weight: 25%
        details_score = 0.0
        if job_details.get('title'):
            details_score += 0.5
        if job_details.get('location'):
            details_score += 0.3
        if job_details.get('job_type'):
            details_score += 0.2
        
        score += details_score * 0.25
        total_weight += 0.25
        
        # Requirements weight: 30%
        req_score = 0.0
        if requirements.get('required_skills'):
            req_score += 0.6
        if requirements.get('experience_level'):
            req_score += 0.4
        
        score += req_score * 0.30
        total_weight += 0.30
        
        # Responsibilities weight: 25%
        resp_score = min(len(responsibilities) / 5, 1.0)  # At least 5 responsibilities
        score += resp_score * 0.25
        total_weight += 0.25
        
        # Company info weight: 20%
        company_score = 0.0
        if company_info.get('name'):
            company_score += 0.6
        if company_info.get('website') or company_info.get('contact'):
            company_score += 0.4
        
        score += company_score * 0.20
        total_weight += 0.20
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _collect_parse_errors(self, job_details: Dict[str, Any], 
                             requirements: Dict[str, Any], 
                             responsibilities: List[str], 
                             company_info: Dict[str, Any]) -> List[str]:
        """Collect parsing errors and warnings.
        
        Args:
            job_details: Parsed job details
            requirements: Parsed requirements
            responsibilities: Parsed responsibilities
            company_info: Parsed company information
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Check for missing critical information
        if not job_details.get('title'):
            errors.append("No job title found")
        
        if not company_info.get('name'):
            errors.append("No company name found")
        
        if not job_details.get('location'):
            errors.append("No location information found")
        
        if len(responsibilities) < 3:
            errors.append("Limited responsibilities information extracted")
        
        if not requirements.get('required_skills'):
            errors.append("No required skills found")
        
        return errors
    
    async def _cleanup_resources(self) -> None:
        """Clean up parser resources."""
        if self.file_handler_registry:
            await self.file_handler_registry.cleanup()
