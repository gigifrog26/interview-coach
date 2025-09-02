"""Resume parser for extracting structured information from candidate resumes."""

import asyncio
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from .base_parser import BaseParser, FileFormat, ParsedData
from .file_handlers import FileHandlerRegistry, TextExtractionResult
from ..models.resume import ResumeData
from ..utils.logging import get_logger


@dataclass
class ResumeParseResult:
    """Result of resume parsing operation."""
    resume_data: ResumeData
    confidence_score: float
    parse_errors: List[str]
    extraction_metadata: Dict[str, Any]


class ResumeParser(BaseParser):
    """Parser for extracting structured information from resumes."""
    
    def __init__(self):
        """Initialize the resume parser."""
        super().__init__("ResumeParser")
        self.supported_formats = [
            FileFormat.PDF, FileFormat.DOCX, FileFormat.TXT, 
            FileFormat.JSON, FileFormat.HTML, FileFormat.RTF
        ]
        self.file_handler_registry: Optional[FileHandlerRegistry] = None
        
        # Common patterns for resume parsing
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Z][A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'website': r'https?://(?:www\.)?[\w-]+\.[a-z]{2,}(?:/[\w-]*)*',
            'linkedin': r'(?:linkedin\.com/in/|linkedin\.com/company/)[\w-]+',
            'github': r'github\.com/[\w-]+',
            'date': r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            'duration': r'\b(\d+)\s*(?:years?|yrs?|months?|mos?)\b',
            'gpa': r'\b(?:GPA|Grade\s+Point\s+Average)\s*:?\s*(\d+\.\d+)\b',
            'percentage': r'\b(\d+(?:\.\d+)?)\s*%\b'
        }
        
        # Common resume section headers
        self.section_headers = {
            'contact': [
                'contact', 'contact information', 'personal information', 'personal details'
            ],
            'summary': [
                'summary', 'objective', 'profile', 'career objective', 'professional summary'
            ],
            'education': [
                'education', 'academic background', 'academic history', 'qualifications'
            ],
            'experience': [
                'experience', 'work experience', 'employment history', 'professional experience',
                'work history', 'career history'
            ],
            'skills': [
                'skills', 'technical skills', 'competencies', 'expertise', 'capabilities'
            ],
            'projects': [
                'projects', 'key projects', 'notable projects', 'academic projects'
            ],
            'certifications': [
                'certifications', 'certificates', 'accreditations', 'licenses'
            ],
            'awards': [
                'awards', 'honors', 'achievements', 'recognition'
            ],
            'languages': [
                'languages', 'language skills', 'fluency'
            ],
            'interests': [
                'interests', 'hobbies', 'activities', 'extracurricular'
            ]
        }
        
        # Common skill categories and keywords
        self.skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'swift', 'kotlin', 'scala', 'php', 'ruby', 'perl', 'r', 'matlab'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django',
                'flask', 'spring', 'asp.net', 'jquery', 'bootstrap', 'sass', 'less'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'oracle', 'sql server', 'sqlite', 'dynamodb', 'firebase'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'google cloud', 'gcp', 'heroku', 'digitalocean',
                'linode', 'vultr', 'ibm cloud'
            ],
            'devops_tools': [
                'docker', 'kubernetes', 'jenkins', 'gitlab', 'github actions',
                'terraform', 'ansible', 'chef', 'puppet', 'prometheus', 'grafana'
            ],
            'frameworks': [
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
                'spring boot', 'laravel', 'rails', 'asp.net core'
            ]
        }
    
    def _load_resources(self) -> None:
        """Load parser resources."""
        self.file_handler_registry = FileHandlerRegistry()
        self.file_handler_registry.initialize()
    
    async def parse_file(self, file_path: Union[str, Path]) -> ParsedData:
        """Parse a resume file and return structured data.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            ParsedData object containing the parsed resume information
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
                structured_data=parse_result.resume_data.model_dump(),
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
            self.logger.error(f"Failed to parse resume file {file_path}: {e}")
            raise
    
    async def parse_text(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> ResumeParseResult:
        """Parse resume text and extract structured information.
        
        Args:
            text: Raw resume text to parse
            source_info: Optional information about the text source
            
        Returns:
            ResumeParseResult containing parsed resume data
        """
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Extract different sections
            sections = self._extract_sections(cleaned_text)
            
            # Parse each section
            contact_info = self._parse_contact_info(cleaned_text, sections)
            summary = self._parse_summary(cleaned_text, sections)
            education = self._parse_education(cleaned_text, sections)
            experience = self._parse_experience(cleaned_text, sections)
            skills = self._parse_skills(cleaned_text, sections)
            projects = self._parse_projects(cleaned_text, sections)
            certifications = self._parse_certifications(cleaned_text, sections)
            awards = self._parse_awards(cleaned_text, sections)
            languages = self._parse_languages(cleaned_text, sections)
            interests = self._parse_interests(cleaned_text, sections)
            
            # Create ResumeData object
            resume_data = ResumeData(
                name=contact_info.get('name', 'Unknown'),
                email=contact_info.get('email'),
                phone=contact_info.get('phone'),
                location=contact_info.get('location'),
                website=contact_info.get('website'),
                linkedin=contact_info.get('linkedin'),
                github=contact_info.get('github'),
                summary=summary,
                education=education,
                experience=experience,
                skills=skills,
                projects=projects,
                certifications=certifications,
                awards=awards,
                languages=languages,
                interests=interests
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                contact_info, summary, education, experience, skills, projects
            )
            
            # Collect parse errors
            parse_errors = self._collect_parse_errors(
                contact_info, summary, education, experience, skills, projects
            )
            
            return ResumeParseResult(
                resume_data=resume_data,
                confidence_score=confidence_score,
                parse_errors=parse_errors,
                extraction_metadata={
                    "sections_found": list(sections.keys()),
                    "text_length": len(cleaned_text),
                    "parse_timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse resume text: {e}")
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
        """Clean and normalize resume text.
        
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
        """Extract different sections from resume text.
        
        Args:
            text: Cleaned resume text
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        # Split text into lines for section detection
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            detected_section = self._detect_section_header(line)
            
            if detected_section:
                # Save previous section if exists
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = detected_section
                current_content = []
            elif current_section:
                # Add line to current section
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _detect_section_header(self, line: str) -> Optional[str]:
        """Detect if a line is a section header.
        
        Args:
            line: Line to check
            
        Returns:
            Section name if detected, None otherwise
        """
        line_lower = line.lower()
        
        for section, headers in self.section_headers.items():
            for header in headers:
                if header in line_lower:
                    return section
        
        # Check for common patterns
        if re.match(r'^[A-Z][A-Z\s]+$', line) and len(line) > 3:
            return 'unknown'
        
        return None
    
    def _parse_contact_info(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Parse contact information from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            Dictionary containing contact information
        """
        contact_info = {}
        
        # Get contact section content
        contact_text = sections.get('contact', text)
        
        # Extract name (usually first line or in header)
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Resume|CV|Curriculum Vitae)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Email|Phone|Address)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                contact_info['name'] = match.group(1).strip()
                break
        
        # Extract email
        email_match = re.search(self.patterns['email'], text)
        if email_match:
            contact_info['email'] = email_match.group(0)
        
        # Extract phone
        phone_match = re.search(self.patterns['phone'], text)
        if phone_match:
            contact_info['phone'] = ''.join(phone_match.groups())
        
        # Extract website
        website_match = re.search(self.patterns['website'], text)
        if website_match:
            contact_info['website'] = website_match.group(0)
        
        # Extract LinkedIn
        linkedin_match = re.search(self.patterns['linkedin'], text)
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group(0)
        
        # Extract GitHub
        github_match = re.search(self.patterns['github'], text)
        if github_match:
            contact_info['github'] = github_match.group(0)
        
        # Extract location (look for city, state patterns)
        location_patterns = [
            r'([A-Z][a-z]+(?:[\s,]+[A-Z]{2})?(?:[\s,]+[A-Z][a-z]+)*)',
            r'(?:Address|Location)\s*:?\s*([A-Z][a-z]+(?:[\s,]+[A-Z]{2})?(?:[\s,]+[A-Z][a-z]+)*)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if len(location) > 5 and not any(skill in location.lower() for skills in self.skill_categories.values() for skill in skills):
                    contact_info['location'] = location
                    break
        
        return contact_info
    
    def _parse_summary(self, text: str, sections: Dict[str, str]) -> str:
        """Parse professional summary from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            Professional summary text
        """
        summary_text = sections.get('summary', '')
        
        if not summary_text:
            # Look for summary-like content near the top
            lines = text.split('\n')[:10]  # Check first 10 lines
            for line in lines:
                line = line.strip()
                if len(line) > 50 and len(line) < 300:  # Reasonable summary length
                    if not any(header in line.lower() for headers in self.section_headers.values()):
                        summary_text = line
                        break
        
        return summary_text.strip()
    
    def _parse_education(self, text: str, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Parse education information from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            List of education entries
        """
        education = []
        education_text = sections.get('education', text)
        
        # Common education patterns
        education_patterns = [
            r'([A-Z][a-z\s&]+(?:University|College|Institute|School))\s*[-–]\s*([A-Z][a-z\s]+)\s*[-–]\s*(\d{4})',
            r'([A-Z][a-z\s&]+(?:University|College|Institute|School))\s*[-–]\s*([A-Z][a-z\s]+)\s*[-–]\s*(\d{4})\s*[-–]\s*(\d{4})',
            r'([A-Z][a-z\s&]+(?:University|College|Institute|School))\s*[-–]\s*([A-Z][a-z\s]+)\s*[-–]\s*(\d{4})\s*[-–]\s*(\d{4})\s*[-–]\s*([A-Z][a-z\s]+)'
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, education_text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 3:
                    edu_entry = {
                        'institution': match[0].strip(),
                        'degree': match[1].strip(),
                        'graduation_year': match[2].strip()
                    }
                    
                    if len(match) >= 4:
                        edu_entry['start_year'] = match[3].strip()
                    
                    if len(match) >= 5:
                        edu_entry['gpa'] = match[4].strip()
                    
                    education.append(edu_entry)
        
        # If no structured format found, try to extract by lines
        if not education:
            lines = education_text.split('\n')
            current_edu = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for institution names
                if any(keyword in line.lower() for keyword in ['university', 'college', 'institute', 'school']):
                    if current_edu:
                        education.append(current_edu)
                    current_edu = {'institution': line}
                elif current_edu and 'degree' not in current_edu:
                    current_edu['degree'] = line
                elif current_edu and 'graduation_year' not in current_edu:
                    year_match = re.search(r'\d{4}', line)
                    if year_match:
                        current_edu['graduation_year'] = year_match.group(0)
            
            if current_edu:
                education.append(current_edu)
        
        return education[:5]  # Limit to 5 education entries
    
    def _parse_experience(self, text: str, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Parse work experience from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            List of work experience entries
        """
        experience = []
        experience_text = sections.get('experience', text)
        
        # Common experience patterns
        experience_patterns = [
            r'([A-Z][a-z\s&]+(?:Inc|Corp|LLC|Ltd|Company|Tech|Solutions))\s*[-–]\s*([A-Z][a-z\s]+)\s*[-–]\s*(\d{4})\s*[-–]\s*(\d{4}|Present)',
            r'([A-Z][a-z\s&]+(?:Inc|Corp|LLC|Ltd|Company|Tech|Solutions))\s*[-–]\s*([A-Z][a-z\s]+)\s*[-–]\s*(\d{4})\s*[-–]\s*(\d{4}|Present)\s*[-–]\s*([A-Z][a-z\s]+)'
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, experience_text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 4:
                    exp_entry = {
                        'company': match[0].strip(),
                        'position': match[1].strip(),
                        'start_date': match[2].strip(),
                        'end_date': match[3].strip()
                    }
                    
                    if len(match) >= 5:
                        exp_entry['location'] = match[4].strip()
                    
                    experience.append(exp_entry)
        
        # If no structured format found, try to extract by lines
        if not experience:
            lines = experience_text.split('\n')
            current_exp = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for company names
                if any(keyword in line.lower() for keyword in ['inc', 'corp', 'llc', 'ltd', 'company', 'tech', 'solutions']):
                    if current_exp:
                        experience.append(current_exp)
                    current_exp = {'company': line}
                elif current_exp and 'position' not in current_exp:
                    current_exp['position'] = line
                elif current_exp and 'start_date' not in current_exp:
                    date_match = re.search(r'\d{4}', line)
                    if date_match:
                        current_exp['start_date'] = date_match.group(0)
            
            if current_exp:
                experience.append(current_exp)
        
        return experience[:10]  # Limit to 10 experience entries
    
    def _parse_skills(self, text: str, sections: Dict[str, str]) -> Dict[str, List[str]]:
        """Parse skills from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            Dictionary mapping skill categories to skill lists
        """
        skills = {}
        skills_text = sections.get('skills', text)
        
        # Initialize skill categories
        for category in self.skill_categories.keys():
            skills[category] = []
        
        # Extract skills by category
        for category, keywords in self.skill_categories.items():
            for keyword in keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', skills_text, re.IGNORECASE):
                    skills[category].append(keyword.title())
        
        # Look for additional skills in structured format
        skill_patterns = [
            r'•\s*(.*?)(?=\n\s*•|\Z)',
            r'-\s*(.*?)(?=\n\s*-|\Z)',
            r'\d+\.\s*(.*?)(?=\n\s*\d+\.|\Z)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, skills_text, re.DOTALL)
            for match in matches:
                skill = match.strip()
                if skill and len(skill) > 2:
                    # Try to categorize the skill
                    categorized = False
                    for category, keywords in self.skill_categories.items():
                        if any(keyword.lower() in skill.lower() for keyword in keywords):
                            if skill not in skills[category]:
                                skills[category].append(skill)
                            categorized = True
                            break
                    
                    # If not categorized, add to general skills
                    if not categorized:
                        if 'general' not in skills:
                            skills['general'] = []
                        skills['general'].append(skill)
        
        return skills
    
    def _parse_projects(self, text: str, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Parse projects from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            List of project entries
        """
        projects = []
        projects_text = sections.get('projects', text)
        
        # Common project patterns
        project_patterns = [
            r'([A-Z][a-z\s]+)\s*[-–]\s*(.*?)(?=\n\s*[A-Z][a-z\s]+[-–]|\Z)',
            r'([A-Z][a-z\s]+)\s*:?\s*(.*?)(?=\n\s*[A-Z][a-z\s]+:|\Z)'
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, projects_text, re.DOTALL)
            for match in matches:
                if len(match) >= 2:
                    project_entry = {
                        'name': match[0].strip(),
                        'description': match[1].strip()
                    }
                    projects.append(project_entry)
        
        # If no structured format found, try to extract by lines
        if not projects:
            lines = projects_text.split('\n')
            current_project = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for project names (usually start with capital letters)
                if re.match(r'^[A-Z][a-z\s]+$', line) and len(line) > 3:
                    if current_project:
                        projects.append(current_project)
                    current_project = {'name': line}
                elif current_project and 'description' not in current_project:
                    current_project['description'] = line
            
            if current_project:
                projects.append(current_project)
        
        return projects[:10]  # Limit to 10 projects
    
    def _parse_certifications(self, text: str, sections: Dict[str, str]) -> List[str]:
        """Parse certifications from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            List of certifications
        """
        certifications = []
        cert_text = sections.get('certifications', text)
        
        # Extract certifications by patterns
        cert_patterns = [
            r'•\s*(.*?)(?=\n\s*•|\Z)',
            r'-\s*(.*?)(?=\n\s*-|\Z)',
            r'\d+\.\s*(.*?)(?=\n\s*\d+\.|\Z)'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, cert_text, re.DOTALL)
            for match in matches:
                cert = match.strip()
                if cert and len(cert) > 5:
                    certifications.append(cert)
        
        return certifications[:10]  # Limit to 10 certifications
    
    def _parse_awards(self, text: str, sections: Dict[str, str]) -> List[str]:
        """Parse awards from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            List of awards
        """
        awards = []
        awards_text = sections.get('awards', text)
        
        # Extract awards by patterns
        award_patterns = [
            r'•\s*(.*?)(?=\n\s*•|\Z)',
            r'-\s*(.*?)(?=\n\s*-|\Z)',
            r'\d+\.\s*(.*?)(?=\n\s*\d+\.|\Z)'
        ]
        
        for pattern in award_patterns:
            matches = re.findall(pattern, awards_text, re.DOTALL)
            for match in matches:
                award = match.strip()
                if award and len(award) > 5:
                    awards.append(award)
        
        return awards[:10]  # Limit to 10 awards
    
    def _parse_languages(self, text: str, sections: Dict[str, str]) -> List[str]:
        """Parse languages from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            List of languages
        """
        languages = []
        languages_text = sections.get('languages', text)
        
        # Common language keywords
        language_keywords = [
            'english', 'spanish', 'french', 'german', 'italian', 'portuguese',
            'chinese', 'japanese', 'korean', 'russian', 'arabic', 'hindi'
        ]
        
        # Extract languages by keywords
        for keyword in language_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', languages_text, re.IGNORECASE):
                languages.append(keyword.title())
        
        # Extract languages by patterns
        language_patterns = [
            r'•\s*(.*?)(?=\n\s*•|\Z)',
            r'-\s*(.*?)(?=\n\s*-|\Z)',
            r'\d+\.\s*(.*?)(?=\n\s*\d+\.|\Z)'
        ]
        
        for pattern in language_patterns:
            matches = re.findall(pattern, languages_text, re.DOTALL)
            for match in matches:
                language = match.strip()
                if language and len(language) > 2:
                    if language.lower() not in [lang.lower() for lang in languages]:
                        languages.append(language)
        
        return languages[:5]  # Limit to 5 languages
    
    def _parse_interests(self, text: str, sections: Dict[str, str]) -> List[str]:
        """Parse interests from resume text.
        
        Args:
            text: Resume text to parse
            sections: Extracted sections
            
        Returns:
            List of interests
        """
        interests = []
        interests_text = sections.get('interests', text)
        
        # Extract interests by patterns
        interest_patterns = [
            r'•\s*(.*?)(?=\n\s*•|\Z)',
            r'-\s*(.*?)(?=\n\s*-|\Z)',
            r'\d+\.\s*(.*?)(?=\n\s*\d+\.|\Z)'
        ]
        
        for pattern in interest_patterns:
            matches = re.findall(pattern, interests_text, re.DOTALL)
            for match in matches:
                interest = match.strip()
                if interest and len(interest) > 3:
                    interests.append(interest)
        
        return interests[:10]  # Limit to 10 interests
    
    def _calculate_confidence_score(self, contact_info: Dict[str, Any], 
                                  summary: str, education: List[Dict[str, Any]], 
                                  experience: List[Dict[str, Any]], 
                                  skills: Dict[str, List[str]], 
                                  projects: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the parsing result.
        
        Args:
            contact_info: Parsed contact information
            summary: Parsed summary
            education: Parsed education
            experience: Parsed experience
            skills: Parsed skills
            projects: Parsed projects
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        total_weight = 0.0
        
        # Contact info weight: 20%
        contact_score = 0.0
        if contact_info.get('name'):
            contact_score += 0.4
        if contact_info.get('email'):
            contact_score += 0.3
        if contact_info.get('phone'):
            contact_score += 0.2
        if contact_info.get('location'):
            contact_score += 0.1
        
        score += contact_score * 0.20
        total_weight += 0.20
        
        # Summary weight: 15%
        summary_score = 1.0 if summary and len(summary) > 20 else 0.0
        score += summary_score * 0.15
        total_weight += 0.15
        
        # Education weight: 20%
        education_score = min(len(education) / 2, 1.0)  # At least 2 education entries
        score += education_score * 0.20
        total_weight += 0.20
        
        # Experience weight: 25%
        experience_score = min(len(experience) / 3, 1.0)  # At least 3 experience entries
        score += experience_score * 0.25
        total_weight += 0.25
        
        # Skills weight: 15%
        skills_score = 0.0
        total_skills = sum(len(skill_list) for skill_list in skills.values())
        if total_skills >= 5:
            skills_score = 1.0
        elif total_skills >= 3:
            skills_score = 0.7
        elif total_skills >= 1:
            skills_score = 0.4
        
        score += skills_score * 0.15
        total_weight += 0.15
        
        # Projects weight: 5%
        projects_score = min(len(projects) / 3, 1.0)  # At least 3 projects
        score += projects_score * 0.05
        total_weight += 0.05
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _collect_parse_errors(self, contact_info: Dict[str, Any], 
                             summary: str, education: List[Dict[str, Any]], 
                             experience: List[Dict[str, Any]], 
                             skills: Dict[str, List[str]], 
                             projects: List[Dict[str, Any]]) -> List[str]:
        """Collect parsing errors and warnings.
        
        Args:
            contact_info: Parsed contact information
            summary: Parsed summary
            education: Parsed education
            experience: Parsed experience
            skills: Parsed skills
            projects: Parsed projects
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Check for missing critical information
        if not contact_info.get('name'):
            errors.append("No name found")
        
        if not contact_info.get('email'):
            errors.append("No email found")
        
        if not summary:
            errors.append("No professional summary found")
        
        if len(education) < 1:
            errors.append("No education information found")
        
        if len(experience) < 1:
            errors.append("No work experience found")
        
        total_skills = sum(len(skill_list) for skill_list in skills.values())
        if total_skills < 3:
            errors.append("Limited skills information extracted")
        
        return errors
    
    async def _cleanup_resources(self) -> None:
        """Clean up parser resources."""
        if self.file_handler_registry:
            await self.file_handler_registry.cleanup()
