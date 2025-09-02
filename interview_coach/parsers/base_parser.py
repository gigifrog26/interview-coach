"""Base parser interface for resume and job description parsing."""

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger


class FileFormat(Enum):
    """Supported file formats for parsing."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
    HTML = "html"
    RTF = "rtf"


class ParseResult(Enum):
    """Result status of parsing operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    UNSUPPORTED_FORMAT = "unsupported_format"


@dataclass
class ParsedData:
    """Base class for parsed data."""
    raw_text: str
    structured_data: Dict[str, Any]
    confidence_score: float
    parse_errors: List[str]
    metadata: Dict[str, Any]
    source_file: str
    file_format: FileFormat


class BaseParser(ABC):
    """Abstract base class for all parsers."""
    
    def __init__(self, name: str):
        """Initialize the base parser.
        
        Args:
            name: Name of the parser
        """
        self.name = name
        self.logger = get_logger(f"parser.{name}")
        self.supported_formats: List[FileFormat] = []
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the parser and load any required resources."""
        if self._initialized:
            return
        
        try:
            self._load_resources()
            self._initialized = True
            self.logger.info(f"{self.name} parser initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name} parser: {e}")
            raise
    
    @abstractmethod
    def _load_resources(self) -> None:
        """Load parser-specific resources."""
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Union[str, Path]) -> ParsedData:
        """Parse a file and return structured data.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            ParsedData object containing the parsed information
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
            ParseError: If parsing fails
        """
        pass
    
    @abstractmethod
    def parse_text(self, text: str, source_info: Optional[Dict[str, Any]] = None) -> ParsedData:
        """Parse raw text and return structured data.
        
        Args:
            text: Raw text to parse
            source_info: Optional information about the text source
            
        Returns:
            ParsedData object containing the parsed information
        """
        pass
    
    def supports_format(self, file_format: FileFormat) -> bool:
        """Check if the parser supports a specific file format.
        
        Args:
            file_format: File format to check
            
        Returns:
            True if format is supported, False otherwise
        """
        return file_format in self.supported_formats
    
    def get_supported_formats(self) -> List[FileFormat]:
        """Get list of supported file formats.
        
        Returns:
            List of supported FileFormat enums
        """
        return self.supported_formats.copy()
    
    async def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate that a file can be parsed by this parser.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            # Check file extension
            file_format = self._detect_file_format(path)
            if not self.supports_format(file_format):
                return False
            
            # Check file size (reasonable limits)
            if not await self._check_file_size(path):
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"File validation failed for {file_path}: {e}")
            return False
    
    def _detect_file_format(self, file_path: Path) -> FileFormat:
        """Detect file format based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected FileFormat enum
        """
        extension = file_path.suffix.lower().lstrip('.')
        
        format_mapping = {
            'pdf': FileFormat.PDF,
            'docx': FileFormat.DOCX,
            'txt': FileFormat.TXT,
            'json': FileFormat.JSON,
            'html': FileFormat.HTML,
            'htm': FileFormat.HTML,
            'rtf': FileFormat.RTF,
        }
        
        return format_mapping.get(extension, FileFormat.TXT)
    
    async def _check_file_size(self, file_path: Path) -> bool:
        """Check if file size is within acceptable limits.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file size is acceptable, False otherwise
        """
        try:
            # 50MB limit for most files
            max_size = 50 * 1024 * 1024
            
            # For text files, use smaller limit
            if file_path.suffix.lower() in ['.txt', '.json', '.html']:
                max_size = 10 * 1024 * 1024
            
            file_size = file_path.stat().st_size
            return file_size <= max_size
            
        except Exception as e:
            self.logger.warning(f"File size check failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Clean up parser resources."""
        try:
            await self._cleanup_resources()
            self._initialized = False
            self.logger.info(f"{self.name} parser cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error during {self.name} parser cleanup: {e}")
    
    @abstractmethod
    async def _cleanup_resources(self) -> None:
        """Clean up parser-specific resources."""
        pass
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the parser.
        
        Returns:
            Dictionary containing parser information
        """
        return {
            "name": self.name,
            "supported_formats": [fmt.value for fmt in self.supported_formats],
            "initialized": self._initialized,
            "class": self.__class__.__name__
        }
