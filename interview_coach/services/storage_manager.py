"""Storage Manager for handling data persistence and retrieval."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import aiofiles
import aiofiles.os

from ..models.interview import InterviewSession, InterviewSessionReport, Question, Evaluation
from ..utils.exceptions import StorageError
from ..utils.logging import get_logger


class StorageInterface:
    """Abstract interface for storage operations."""
    
    def save_session(self, session: InterviewSession) -> bool:
        """Save an interview session."""
        raise NotImplementedError
    
    def load_session(self, session_id: str) -> Optional[InterviewSession]:
        """Load an interview session by ID."""
        raise NotImplementedError
    
    def save_report(self, report: InterviewSessionReport) -> bool:
        """Save an interview session report."""
        raise NotImplementedError
    
    def load_report(self, session_id: str) -> Optional[InterviewSessionReport]:
        """Load an interview session report by session ID."""
        raise NotImplementedError
    
    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        raise NotImplementedError
    
    def delete_session(self, session_id: str) -> bool:
        """Delete an interview session."""
        raise NotImplementedError


class FileStorageManager(StorageInterface):
    """File-based storage manager using JSON files."""
    
    def __init__(self, base_path: str = "data"):
        """Initialize the file storage manager.
        
        Args:
            base_path: Base directory for storing data files.
        """
        self.base_path = Path(base_path)
        self.sessions_path = self.base_path / "sessions"
        self.reports_path = self.base_path / "reports"
        self.backup_path = self.base_path / "backups"
        self.logger = get_logger(__name__)
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        try:
            self.sessions_path.mkdir(parents=True, exist_ok=True)
            self.reports_path.mkdir(parents=True, exist_ok=True)
            self.backup_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise StorageError(f"Failed to create storage directories: {str(e)}")
    
    def initialize(self) -> None:
        """Initialize the storage manager."""
        try:
            # Verify directory permissions
            self._verify_permissions()
            
            # Create backup of existing data if needed
            self._create_initial_backup()
            
            self.logger.info("FileStorageManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FileStorageManager: {str(e)}")
            raise StorageError(f"Storage initialization failed: {str(e)}")
    
    def _verify_permissions(self) -> None:
        """Verify that the storage directories have proper permissions."""
        try:
            # Test write permission
            test_file = self.base_path / ".test_write"
            with open(test_file, "w") as f:
                f.write("test")
            
            # Test read permission
            with open(test_file, "r") as f:
                content = f.read()
                assert content == "test"
            
            # Clean up test file
            os.remove(test_file)
            
        except Exception as e:
            raise StorageError(f"Storage permission verification failed: {str(e)}")
    
    def _create_initial_backup(self) -> None:
        """Create initial backup of existing data if any."""
        try:
            existing_sessions = list(self.sessions_path.glob("*.json"))
            if existing_sessions:
                backup_dir = self.backup_path / f"initial_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_dir.mkdir(exist_ok=True)
                
                for session_file in existing_sessions:
                    backup_file = backup_dir / session_file.name
                    with open(session_file, "r", encoding='utf-8') as src, \
                               open(backup_file, "w", encoding='utf-8') as dst:
                        content = src.read()
                        dst.write(content)
                
                self.logger.info(f"Created initial backup with {len(existing_sessions)} sessions")
                
        except Exception as e:
            self.logger.warning(f"Failed to create initial backup: {str(e)}")
    
    def save_session(self, session: InterviewSession) -> bool:
        """Save an interview session to file.
        
        Args:
            session: The interview session to save.
            
        Returns:
            True if save was successful.
            
        Raises:
            StorageError: If save operation fails.
        """
        try:
            session_file = self.sessions_path / f"{session.session_id}.json"
            
            # Create backup of existing file if it exists
            if session_file.exists():
                self._backup_file(session_file)
            
            # Convert session to JSON using Pydantic's built-in JSON serialization
            session_json = session.model_dump_json(indent=2)
            
            # Save to file
            with open(session_file, "w", encoding='utf-8') as f:
                f.write(session_json)
            
            self.logger.info(f"Session {session.session_id} saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session {session.session_id}: {str(e)}")
            raise StorageError(f"Session save failed: {str(e)}")
    
    async def load_session(self, session_id: str) -> Optional[InterviewSession]:
        """Load an interview session from file.
        
        Args:
            session_id: The ID of the session to load.
            
        Returns:
            InterviewSession object if found, None otherwise.
            
        Raises:
            StorageError: If load operation fails.
        """
        try:
            session_file = self.sessions_path / f"{session_id}.json"
            
            if not session_file.exists():
                self.logger.warning(f"Session file not found: {session_id}")
                return None
            
            # Load from file
            async with aiofiles.open(session_file, "r") as f:
                content = await f.read()
            
            # Parse JSON and create session object
            session_data = json.loads(content)
            session = InterviewSession.model_validate(session_data)
            
            self.logger.info(f"Session {session_id} loaded successfully")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {str(e)}")
            raise StorageError(f"Session load failed: {str(e)}")
    
    def save_report(self, report: InterviewSessionReport) -> bool:
        """Save an interview session report to file.
        
        Args:
            report: The report to save.
            
        Returns:
            True if save was successful.
            
        Raises:
            StorageError: If save operation fails.
        """
        try:
            report_file = self.reports_path / f"{report.session_id}_report.json"
            
            # Create backup of existing file if it exists
            if report_file.exists():
                self._backup_file(report_file)
            
            # Convert report to JSON using Pydantic's built-in JSON serialization
            report_json = report.model_dump_json(indent=2)
            
            # Save to file
            with open(report_file, "w", encoding='utf-8') as f:
                f.write(report_json)
            
            self.logger.info(f"Report for session {report.session_id} saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save report for session {report.session_id}: {str(e)}")
            raise StorageError(f"Report save failed: {str(e)}")
    
    async def load_report(self, session_id: str) -> Optional[InterviewSessionReport]:
        """Load an interview session report from file.
        
        Args:
            session_id: The ID of the session to load report for.
            
        Returns:
            InterviewSessionReport object if found, None otherwise.
            
        Raises:
            StorageError: If load operation fails.
        """
        try:
            report_file = self.reports_path / f"{session_id}_report.json"
            
            if not report_file.exists():
                self.logger.warning(f"Report file not found for session: {session_id}")
                return None
            
            # Load from file
            async with aiofiles.open(report_file, "r") as f:
                content = await f.read()
            
            # Parse JSON and create report object
            report_data = json.loads(content)
            report = InterviewSessionReport.model_validate(report_data)
            
            self.logger.info(f"Report for session {session_id} loaded successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to load report for session {session_id}: {str(e)}")
            raise StorageError(f"Report load failed: {str(e)}")
    
    async def list_sessions(self) -> List[str]:
        """List all available session IDs.
        
        Returns:
            List of session IDs.
            
        Raises:
            StorageError: If list operation fails.
        """
        try:
            session_files = list(self.sessions_path.glob("*.json"))
            session_ids = [f.stem for f in session_files]
            
            self.logger.info(f"Found {len(session_ids)} sessions")
            return session_ids
            
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {str(e)}")
            raise StorageError(f"Session listing failed: {str(e)}")
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete an interview session and its associated report.
        
        Args:
            session_id: The ID of the session to delete.
            
        Returns:
            True if deletion was successful.
            
        Raises:
            StorageError: If deletion fails.
        """
        try:
            session_file = self.sessions_path / f"{session_id}.json"
            report_file = self.reports_path / f"{session_id}_report.json"
            
            # Create backup before deletion
            if session_file.exists():
                await self._backup_file(session_file)
            if report_file.exists():
                await self._backup_file(report_file)
            
            # Delete files
            deleted = False
            if session_file.exists():
                await aiofiles.os.remove(session_file)
                deleted = True
            
            if report_file.exists():
                await aiofiles.os.remove(report_file)
                deleted = True
            
            if deleted:
                self.logger.info(f"Session {session_id} deleted successfully")
            else:
                self.logger.warning(f"No files found for session {session_id}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {str(e)}")
            raise StorageError(f"Session deletion failed: {str(e)}")
    
    def _backup_file(self, file_path: Path) -> None:
        """Create a backup of a file before modification.
        
        Args:
            file_path: Path to the file to backup.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backup_path / timestamp
            backup_dir.mkdir(exist_ok=True)
            
            backup_file = backup_dir / file_path.name
            with open(file_path, "r") as f:
                content = f.read()
            
            with open(backup_file, "w") as dst:
                dst.write(content)
            
            self.logger.debug(f"Created backup of {file_path.name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup of {file_path.name}: {str(e)}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics.
        """
        try:
            session_count = len(list(self.sessions_path.glob("*.json")))
            report_count = len(list(self.reports_path.glob("*.json")))
            backup_count = len(list(self.backup_path.glob("*")))
            
            # Calculate total size
            total_size = 0
            for file_path in self.sessions_path.glob("*.json"):
                if file_path.exists():
                    total_size += file_path.stat().st_size
            
            return {
                "sessions": session_count,
                "reports": report_count,
                "backups": backup_count,
                "total_size_bytes": total_size,
                "storage_path": str(self.base_path.absolute())
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {str(e)}")
            return {}
    
    async def cleanup(self) -> None:
        """Clean up storage manager resources."""
        try:
            # Clean up old backups (keep only last 10)
            await self._cleanup_old_backups()
            self.logger.info("StorageManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    async def _cleanup_old_backups(self) -> None:
        """Clean up old backup directories, keeping only the most recent ones."""
        try:
            backup_dirs = sorted(
                [d for d in self.backup_path.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True
            )
            
            # Keep only the last 10 backups
            if len(backup_dirs) > 10:
                for old_backup in backup_dirs[10:]:
                    await self._remove_directory_recursively(old_backup)
                    self.logger.info(f"Removed old backup: {old_backup.name}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old backups: {str(e)}")
    
    async def _remove_directory_recursively(self, directory: Path) -> None:
        """Remove a directory and all its contents recursively.
        
        Args:
            directory: Path to the directory to remove.
        """
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    await self._remove_directory_recursively(item)
                else:
                    await aiofiles.os.remove(item)
            
            await aiofiles.os.rmdir(directory)
            
        except Exception as e:
            self.logger.warning(f"Failed to remove directory {directory}: {str(e)}")


class StorageManager:
    """Main storage manager that provides a unified interface."""
    
    def __init__(self, storage_type: str = "file", **kwargs):
        """Initialize the storage manager.
        
        Args:
            storage_type: Type of storage to use ("file" for now).
            **kwargs: Additional configuration parameters.
        """
        self.storage_type = storage_type
        self.storage_interface: Optional[StorageInterface] = None
        
        if storage_type == "file":
            self.storage_interface = FileStorageManager(**kwargs)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    def initialize(self) -> None:
        """Initialize the storage manager."""
        if self.storage_interface:
            self.storage_interface.initialize()
    
    def save_session(self, session: InterviewSession) -> bool:
        """Save an interview session."""
        if not self.storage_interface:
            raise StorageError("Storage interface not initialized")
        return self.storage_interface.save_session(session)
    
    async def load_session(self, session_id: str) -> Optional[InterviewSession]:
        """Load an interview session by ID."""
        if not self.storage_interface:
            raise StorageError("Storage interface not initialized")
        return await self.storage_interface.load_session(session_id)
    
    def save_report(self, report: InterviewSessionReport) -> bool:
        """Save an interview session report."""
        if not self.storage_interface:
            raise StorageError("Storage interface not initialized")
        return self.storage_interface.save_report(report)
    
    async def load_report(self, session_id: str) -> Optional[InterviewSessionReport]:
        """Load an interview session report by session ID."""
        if not self.storage_interface:
            raise StorageError("Storage interface not initialized")
        return await self.storage_interface.load_report(session_id)
    
    async def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        if not self.storage_interface:
            raise StorageError("Storage interface not initialized")
        return await self.storage_interface.list_sessions()
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete an interview session."""
        if not self.storage_interface:
            raise StorageError("Storage interface not initialized")
        return await self.storage_interface.delete_session(session_id)
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.storage_interface:
            return {}
        
        if hasattr(self.storage_interface, "get_storage_stats"):
            return await self.storage_interface.get_storage_stats()
        return {}
    
    async def cleanup(self) -> None:
        """Clean up storage manager resources."""
        if self.storage_interface and hasattr(self.storage_interface, "cleanup"):
            await self.storage_interface.cleanup()
