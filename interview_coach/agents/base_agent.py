"""Base agent interface for the Interview Coach System."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import uuid
from contextvars import ContextVar

from ..utils.logging import get_logger

# Correlation ID context for request tracing
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class BaseAgent(ABC):
    """Base interface for all agents."""

    def __init__(self, agent_name: str):
        """Initialize the base agent.
        
        Args:
            agent_name: Name of the agent for logging and identification
        """
        self.agent_name = agent_name
        self.logger = get_logger(f"agent.{agent_name}")
        self._correlation_id = correlation_id.get()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize agent resources and dependencies."""
        if self._initialized:
            self.logger.warning(f"Agent {self.agent_name} already initialized")
            return
        
        try:
            self._initialize_resources()
            self._initialized = True
            self.logger.info(f"Agent {self.agent_name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.agent_name}: {e}")
            raise

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data and return results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed results
            
        Raises:
            AgentProcessingError: If processing fails
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if not self._initialized:
            return
        
        try:
            await self._cleanup_resources()
            self._initialized = False
            self.logger.info(f"Agent {self.agent_name} cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Failed to cleanup agent {self.agent_name}: {e}")
            raise

    @abstractmethod
    def _initialize_resources(self) -> None:
        """Initialize agent-specific resources."""
        pass

    @abstractmethod
    async def _cleanup_resources(self) -> None:
        """Cleanup agent-specific resources."""
        pass

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for request tracing."""
        self._correlation_id = correlation_id
        self.logger = get_logger(f"agent.{self.agent_name}", correlation_id=correlation_id)

    def get_correlation_id(self) -> str:
        """Get current correlation ID."""
        return self._correlation_id

    def log_operation(self, operation: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log agent operation with correlation ID."""
        extra = {"correlation_id": self._correlation_id, "agent": self.agent_name}
        if details:
            extra.update(details)
        
        self.logger.info(f"Operation: {operation}", extra=extra)

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log agent error with correlation ID."""
        extra = {"correlation_id": self._correlation_id, "agent": self.agent_name}
        if context:
            extra.update(context)
        
        self.logger.error(f"Error in {self.agent_name}: {error}", extra=extra, exc_info=True)

    @property
    def is_initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._initialized

    @property
    def health_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        return {
            "agent_name": self.agent_name,
            "initialized": self._initialized,
            "correlation_id": self._correlation_id,
            "status": "healthy" if self._initialized else "uninitialized",
        }
