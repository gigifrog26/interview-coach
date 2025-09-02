"""Custom exceptions for the Interview Coach System."""

from typing import Optional, Any, Dict


class InterviewCoachError(Exception):
    """Base exception for all Interview Coach System errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(InterviewCoachError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the configuration error.
        
        Args:
            message: Error message
            config_key: Optional configuration key that caused the error
            details: Optional additional error details
        """
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class LLMProviderError(InterviewCoachError):
    """Exception raised for LLM provider-related errors."""
    
    def __init__(self, message: str, provider_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the LLM provider error.
        
        Args:
            message: Error message
            provider_name: Optional name of the LLM provider that caused the error
            details: Optional additional error details
        """
        super().__init__(message, "LLM_PROVIDER_ERROR", details)
        self.provider_name = provider_name





class StorageError(InterviewCoachError):
    """Exception raised for storage-related errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the storage error.
        
        Args:
            message: Error message
            file_path: Optional file path that caused the error
            details: Optional additional error details
        """
        super().__init__(message, "STORAGE_ERROR", details)
        self.file_path = file_path


class AgentError(InterviewCoachError):
    """Exception raised for agent-related errors."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the agent error.
        
        Args:
            message: Error message
            agent_name: Optional name of the agent that caused the error
            details: Optional additional error details
        """
        super().__init__(message, "AGENT_ERROR", details)
        self.agent_name = agent_name


class QuestionGenerationError(InterviewCoachError):
    """Exception raised for question generation errors."""
    
    def __init__(self, message: str, topic: Optional[str] = None, difficulty: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the question generation error.
        
        Args:
            message: Error message
            topic: Optional topic that failed to generate question for
            difficulty: Optional difficulty level that failed
            details: Optional additional error details
        """
        super().__init__(message, "QUESTION_GENERATION_ERROR", details)
        self.topic = topic
        self.difficulty = difficulty


class EvaluationError(InterviewCoachError):
    """Exception raised for evaluation errors."""
    
    def __init__(self, message: str, question_id: Optional[str] = None, evaluation_type: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the evaluation error.
        
        Args:
            message: Error message
            question_id: Optional question ID that failed evaluation
            evaluation_type: Optional type of evaluation that failed
            details: Optional additional error details
        """
        super().__init__(message, "EVALUATION_ERROR", details)
        self.question_id = question_id
        self.evaluation_type = evaluation_type


class OrchestrationError(InterviewCoachError):
    """Exception raised for orchestration errors."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, step_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the orchestration error.
        
        Args:
            message: Error message
            agent_name: Optional name of the agent that failed
            step_name: Optional name of the step that failed
            details: Optional additional error details
        """
        super().__init__(message, "ORCHESTRATION_ERROR", details)
        self.agent_name = agent_name
        self.step_name = step_name


class SessionError(InterviewCoachError):
    """Exception raised for session-related errors."""
    
    def __init__(self, message: str, session_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the session error.
        
        Args:
            message: Error message
            session_id: Optional session ID that caused the error
            details: Optional additional error details
        """
        super().__init__(message, "SESSION_ERROR", details)
        self.session_id = session_id


class ValidationError(InterviewCoachError):
    """Exception raised for data validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the validation error.
        
        Args:
            message: Error message
            field_name: Optional field name that failed validation
            details: Optional additional error details
        """
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field_name = field_name


class ParsingError(InterviewCoachError):
    """Exception raised for parsing-related errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, parser_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the parsing error.
        
        Args:
            message: Error message
            file_path: Optional file path that caused the parsing error
            parser_name: Optional name of the parser that caused the error
            details: Optional additional error details
        """
        super().__init__(message, "PARSING_ERROR", details)
        self.file_path = file_path
        self.parser_name = parser_name


class FileValidationError(InterviewCoachError):
    """Exception raised for file validation errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, validation_rule: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the file validation error.
        
        Args:
            message: Error message
            file_path: Optional file path that failed validation
            validation_rule: Optional validation rule that failed
            details: Optional additional error details
        """
        super().__init__(message, "FILE_VALIDATION_ERROR", details)
        self.file_path = file_path
        self.validation_rule = validation_rule


class NetworkError(InterviewCoachError):
    """Exception raised for network-related errors."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the network error.
        
        Args:
            message: Error message
            url: Optional URL that caused the network error
            status_code: Optional HTTP status code
            details: Optional additional error details
        """
        super().__init__(message, "NETWORK_ERROR", details)
        self.url = url
        self.status_code = status_code


class RateLimitError(InterviewCoachError):
    """Exception raised for rate limiting errors."""
    
    def __init__(self, message: str, provider_name: Optional[str] = None, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the rate limit error.
        
        Args:
            message: Error message
            provider_name: Optional name of the provider that hit rate limits
            retry_after: Optional seconds to wait before retrying
            details: Optional additional error details
        """
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.provider_name = provider_name
        self.retry_after = retry_after


class CircuitBreakerError(InterviewCoachError):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, provider_name: Optional[str] = None, circuit_state: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the circuit breaker error.
        
        Args:
            message: Error message
            provider_name: Optional name of the provider with open circuit
            circuit_state: Optional current state of the circuit breaker
            details: Optional additional error details
        """
        super().__init__(message, "CIRCUIT_BREAKER_ERROR", details)
        self.provider_name = provider_name
        self.circuit_state = circuit_state


class TimeoutError(InterviewCoachError):
    """Exception raised for timeout errors."""
    
    def __init__(self, message: str, operation: Optional[str] = None, timeout_seconds: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the timeout error.
        
        Args:
            message: Error message
            operation: Optional operation that timed out
            timeout_seconds: Optional timeout duration in seconds
            details: Optional additional error details
        """
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class AuthenticationError(InterviewCoachError):
    """Exception raised for authentication errors."""
    
    def __init__(self, message: str, auth_method: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the authentication error.
        
        Args:
            message: Error message
            auth_method: Optional authentication method that failed
            details: Optional additional error details
        """
        super().__init__(message, "AUTHENTICATION_ERROR", details)
        self.auth_method = auth_method


class AuthorizationError(InterviewCoachError):
    """Exception raised for authorization errors."""
    
    def __init__(self, message: str, required_permission: Optional[str] = None, user_role: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the authorization error.
        
        Args:
            message: Error message
            required_permission: Optional permission that was required
            user_role: Optional role of the user
            details: Optional additional error details
        """
        super().__init__(message, "AUTHORIZATION_ERROR", details)
        self.required_permission = required_permission
        self.user_role = user_role


class ResourceNotFoundError(InterviewCoachError):
    """Exception raised when a requested resource is not found."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the resource not found error.
        
        Args:
            message: Error message
            resource_type: Optional type of resource that was not found
            resource_id: Optional ID of resource that was not found
            details: Optional additional error details
        """
        super().__init__(message, "RESOURCE_NOT_FOUND_ERROR", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class DuplicateResourceError(InterviewCoachError):
    """Exception raised when trying to create a duplicate resource."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the duplicate resource error.
        
        Args:
            message: Error message
            resource_type: Optional type of resource that is duplicate
            resource_id: Optional ID of duplicate resource
            details: Optional additional error details
        """
        super().__init__(message, "DUPLICATE_RESOURCE_ERROR", details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ServiceUnavailableError(InterviewCoachError):
    """Exception raised when a service is unavailable."""
    
    def __init__(self, message: str, service_name: Optional[str] = None, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the service unavailable error.
        
        Args:
            message: Error message
            service_name: Optional name of the unavailable service
            retry_after: Optional seconds to wait before retrying
            details: Optional additional error details
        """
        super().__init__(message, "SERVICE_UNAVAILABLE_ERROR", details)
        self.service_name = service_name
        self.retry_after = retry_after


class DataIntegrityError(InterviewCoachError):
    """Exception raised for data integrity violations."""
    
    def __init__(self, message: str, data_type: Optional[str] = None, constraint: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize the data integrity error.
        
        Args:
            message: Error message
            data_type: Optional type of data that violated integrity
            constraint: Optional constraint that was violated
            details: Optional additional error details
        """
        super().__init__(message, "DATA_INTEGRITY_ERROR", details)
        self.data_type = data_type
        self.constraint = constraint
