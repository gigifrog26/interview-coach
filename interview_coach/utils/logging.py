"""Logging utilities for the Interview Coach System."""

import logging
import logging.handlers
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

# Correlation ID context for request tracing
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class CorrelationIdFilter(logging.Filter):
    """Log filter to add correlation ID to log records."""
    
    def filter(self, record):
        """Add correlation ID to log record."""
        record.correlation_id = correlation_id.get()
        return True


class StructuredFormatter(logging.Formatter):
    """Structured log formatter for machine-readable logs."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", ""),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", 
                          "msecs", "relativeCreated", "thread", "threadName", 
                          "processName", "process", "getMessage", "exc_info", 
                          "exc_text", "stack_info", "correlation_id"]:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable log formatter for console output."""
    
    def format(self, record):
        """Format log record for human reading."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(8)
        logger_name = record.name.ljust(25)
        correlation_id = getattr(record, "correlation_id", "")
        correlation_str = f"[{correlation_id}] " if correlation_id else ""
        
        message = record.getMessage()
        
        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return f"{timestamp} | {level} | {logger_name} | {correlation_str}{message}"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = False,
    structured: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """Setup logging configuration for the Interview Coach System.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_console: Enable console logging
        enable_file: Enable file logging
        structured: Use structured JSON logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatters
    if structured:
        console_formatter = StructuredFormatter()
        file_formatter = StructuredFormatter()
    else:
        console_formatter = HumanReadableFormatter()
        file_formatter = HumanReadableFormatter()
    
    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(correlation_filter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Log startup message
    startup_logger = logging.getLogger("startup")
    startup_logger.info("Logging system initialized", extra={
        "level": level,
        "console_enabled": enable_console,
        "file_enabled": enable_file,
        "structured": structured
    })


def get_logger(name: str, correlation_id: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with optional correlation ID.
    
    Args:
        name: Logger name
        correlation_id: Optional correlation ID for request tracing
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if correlation_id:
        # Set correlation ID in context
        correlation_id.set(correlation_id)
    
    return logger


def set_correlation_id(correlation_id_value: str) -> None:
    """Set correlation ID for current context.
    
    Args:
        correlation_id_value: Correlation ID value
    """
    correlation_id.set(correlation_id_value)


def get_correlation_id() -> str:
    """Get current correlation ID.
    
    Returns:
        Current correlation ID or empty string if not set
    """
    return correlation_id.get()


def log_function_call(func_name: str, args: dict = None, kwargs: dict = None):
    """Decorator to log function calls with correlation ID.
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(f"function.{func_name}")
            correlation_id_value = get_correlation_id()
            
            logger.debug(f"Function call: {func_name}", extra={
                "function_name": func_name,
                "args": str(args),
                "kwargs": str(kwargs),
                "correlation_id": correlation_id_value
            })
            
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Function completed: {func_name}", extra={
                    "function_name": func_name,
                    "correlation_id": correlation_id_value
                })
                return result
            except Exception as e:
                logger.error(f"Function failed: {func_name}", extra={
                    "function_name": func_name,
                    "error": str(e),
                    "correlation_id": correlation_id_value
                }, exc_info=True)
                raise
        
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(f"function.{func_name}")
            correlation_id_value = get_correlation_id()
            
            logger.debug(f"Function call: {func_name}", extra={
                "function_name": func_name,
                "args": str(args),
                "kwargs": str(kwargs),
                "correlation_id": correlation_id_value
            })
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function completed: {func_name}", extra={
                    "function_name": func_name,
                    "correlation_id": correlation_id_value
                })
                return result
            except Exception as e:
                logger.error(f"Function failed: {func_name}", extra={
                    "function_name": func_name,
                    "error": str(e),
                    "correlation_id": correlation_id_value
                }, exc_info=True)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_performance(operation: str, duration: float, details: dict = None):
    """Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        details: Additional performance details
    """
    logger = get_logger("performance")
    extra = {
        "operation": operation,
        "duration_seconds": duration,
        "correlation_id": get_correlation_id()
    }
    
    if details:
        extra.update(details)
    
    logger.info(f"Performance: {operation} took {duration:.3f}s", extra=extra)


def log_error(error: Exception, context: dict = None, level: str = "ERROR"):
    """Log error with context and correlation ID.
    
    Args:
        error: Exception to log
        context: Additional context information
        level: Log level for the error
    """
    logger = get_logger("error")
    extra = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "correlation_id": get_correlation_id()
    }
    
    if context:
        extra.update(context)
    
    log_method = getattr(logger, level.lower())
    log_method(f"Error occurred: {error}", extra=extra, exc_info=True)
