"""LLM Provider Management Service."""

import json
import traceback
import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger
from ..utils.exceptions import LLMProviderError
from ..services.configuration_manager import ConfigurationManager


class ProviderHealth(Enum):
    """LLM provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class LLMRequest:
    """LLM request data."""
    type: str
    prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    timeout: Optional[int] = None


@dataclass
class LLMResponse:
    """LLM response data."""
    content: str
    provider: str
    response_time: float
    tokens_used: Optional[int] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.provider_name = config.get("name", "unknown")
        self.logger = get_logger(f"llm.provider.{self.provider_name}")
        self._is_available = True
        self._last_health_check = None
        self._health_status = ProviderHealth.UNKNOWN

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up provider resources."""
        pass

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the provider."""
        pass

    @abstractmethod
    def generate_question(self, context: Dict[str, Any]) -> str:
        """Generate question based on context."""
        pass

    @abstractmethod
    def evaluate_response(self, question: str, response: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate response based on criteria."""
        pass

    async def health_check(self) -> ProviderHealth:
        """Check provider health status."""
        try:
            # Simple health check - try to generate a short text
            await self.generate_text("Hello", max_tokens=5)
            return ProviderHealth.HEALTHY
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return ProviderHealth.UNHEALTHY

    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self._is_available

    @property
    def health_status(self) -> ProviderHealth:
        """Get current health status."""
        return self._health_status

    def mark_unavailable(self) -> None:
        """Mark provider as unavailable."""
        self._is_available = False
        self._health_status = ProviderHealth.UNHEALTHY
        self.logger.warning(f"Provider {self.provider_name} marked as unavailable")

    def mark_available(self) -> None:
        """Mark provider as available."""
        self._is_available = True
        self._health_status = ProviderHealth.HEALTHY
        self.logger.info(f"Provider {self.provider_name} marked as available")


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.logger = get_logger("circuit_breaker")

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (datetime.now() - self.last_failure_time).seconds >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to half-open state")
                return True
            return False
        
        return True

    def on_success(self) -> None:
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.logger.info("Circuit breaker closed after successful operation")

    def on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.warning("Circuit breaker reopened after failure in half-open state")

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state


class LLMProviderManager:
    """Manages multiple LLM providers with smart routing."""

    def __init__(self, config_manager: ConfigurationManager):
        """Initialize LLM provider manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.providers: Dict[str, LLMProvider] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = get_logger("llm.manager")
        self._initialized = False
        self._provider_performance: Dict[str, List[float]] = {}
        self._routing_strategy = "performance_based"

    def initialize(self) -> None:
        """Initialize the LLM provider manager (alias for initialize_providers)."""
        self.initialize_providers()

    def initialize_providers(self) -> None:
        """Initialize all configured LLM providers."""
        if self._initialized:
            return
        
        try:
            provider_configs = self.config_manager.get_llm_provider_configs()
            for provider_name, config in provider_configs.items():
                if config.get("is_enabled", False):
                    provider = self._create_provider(provider_name, config)
                    if provider:
                        self.providers[provider_name] = provider
                        self.circuit_breakers[provider_name] = CircuitBreaker(
                            failure_threshold=config.get("retries", 3),
                            timeout=config.get("timeout", 30)
                        )
                        self._provider_performance[provider_name] = []
            
            self._initialized = True
            self.logger.info(f"Initialized {len(self.providers)} LLM providers")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM providers: {e}")
            self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    def _create_provider(self, provider_name: str, config: Dict[str, Any]) -> Optional[LLMProvider]:
        """Create provider instance based on configuration."""
        try:
            if provider_name.lower() == "deepseek":
                from ..providers.deepseek_provider import DeepSeekProvider
                provider = DeepSeekProvider(config)
                provider.initialize()
                return provider
            elif provider_name.lower() == "qwen":
                from ..providers.qwen_provider import QwenProvider
                provider = QwenProvider(config)
                provider.initialize()
                return provider
            else:
                self.logger.warning(f"Unknown provider type: {provider_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create provider {provider_name}: {e}")
            self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return None

    def get_best_provider(self) -> LLMProvider:
        """Select best provider based on routing strategy."""
        if not self.providers:
            # Provide more helpful error message
            raise LLMProviderError(
                "No LLM providers available. Please check:\n"
                "1. Your .env file contains valid API keys\n"
                "2. The providers.yaml file has enabled providers\n"
                "3. Environment variables are properly set\n"
                "4. API keys are valid and have sufficient credits"
            )
        
        available_providers = [
            name for name, provider in self.providers.items()
            if provider.is_available and self.circuit_breakers[name].can_execute()
        ]
        
        if not available_providers:
            raise LLMProviderError("No healthy LLM providers available")
        
        if self._routing_strategy == "performance_based":
            return self._select_best_performing_provider(available_providers)
        elif self._routing_strategy == "round_robin":
            return self._select_round_robin_provider(available_providers)
        else:
            return self.providers[available_providers[0]]

    def _select_best_performing_provider(self, available_providers: List[str]) -> LLMProvider:
        """Select provider with best performance."""
        best_provider = None
        best_performance = float('inf')
        
        for provider_name in available_providers:
            performance_history = self._provider_performance.get(provider_name, [])
            if performance_history:
                avg_performance = sum(performance_history) / len(performance_history)
                if avg_performance < best_performance:
                    best_performance = avg_performance
                    best_provider = provider_name
        
        if best_provider:
            return self.providers[best_provider]
        
        # Fallback to first available provider
        return self.providers[available_providers[0]]

    def _select_round_robin_provider(self, available_providers: List[str]) -> LLMProvider:
        """Select provider using round-robin strategy."""
        # Simple round-robin implementation
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        provider_name = available_providers[self._round_robin_index % len(available_providers)]
        self._round_robin_index += 1
        
        return self.providers[provider_name]

    def make_request(self, request: LLMRequest) -> LLMResponse:
        """Make request to appropriate LLM provider."""
        start_time = time.time()
        
        try:
            provider = self.get_best_provider()
            circuit_breaker = self.circuit_breakers[provider.provider_name]
            
            if not circuit_breaker.can_execute():
                raise LLMProviderError(f"Circuit breaker is open for provider {provider.provider_name}")
            
            self.logger.info(f"Making request to provider {provider.provider_name}")
            # Make the request
            if request.type == "question_generation":
                content = provider.generate_question(request.context)
            elif request.type == "response_evaluation":
                content = provider.evaluate_response(
                    request.context.get("question", ""),
                    request.context.get("response", ""),
                    request.context.get("criteria", {})
                )
            else:
                raise LLMProviderError(f"Unknown request type: {request.type}")
            
            response_time = time.time() - start_time
            
            # Update performance metrics
            self._update_provider_performance(provider.provider_name, response_time)
            
            # Mark circuit breaker as successful
            circuit_breaker.on_success()
            
            # Handle content type conversion for evaluation responses
            if request.type == "response_evaluation" and isinstance(content, dict):
                # For evaluation responses, store the dict in metadata and convert to string
                content_str = json.dumps(content, ensure_ascii=False)
                metadata = {
                    "request_type": request.type,
                    "evaluation_data": content
                }
            else:
                # For other response types, content is already a string
                content_str = str(content) if content is not None else ""
                metadata = {"request_type": request.type}
            
            return LLMResponse(
                content=content_str,
                provider=provider.provider_name,
                response_time=response_time,
                metadata=metadata
            )
            
        except Exception as e:
            # Mark circuit breaker as failed
            if 'provider' in locals() and 'circuit_breaker' in locals():
                circuit_breaker.on_failure()
                provider.mark_unavailable()
            
            self.logger.error(f"LLM request failed: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Stack trace:\n{traceback.format_exc()}")
            raise LLMProviderError(f"LLM request failed: {e}")

    def _update_provider_performance(self, provider_name: str, response_time: float) -> None:
        """Update provider performance metrics."""
        if provider_name not in self._provider_performance:
            self._provider_performance[provider_name] = []
        
        performance_history = self._provider_performance[provider_name]
        performance_history.append(response_time)
        
        # Keep only last 10 performance measurements
        if len(performance_history) > 10:
            performance_history.pop(0)

    async def handle_provider_failure(self, provider: LLMProvider) -> None:
        """Handle provider failure and update routing."""
        provider.mark_unavailable()
        circuit_breaker = self.circuit_breakers.get(provider.provider_name)
        if circuit_breaker:
            circuit_breaker.on_failure()
        
        self.logger.warning(f"Provider {provider.provider_name} marked as failed")

    def get_provider_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all providers."""
        health_status = {}
        
        for provider_name, provider in self.providers.items():
            circuit_breaker = self.circuit_breakers.get(provider_name)
            performance_history = self._provider_performance.get(provider_name, [])
            
            health_status[provider_name] = {
                "available": provider.is_available,
                "health_status": provider.health_status.value,
                "circuit_breaker_state": circuit_breaker.get_state().value if circuit_breaker else "unknown",
                "avg_response_time": sum(performance_history) / len(performance_history) if performance_history else 0,
                "total_requests": len(performance_history),
            }
        
        return health_status

    async def health_check_all_providers(self) -> None:
        """Perform health check on all providers."""
        for provider_name, provider in self.providers.items():
            try:
                health_status = await provider.health_check()
                provider._health_status = health_status
                
                if health_status == ProviderHealth.HEALTHY:
                    provider.mark_available()
                else:
                    provider.mark_unavailable()
                    
            except Exception as e:
                self.logger.error(f"Health check failed for {provider_name}: {e}")
                provider.mark_unavailable()

    @property
    def is_initialized(self) -> bool:
        """Check if provider manager is initialized."""
        return self._initialized

    @property
    def available_providers_count(self) -> int:
        """Get count of available providers."""
        return sum(1 for provider in self.providers.values() if provider.is_available)

    async def cleanup(self) -> None:
        """Clean up all providers and resources."""
        try:
            for provider in self.providers.values():
                try:
                    await provider.cleanup()
                except Exception as e:
                    self.logger.error(f"Failed to cleanup provider {provider.provider_name}: {e}")
            
            self.providers.clear()
            self.circuit_breakers.clear()
            self._provider_performance.clear()
            self._initialized = False
            
            self.logger.info("LLM Provider Manager cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Failed to cleanup LLM Provider Manager: {e}")
