"""Configuration Manager for handling application configuration and settings."""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator

from ..utils.exceptions import ConfigurationError
from ..utils.logging import get_logger


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "interview_coach"
    username: str = ""
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseConfig":
        """Create DatabaseConfig from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})





@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    
    level: str = "INFO"
    format: str = "json"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggingConfig":
        """Create LoggingConfig from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    secret_key: str = ""
    token_expiry: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    require_https: bool = False
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityConfig":
        """Create SecurityConfig from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PerformanceConfig:
    """Performance configuration settings."""
    
    max_concurrent_sessions: int = 10
    session_timeout: int = 7200  # 2 hours
    auto_save_interval: int = 180  # 3 minutes
    max_questions_per_session: int = 20
    llm_timeout: int = 30  # 30 seconds
    retry_attempts: int = 3
    retry_backoff: float = 1.5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceConfig":
        """Create PerformanceConfig from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StorageConfig:
    """Storage configuration settings."""
    
    base_path: str = "data"
    backup_enabled: bool = True
    max_backup_count: int = 10
    cleanup_interval_hours: int = 24
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageConfig":
        """Create StorageConfig from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LLMProviderConfig(BaseModel):
    """LLM Provider configuration model."""
    
    name: str = Field(..., description="Provider name")
    enabled: bool = Field(default=True, description="Whether provider is enabled")
    api_key: str = Field(..., description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for API calls")
    model: str = Field(..., description="Model name to use")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_tokens: int = Field(default=1000, description="Maximum tokens for responses")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    retries: int = Field(default=3, description="Number of retry attempts")
    rate_limit: int = Field(default=100, description="Requests per minute")
    
    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @validator("timeout")
    def validate_timeout(cls, v):
        if v < 1:
            raise ValueError("Timeout must be at least 1 second")
        return v


class AppConfig(BaseModel):
    """Main application configuration model."""
    
    app_name: str = Field(default="Interview Coach System", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database settings")

    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging settings")
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security settings")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance settings")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage settings")
    
    # LLM Providers
    llm_providers: List[LLMProviderConfig] = Field(default_factory=list, description="LLM provider configurations")
    
    # Feature flags
    features: Dict[str, bool] = Field(default_factory=dict, description="Feature flags")
    
    class Config:
        validate_assignment = True


class ConfigurationManager:
    """Manages application configuration and settings."""
    
    def __init__(self, config_path: str = "config", env_file: str = ".env"):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration directory.
            env_file: Path to environment file.
        """
        self.config_path = Path(config_path)
        self.env_file = Path(env_file)
        self.config: Optional[AppConfig] = None
        self.logger = get_logger("configuration_manager")
        
        # Configuration watchers for hot-reloading
        self._watchers: List[callable] = []
        self._watch_task: Optional[asyncio.Task] = None
        
        # Default configuration
        self._default_config = self._create_default_config()
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration."""
        return AppConfig(
            app_name="Interview Coach System",
            version="1.0.0",
            debug=False,
            environment="development",
            database=DatabaseConfig(),

            logging=LoggingConfig(),
            security=SecurityConfig(),
            performance=PerformanceConfig(),
            storage=StorageConfig(),
            llm_providers=[

                LLMProviderConfig(
                    name="deepseek",
                    enabled=False,
                    api_key="",
                    model="deepseek-chat",
                    timeout=30,
                    max_tokens=1000,
                    temperature=0.7
                ),

            ],
            features={
                "auto_save": True,
                "session_recovery": True,
                "graceful_degradation": True,
                "performance_monitoring": True,
                "audit_logging": False
            }
        )
    
    def initialize(self) -> None:
        """Initialize the configuration manager."""
        try:
            # Load environment variables
            self._load_environment_variables()
            # Load configuration files
            self._load_configuration_files()
            # Validate configuration
            self._validate_configuration()
            # Start configuration watcher if enabled
            # if self.config and self.config.debug:
            #     asyncio.create_task(self._start_configuration_watcher())
            
            self.logger.info("ConfigurationManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ConfigurationManager: {str(e)}")
            raise ConfigurationError(f"Configuration initialization failed: {str(e)}")
    
    def _load_environment_variables(self) -> None:
        """Load environment variables from .env file."""
        try:
            if self.env_file.exists():
                # Load environment variables from .env file
                from dotenv import load_dotenv
                load_dotenv(self.env_file)
                self.logger.info(f"Loaded environment variables from {self.env_file}")
            
            # Set default values for required environment variables
            os.environ.setdefault("DEBUG", "false")
            os.environ.setdefault("LOG_LEVEL", "INFO")
            os.environ.setdefault("ENVIRONMENT", "development")
            
        except Exception as e:
            self.logger.warning(f"Failed to load environment variables: {str(e)}")
    
    def _load_configuration_files(self) -> None:
        """Load configuration from YAML and JSON files."""
        try:
            # Start with default configuration
            config_data = self._default_config.model_dump()
            
            # Load main config file
            main_config_file = self.config_path / "config.yaml"
            if main_config_file.exists():
                yaml_config = self._load_yaml_file(main_config_file)
                # Map flat config structure to nested structure
                config_data = self._map_flat_config_to_nested(yaml_config, config_data)
                self.logger.info(f"Loaded main configuration from {main_config_file}")
            
            # Load environment-specific config
            environment = os.getenv("ENVIRONMENT", "development")
            env_config_file = self.config_path / f"config.{environment}.yaml"
            if env_config_file.exists():
                env_config = self._load_yaml_file(env_config_file)
                config_data = self._map_flat_config_to_nested(env_config, config_data)
                self.logger.info(f"Loaded environment configuration from {env_config_file}")
            
            # Load providers configuration
            providers_file = self.config_path / "providers.yaml"
            if providers_file.exists():
                providers_config = self._load_yaml_file(providers_file)
                if "providers" in providers_config:
                    # Convert providers to LLMProviderConfig format
                    llm_providers = []
                    for name, provider_data in providers_config["providers"].items():
                        if provider_data.get("enabled", False):
                            # Handle environment variable substitution for API keys
                            api_key = provider_data.get("api_key", "")
                            if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
                                env_var_name = api_key[2:-1]  # Remove ${ and }
                                api_key = os.getenv(env_var_name, "")
                                if not api_key:
                                    self.logger.warning(f"Environment variable {env_var_name} not set for provider {name}")
                                    continue  # Skip this provider if API key is not available
                            
                            provider_config = LLMProviderConfig(
                                name=name,
                                enabled=provider_data.get("enabled", False),
                                api_key=api_key,
                                base_url=provider_data.get("base_url"),
                                model=provider_data.get("model", "gpt-3.5-turbo"),
                                timeout=provider_data.get("timeout", 30),
                                max_tokens=provider_data.get("max_tokens", 1000),
                                temperature=provider_data.get("temperature", 0.7),
                                retries=provider_data.get("retries", 3),
                                rate_limit=provider_data.get("rate_limit", 100)
                            )
                            llm_providers.append(provider_config)
                    config_data["llm_providers"] = llm_providers
                    self.logger.info(f"Loaded {len(llm_providers)} enabled LLM providers from {providers_file}")
                    for provider in llm_providers:
                        self.logger.debug(f"Provider: {provider.name}, Model: {provider.model}, Enabled: {provider.enabled}")
                else:
                    self.logger.warning("No 'providers' section found in providers.yaml")
            else:
                self.logger.warning(f"Providers configuration file not found: {providers_file}")
            
            # Create configuration object
            self.config = AppConfig.model_validate(config_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration files: {str(e)}")
            # Fall back to default configuration
            self.config = self._default_config
            self.logger.warning("Using default configuration due to load failure")
    
    def _map_flat_config_to_nested(self, flat_config: Dict[str, Any], nested_config: Dict[str, Any]) -> Dict[str, Any]:
        """Map flat configuration structure to nested structure.
        
        Args:
            flat_config: Flat configuration from YAML file
            nested_config: Nested configuration structure
            
        Returns:
            Updated nested configuration
        """
        try:
            # Map app configuration
            if "app" in flat_config:
                nested_config["app_name"] = flat_config["app"].get("name", nested_config["app_name"])
                nested_config["version"] = flat_config["app"].get("version", nested_config["version"])
                nested_config["debug"] = flat_config["app"].get("debug", nested_config["debug"])
                nested_config["environment"] = flat_config["app"].get("environment", nested_config["environment"])
            
            # Map logging configuration
            if "logging" in flat_config:
                nested_config["logging"]["level"] = flat_config["logging"].get("level", nested_config["logging"]["level"])
                nested_config["logging"]["format"] = flat_config["logging"].get("format", nested_config["logging"]["format"])
                nested_config["logging"]["file_path"] = flat_config["logging"].get("file", nested_config["logging"]["file_path"])
                nested_config["logging"]["max_file_size"] = flat_config["logging"].get("max_size_mb", 10) * 1024 * 1024
                nested_config["logging"]["backup_count"] = flat_config["logging"].get("backup_count", nested_config["logging"]["backup_count"])
                nested_config["logging"]["console_output"] = True
                nested_config["logging"]["file_output"] = flat_config["logging"].get("file") is not None
            

            
            # Map storage configuration
            if "storage" in flat_config:
                nested_config["storage"]["base_path"] = flat_config["storage"].get("base_path", nested_config["storage"]["base_path"])
                nested_config["storage"]["backup_enabled"] = flat_config["storage"].get("backup_enabled", nested_config["storage"]["backup_enabled"])
                nested_config["storage"]["max_backup_count"] = flat_config["storage"].get("max_backup_count", nested_config["storage"]["max_backup_count"])
                nested_config["storage"]["cleanup_interval_hours"] = flat_config["storage"].get("cleanup_interval_hours", nested_config["storage"]["cleanup_interval_hours"])
            
            # Map performance configuration
            if "performance" in flat_config:
                nested_config["performance"]["metrics_enabled"] = flat_config["performance"].get("metrics_enabled", True)
                nested_config["performance"]["collection_interval"] = flat_config["performance"].get("collection_interval", 60)
                nested_config["performance"]["retention_period"] = flat_config["performance"].get("retention_period", 86400)
            
            # Map security configuration
            if "security" in flat_config:
                nested_config["security"]["secret_key"] = flat_config["security"].get("secret_key", nested_config["security"]["secret_key"])
                nested_config["security"]["rate_limiting"] = flat_config["security"].get("rate_limiting", {})
            
            # Map session configuration
            if "session" in flat_config:
                nested_config["performance"]["session_timeout"] = flat_config["session"].get("max_duration_hours", 4) * 3600
                nested_config["performance"]["auto_save_interval"] = flat_config["session"].get("auto_save_interval", nested_config["performance"]["auto_save_interval"])
            
            return nested_config
            
        except Exception as e:
            self.logger.error(f"Failed to map flat config to nested: {str(e)}")
            return nested_config
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file content.
        
        Args:
            file_path: Path to YAML file.
            
        Returns:
            Dictionary containing file content.
        """
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            return yaml.safe_load(content) or {}
            
        except Exception as e:
            self.logger.error(f"Failed to load YAML file {file_path}: {str(e)}")
            return {}
    
    def _validate_configuration(self) -> None:
        """Validate the loaded configuration."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        try:
            # Validate required settings
            if not self.config.llm_providers:
                self.logger.warning("No LLM providers configured")
            
            # Validate enabled providers have API keys
            for provider in self.config.llm_providers:
                if provider.enabled and not provider.api_key:
                    self.logger.warning(f"Provider {provider.name} is enabled but has no API key")
            
            # Validate security settings
            if self.config.security.secret_key == "":
                self.logger.warning("No secret key configured - using default")
                self.config.security.secret_key = "default-secret-key-change-in-production"
            
            # Validate performance settings
            if self.config.performance.max_concurrent_sessions < 1:
                raise ConfigurationError("max_concurrent_sessions must be at least 1")
            
            if self.config.performance.session_timeout < 300:  # 5 minutes
                raise ConfigurationError("session_timeout must be at least 300 seconds")
            
            self.logger.info("Configuration validation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    async def _start_configuration_watcher(self) -> None:
        """Start watching configuration files for changes."""
        try:
            self._watch_task = asyncio.create_task(self._watch_configuration_files())
            self.logger.info("Configuration file watcher started")
            
        except Exception as e:
            self.logger.warning(f"Failed to start configuration watcher: {str(e)}")
    
    async def _watch_configuration_files(self) -> None:
        """Watch configuration files for changes."""
        try:
            while True:
                # Check for configuration file changes
                if await self._check_configuration_changes():
                    self.logger.info("Configuration files changed, reloading...")
                    await self._reload_configuration()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except asyncio.CancelledError:
            self.logger.info("Configuration watcher stopped")
        except Exception as e:
            self.logger.error(f"Configuration watcher error: {str(e)}")
    
    async def _check_configuration_changes(self) -> bool:
        """Check if configuration files have changed.
        
        Returns:
            True if changes detected, False otherwise.
        """
        try:
            # This is a simplified check - in practice, you might want to use
            # a more sophisticated file watching mechanism
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking configuration changes: {str(e)}")
            return False
    
    async def _reload_configuration(self) -> None:
        """Reload configuration from files."""
        try:
            # Reload configuration
            await self._load_configuration_files()
            
            # Validate new configuration
            await self._validate_configuration()
            
            # Notify watchers
            await self._notify_configuration_changed()
            
            self.logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {str(e)}")
    
    async def _notify_configuration_changed(self) -> None:
        """Notify all configuration change watchers."""
        for watcher in self._watchers:
            try:
                if asyncio.iscoroutinefunction(watcher):
                    await watcher(self.config)
                else:
                    watcher(self.config)
            except Exception as e:
                self.logger.error(f"Error in configuration watcher: {str(e)}")
    
    def add_configuration_watcher(self, watcher: callable) -> None:
        """Add a configuration change watcher.
        
        Args:
            watcher: Function to call when configuration changes.
        """
        self._watchers.append(watcher)
    
    def remove_configuration_watcher(self, watcher: callable) -> None:
        """Remove a configuration change watcher.
        
        Args:
            watcher: Function to remove.
        """
        if watcher in self._watchers:
            self._watchers.remove(watcher)
    
    def get_config(self) -> AppConfig:
        """Get the current configuration.
        
        Returns:
            Current application configuration.
            
        Raises:
            ConfigurationError: If configuration is not loaded.
        """
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        return self.config
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration setting.
        
        Args:
            key: Configuration key (dot notation supported).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        if not self.config:
            return default
        
        try:
            # Support dot notation for nested keys
            keys = key.split(".")
            value = self.config
            
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception:
            return default
    
    def set_setting(self, key: str, value: Any) -> bool:
        """Set a configuration setting.
        
        Args:
            key: Configuration key (dot notation supported).
            value: Value to set.
            
        Returns:
            True if setting was updated, False otherwise.
        """
        if not self.config:
            return False
        
        try:
            # Support dot notation for nested keys
            keys = key.split(".")
            target = self.config
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if hasattr(target, k):
                    target = getattr(target, k)
                else:
                    return False
            
            # Set the value
            last_key = keys[-1]
            if hasattr(target, last_key):
                setattr(target, last_key, value)
                return True
            
            return False
            
        except Exception:
            return False
    
    async def save_configuration(self, file_path: Optional[Path] = None) -> bool:
        """Save current configuration to file.
        
        Args:
            file_path: Path to save configuration to.
            
        Returns:
            True if save was successful.
        """
        if not self.config:
            return False
        
        try:
            if file_path is None:
                file_path = self.config_path / "config.yaml"
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            config_dict = self.config.model_dump()
            
            # Save as YAML
            import aiofiles
            
            async with aiofiles.open(file_path, "w") as f:
                await f.write(yaml.dump(config_dict, default_flow_style=False, indent=2))
            
            self.logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def get_llm_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific LLM provider.
        
        Args:
            provider_name: Name of the provider.
            
        Returns:
            Provider configuration if found, None otherwise.
        """
        if not self.config:
            return None
        
        for provider in self.config.llm_providers:
            if provider.name.lower() == provider_name.lower():
                return provider
        
        return None
    
    def get_enabled_llm_providers(self) -> List[LLMProviderConfig]:
        """Get list of enabled LLM providers.
        
        Returns:
            List of enabled provider configurations.
        """
        if not self.config:
            return []
        
        return [p for p in self.config.llm_providers if p.enabled]
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled.
        
        Args:
            feature_name: Name of the feature.
            
        Returns:
            True if feature is enabled, False otherwise.
        """
        if not self.config:
            return False
        
        return self.config.features.get(feature_name, False)
    
    def get_environment(self) -> str:
        """Get current environment.
        
        Returns:
            Current environment string.
        """
        if not self.config:
            return "development"
        
        return self.config.environment
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled.
        
        Returns:
            True if debug mode is enabled.
        """
        if not self.config:
            return False
        
        return self.config.debug
    
    async def cleanup(self) -> None:
        """Clean up configuration manager resources."""
        try:
            # Stop configuration watcher
            if self._watch_task and not self._watch_task.done():
                self._watch_task.cancel()
                try:
                    await self._watch_task
                except asyncio.CancelledError:
                    pass
            
            # Clear watchers
            self._watchers.clear()
            
            self.logger.info("ConfigurationManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Dictionary containing logging configuration.
        """
        if not self.config:
            return {"level": "INFO", "format": "json"}
        
        try:
            return {
                "level": self.config.logging.level,
                "format": self.config.logging.format,
                "file_path": self.config.logging.file_path,
                "max_file_size": self.config.logging.max_file_size,
                "backup_count": self.config.logging.backup_count,
                "console_output": self.config.logging.console_output,
                "file_output": self.config.logging.file_output
            }
        except Exception as e:
            self.logger.error(f"Failed to get logging config: {str(e)}")
            return {"level": "INFO", "format": "json"}
    

    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration.
        
        Returns:
            Dictionary containing storage configuration.
        """
        if not self.config:
            return {}
        
        try:
            return {
                "base_path": self.config.storage.base_path,
                "backup_enabled": self.config.storage.backup_enabled,
                "max_backup_count": self.config.storage.max_backup_count,
                "cleanup_interval_hours": self.config.storage.cleanup_interval_hours
            }
        except Exception as e:
            self.logger.error(f"Failed to get storage config: {str(e)}")
            return {}
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration.
        
        Returns:
            Dictionary containing LLM configuration.
        """
        if not self.config:
            return {}
        
        try:
            return {
                "routing_strategy": "performance_based",
                "health_check_interval": 300,
                "circuit_breaker": {
                    "failure_threshold": 5,
                    "timeout": 60,
                    "half_open_timeout": 30
                },
                "fallback_order": ["deepseek", "qwen"]
            }
        except Exception as e:
            self.logger.error(f"Failed to get LLM config: {str(e)}")
            return {}
    
    def get_llm_provider_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get LLM provider configurations.
        
        Returns:
            Dictionary mapping provider names to their configurations.
        """
        if not self.config:
            return {}

        try:
            provider_configs = {}
            for provider in self.config.llm_providers:
                if provider.enabled:
                    provider_configs[provider.name.lower()] = {
                        "name": provider.name,
                        "is_enabled": provider.enabled,
                        "api_key": provider.api_key,
                        "base_url": provider.base_url,
                        "model": provider.model,
                        "timeout": provider.timeout,
                        "max_tokens": provider.max_tokens,
                        "temperature": provider.temperature,
                        "retries": provider.retries,
                        "rate_limit": provider.rate_limit
                    }
            return provider_configs
        except Exception as e:
            self.logger.error(f"Failed to get LLM provider configs: {str(e)}")
            return {}
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration.
        
        Returns:
            Dictionary containing configuration summary.
        """
        if not self.config:
            return {"error": "Configuration not loaded"}
        
        try:
            return {
                "app_name": self.config.app_name,
                "version": self.config.version,
                "environment": self.config.environment,
                "debug": self.config.debug,
                "llm_providers": {
                    p.name: {
                        "enabled": p.enabled,
                        "model": p.model,
                        "timeout": p.timeout
                    }
                    for p in self.config.llm_providers
                },
                "features": self.config.features,
                "performance": {
                    "max_concurrent_sessions": self.config.performance.max_concurrent_sessions,
                    "session_timeout": self.config.performance.session_timeout,
                    "auto_save_interval": self.config.performance.auto_save_interval
                },

            }
            
        except Exception as e:
            self.logger.error(f"Failed to get configuration summary: {str(e)}")
            return {"error": str(e)}
