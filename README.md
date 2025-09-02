# Intelligent Interview Coach System

An AI-powered mock technical interview system using multi-agent architecture, async/await patterns, and external LLM provider APIs.

## 🎯 Features

- **Multi-Agent Architecture**: Intelligent agents for question generation, topic management, and evaluation
- **Dynamic Question Generation**: Context-aware questions based on resume and job requirements
- **Real-time Evaluation**: Immediate feedback and scoring for candidate responses
- **Customer Tier Management**: MVP (20 questions) and Standard (3 questions) tiers
- **Graceful Degradation**: Smart fallback from LLM APIs to cached responses to pre-generated questions
- **Session Recovery**: Auto-save every 2-3 questions with resume capability
- **Multi-LLM Provider Support**: Smart routing between OpenAI, DeepSeek, and Anthropic with automatic failover
- **Advanced Provider Features**: Circuit breaker pattern, rate limiting, exponential backoff retry logic
- **CLI Interface**: User-friendly command-line interface with progress indicators

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface  │  Session Manager  │  Configuration Manager  │
├─────────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  OrchestratorAgent  │  Agent Coordinator  │  Event Bus        │
├─────────────────────────────────────────────────────────────────┤
│                    Agent Layer                                  │
├─────────────────────────────────────────────────────────────────┤
│ InterviewerAgent │ TopicManagerAgent │ EvaluatorAgent │ Cache  │
├─────────────────────────────────────────────────────────────────┤
│                    Service Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  LLM Provider Manager  │  Storage Manager  │  Logging Service  │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  File System  │  Network Layer  │  Configuration Files        │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- Internet connectivity for LLM API access
- API keys for at least one LLM provider (OpenAI, DeepSeek, or Anthropic)

### LLM Provider Setup

The system supports multiple LLM providers with automatic failover and load balancing:

#### Supported Providers

1. **OpenAI (GPT-4, GPT-3.5-turbo)**
   - Get API key from [OpenAI Platform](https://platform.openai.com/)
   - Set `OPENAI_API_KEY` environment variable

2. **DeepSeek (DeepSeek Chat)**
   - Get API key from [DeepSeek Platform](https://platform.deepseek.com/)
   - Set `DEEPSEEK_API_KEY` environment variable

3. **Anthropic (Claude)**
   - Get API key from [Anthropic Console](https://console.anthropic.com/)
   - Set `ANTHROPIC_API_KEY` environment variable

#### Configuration

1. **Environment Variables** (`.env` file):
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   DEEPSEEK_API_KEY=your_deepseek_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   ```

2. **Provider Configuration** (`config/providers.yaml`):
   ```yaml
   providers:
     openai:
       name: "OpenAI"
       enabled: true
       api_key: "${OPENAI_API_KEY}"
       model: "gpt-4"
       timeout: 30
       rate_limit: 100
   ```

3. **Test Providers**:
   ```bash
   python test_providers.py
   ```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd interview-coach
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure LLM providers**
   ```bash
   cp config/providers.yaml.example config/providers.yaml
   # Edit providers.yaml with your API configurations
   ```

### Basic Usage

1. **Start a new interview session**
   ```bash
   poetry run interview-coach start-interview  --resume sample_resume.txt --job-desc sample_job_description.txt   --customer-id user123  --tier mvp
   ```

2. **Resume an existing session**
   ```bash
   poetry run interview-coach resume-interview --session-id sess_abc123
   ```

3. **List all sessions**
   ```bash
   poetry run interview-coach list-sessions
   ```

4. **Check system status**
   ```bash
   poetry run interview-coach status
   ```

## 📁 Project Structure

```
interview_coach/
├── __init__.py                 # Package initialization
├── cli.py                      # Main CLI interface
├── agents/                     # Multi-agent system
│   ├── __init__.py
│   ├── base_agent.py          # Base agent interface
│   ├── interviewer_agent.py   # Question generation agent
│   ├── topic_manager_agent.py # Topic management agent
│   ├── evaluator_agent.py     # Response evaluation agent
│   └── orchestrator_agent.py  # Main orchestration agent
├── models/                     # Data models
│   ├── __init__.py
│   ├── base.py                # Base model classes
│   ├── enums.py               # Enumeration types
│   ├── interview.py           # Interview session models
│   ├── resume.py              # Resume data models
│   └── job.py                 # Job description models
├── services/                   # Core services
│   ├── __init__.py
│   ├── llm_manager.py         # LLM provider management
│   ├── storage_manager.py     # Data persistence
│   ├── cache_manager.py       # Caching system
│   └── configuration_manager.py # Configuration management
└── utils/                      # Utilities
    ├── __init__.py
    ├── exceptions.py           # Custom exceptions
    └── logging.py              # Logging utilities
```

## ⚙️ Configuration

### Quick Setup

To get started quickly, run the setup script:

```bash
python setup_config.py
```

This will create a `.env` file and verify your configuration.

### Environment Variables

Create a `.env` file with the following variables:

```env
# Application Settings
DEBUG=false
LOG_LEVEL=INFO
LOG_FILE=logs/interview_coach.log

# LLM Provider API Keys - REQUIRED for the system to work
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Cache Settings
CACHE_ENABLED=true
CACHE_EXPIRATION_HOURS=24
CACHE_MAX_SIZE_MB=100

# Session Settings
AUTO_SAVE_FREQUENCY=3
MAX_SESSION_DURATION_HOURS=2
```

### Provider Configuration

Configure LLM providers in `config/providers.yaml`:

```yaml
providers:
  deepseek:
    name: "DeepSeek"
    enabled: true
    api_key: "${DEEPSEEK_API_KEY}"  # Environment variable substitution
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    timeout: 30
    max_tokens: 1000
    temperature: 0.7
    retries: 3
    rate_limit: 100
```

### Testing Configuration

After setting up your API keys, test the configuration:

```bash
python test_config.py
```

This will verify that all providers are properly configured and available.
    max_retries: 3
    rate_limit: 100
```

## 🔧 Development

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting (88 character line limit)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework

### Pre-commit Hooks

Install pre-commit hooks to automatically format and lint code:

```bash
poetry run pre-commit install
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=interview_coach

# Run specific test categories
poetry run pytest -m unit
poetry run pytest -m integration
```

### Code Formatting

```bash
# Format code
poetry run black interview_coach/

# Sort imports
poetry run isort interview_coach/

# Check formatting
poetry run black --check interview_coach/
poetry run isort --check-only interview_coach/
```

## 📊 Monitoring and Logging

### Logging

The system provides comprehensive logging with:

- **Structured logging** for machine readability
- **Correlation IDs** for request tracing
- **Performance metrics** logging
- **Rotating log files** with size limits

### Health Checks

Monitor system health with:

```bash
poetry run interview-coach status
```

This provides information about:
- Configuration status
- LLM provider health
- Agent status
- Available providers

## 🚨 Error Handling

The system implements robust error handling:

- **Circuit breaker pattern** for LLM providers
- **Exponential backoff** for retries
- **Graceful degradation** with fallback mechanisms
- **Comprehensive error logging** with context

## 🔧 Troubleshooting

### Common Issues

#### "No LLM providers available" Error

This error occurs when the system cannot find or initialize any LLM providers. To fix:

1. **Check your .env file exists and contains API keys:**
   ```bash
   python setup_config.py
   ```

2. **Verify your API keys are valid:**
   - Test with the provider's API directly
   - Check if you have sufficient credits
   - Verify the API key format

3. **Test configuration:**
   ```bash
   python test_config.py
   ```

4. **Check logs for specific errors:**
   ```bash
   tail -f logs/interview_coach.log
   ```

#### Provider Initialization Failures

If specific providers fail to initialize:

1. **Check provider configuration in `config/providers.yaml`**
2. **Verify API endpoints are accessible**
3. **Check rate limits and quotas**
4. **Review provider-specific error messages in logs**

#### Configuration Issues

If configuration loading fails:

1. **Validate YAML syntax in config files**
2. **Check file permissions**
3. **Verify environment variable substitution**
4. **Run configuration test: `python test_config.py`**

## 🔒 Security

- **Environment-based** API key management
- **Input validation** and sanitization
- **Secure configuration** handling
- **Audit logging** for sensitive operations

## 📈 Performance

- **Async/await** patterns for concurrent operations
- **Smart caching** with TTL expiration
- **Connection pooling** for HTTP clients
- **Background cleanup** tasks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Include tests for new functionality
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- Check the [documentation](docs/)
- Review [existing issues](issues/)
- Create a [new issue](issues/new)

## 🔮 Roadmap

- [ ] Video interview support
- [ ] Advanced analytics dashboard
- [ ] Integration with learning management systems
- [ ] Multi-language support
- [ ] Advanced question personalization
- [ ] Behavioral analysis integration

## 🙏 Acknowledgments

- Built with modern Python async/await patterns
- Inspired by multi-agent system architectures
- Leverages state-of-the-art LLM providers
- Follows production-ready development practices
