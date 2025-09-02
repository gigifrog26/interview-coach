# Intelligent Interview Coach System

An AI-powered mock technical interview system using multi-agent architecture, async/await patterns, and external LLM provider APIs.

## 🎯 Features

- **Multi-Agent Architecture**: Intelligent agents for question generation, topic management, and evaluation
- **Dynamic Question Generation**: Context-aware questions based on resume and job requirements
- **Real-time Evaluation**: Immediate feedback and scoring for candidate responses
- **Customer Tier Management**: MVP (20 questions) and Standard (3 questions) tiers
- **Graceful Degradation**: Smart fallback from LLM APIs to cached responses to pre-generated questions
- **Session Recovery**: Auto-save every 2-3 questions with resume capability
- **Multi-LLM Provider Support**: Smart routing between OpenAI, Qwen with automatic failover
- **Advanced Provider Features**: Circuit breaker pattern, rate limiting, exponential backoff retry logic
- **CLI Interface**: User-friendly command-line interface with progress indicators

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  CLI Interface  │  Session Manager  │  Configuration Manager    │
├─────────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                     OrchestratorAgent                           │                        
├─────────────────────────────────────────────────────────────────┤
│                    Agent Layer                                  │
├─────────────────────────────────────────────────────────────────┤
│ InterviewerAgent │ TopicManagerAgent │ EvaluatorAgent           │
├─────────────────────────────────────────────────────────────────┤
│                    Service Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  LLM Provider Manager  │  Storage Manager  │  Logging Service   │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  File System  │  Network Layer  │  Configuration Files          │
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


1. **DeepSeek (DeepSeek Chat)**
   - Get API key from [DeepSeek Platform](https://platform.deepseek.com/)
   - Set `DEEPSEEK_API_KEY` environment variable

2. **Qwen (Qwen)**
   - Get API key from Qwen
   - Set `QWEN_API_KEY` environment variable

#### Configuration

1. **Environment Variables** (`.env` file):
   ```bash
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   QWEN_API_KEY=your_qwen_api_key_here
   ```

2. **Provider Configuration** (`config/providers.yaml`):


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
├── providers/                  # LLM provider implementations
│   ├── __init__.py
│   ├── deepseek_provider.py   # DeepSeek API provider
│   └── qwen_provider.py       # Qwen API provider
├── services/                   # Core services
│   ├── __init__.py
│   ├── llm_manager.py         # LLM provider management
│   ├── storage_manager.py     # Data persistence
│   ├── configuration_manager.py # Configuration management
│   └── parsing_service.py     # File parsing service
├── parsers/                    # File parsing modules
│   ├── __init__.py
│   ├── base_parser.py         # Base parser interface
│   ├── resume_parser.py       # Resume parsing logic
│   ├── job_parser.py          # Job description parsing
│   └── file_handlers.py       # File format handlers
└── utils/                      # Utilities
    ├── __init__.py
    ├── exceptions.py           # Custom exceptions
    └── logging.py              # Logging utilities
```


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
