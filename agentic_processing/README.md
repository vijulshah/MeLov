# Agentic Bio Matching System

A sophisticated multi-agent system for intelligent bio data matching using open-source language models and social media integration.

## 🌟 Features

### Multi-Agent Architecture
- **6 Specialized Agents** working together for comprehensive matching
- **Query Processor**: Understands user requirements in natural language
- **Bio Matcher**: Finds compatible profiles using vector similarity
- **Social Finder**: Discovers LinkedIn, Instagram, Facebook profiles
- **Profile Analyzer**: Extracts insights from social media data
- **Compatibility Scorer**: Calculates multi-dimensional compatibility scores
- **Summary Generator**: Creates personalized match summaries and recommendations

### Open-Source LLM Integration
- **Phi-3-mini**: Fast and efficient for quick analysis
- **Llama-3.2-3B**: Balanced performance for general tasks
- **Mistral-7B**: High-quality text generation
- **Falcon-7B**: Advanced reasoning capabilities
- **Hugging Face Integration**: Easy model switching and deployment

### Advanced Matching Capabilities
- **Vector Database**: FAISS-based similarity search
- **Social Media Analysis**: LinkedIn/Instagram profile discovery
- **Personality Assessment**: AI-powered trait extraction
- **Compatibility Scoring**: Multi-factor scoring algorithm
- **Intelligent Summaries**: Personalized match explanations

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Optional: Install transformers for enhanced LLM support
pip install transformers torch
```

### Basic Usage

```python
import asyncio
from agentic_processing import AgenticBioMatcher

async def find_matches():
    # Initialize the bio matcher
    bio_matcher = AgenticBioMatcher()
    
    # Your bio data
    user_bio = {
        "name": "Alex Johnson",
        "age": 29,
        "location": "San Francisco, CA",
        "occupation": "Software Engineer",
        "interests": ["Technology", "Hiking", "Photography", "Travel"],
        "bio": "Passionate about building innovative products and exploring nature."
    }
    
    # What you're looking for
    user_query = """
    Looking for someone who:
    - Works in tech or creative fields
    - Enjoys outdoor activities
    - Loves to travel and explore
    - Values personal growth
    - Lives in Bay Area, age 25-35
    """
    
    # Find matches
    results = await bio_matcher.find_matches(
        user_query=user_query,
        user_bio=user_bio,
        include_social_search=True,
        max_results=10
    )
    
    # Process results
    if results.get("final_results"):
        matches = results["final_results"]["match_summaries"]
        
        print(f"Found {len(matches)} compatible matches!")
        
        for i, match in enumerate(matches[:3], 1):
            print(f"\n{i}. Compatibility Score: {match['overall_score']:.1%}")
            print(f"   Summary: {match['summary_text']}")
            print(f"   Conversation Starter: \"{match['conversation_starters'][0]}\"")

# Run the matcher
asyncio.run(find_matches())
```

### Simple Bio Matching (No Social Media)

```python
async def simple_matching():
    bio_matcher = AgenticBioMatcher()
    
    user_bio = {
        "name": "Sarah Wilson",
        "age": 26,
        "occupation": "Marketing Manager",
        "interests": ["Marketing", "Yoga", "Books", "Art"]
    }
    
    # Simple bio-only matching
    results = await bio_matcher.simple_bio_matching(
        user_bio=user_bio,
        max_results=5
    )
    
    return results

asyncio.run(simple_matching())
```

## 🔧 Configuration

### Model Configuration

Create `agentic_processing/config/agentic_config.yaml`:

```yaml
agent_configs:
  query_processor:
    model_config:
      model_name: "microsoft/DialoGPT-medium"
      model_type: "huggingface"
      max_tokens: 500
      temperature: 0.7
    max_retries: 3
    timeout: 60

workflow_config:
  enable_social_search: true
  enable_profile_analysis: true
  detailed_summaries: true
  max_final_results: 10
  parallel_processing: false

logging_config:
  log_level: "INFO"
  log_to_file: true
  log_to_console: true
```

### Using Different Models

```python
from agentic_processing import AgenticWorkflowOrchestrator
from agentic_processing.models.agentic_models import ModelConfig, AgentConfig

# Configure Llama-3.2 model
llama_config = ModelConfig(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    model_type="huggingface",
    max_tokens=1000,
    temperature=0.6
)

# Use custom configuration
orchestrator = AgenticWorkflowOrchestrator()
orchestrator.agents["query_processor"].model_config = llama_config
```

## 🏗️ Architecture

### Workflow Pipeline

```
User Query → Query Processor → Bio Matcher → Social Finder → 
Profile Analyzer → Compatibility Scorer → Summary Generator → Results
```

### Data Flow

1. **Query Processing**: Parse user requirements and preferences
2. **Bio Matching**: Find similar profiles using vector search
3. **Social Discovery**: Find LinkedIn/Instagram profiles for matches
4. **Profile Analysis**: Extract personality, interests, professional data
5. **Compatibility Scoring**: Calculate multi-dimensional compatibility
6. **Summary Generation**: Create personalized recommendations

### Agent Details

#### Query Processor Agent
- Parses natural language requirements
- Extracts filters (age, location, occupation)
- Identifies personality preferences
- Sets search parameters

#### Bio Matcher Agent  
- Uses FAISS vector database for similarity search
- Integrates with existing `data_vector_store` module
- Applies filters and ranking
- Returns top candidate matches

#### Social Finder Agent
- Searches LinkedIn for professional profiles
- Finds Instagram accounts using name/location
- Validates profile authenticity using LLM
- Confidence scoring for found profiles

#### Profile Analyzer Agent
- Analyzes LinkedIn work experience and skills
- Extracts personality traits from social posts
- Assesses activity levels and engagement
- Professional growth trajectory analysis

#### Compatibility Scorer Agent
- Multi-factor scoring algorithm
- Demographics, professional, interests, personality
- Weighted compatibility assessment
- Relationship potential evaluation

#### Summary Generator Agent
- Personalized match summaries
- Conversation starter suggestions
- Relationship advice and next steps
- Ranked recommendations with explanations

## 📊 Compatibility Scoring

### Scoring Factors

- **Basic Demographics** (15%): Age, location, education compatibility
- **Professional Alignment** (25%): Career level, industry, goals
- **Interest Compatibility** (20%): Shared hobbies and lifestyle
- **Personality Match** (25%): Complementary and similar traits
- **Values Alignment** (10%): Core values and priorities
- **Social Compatibility** (5%): Communication styles

### Score Interpretation

- **80-100%**: Highly Compatible - Excellent potential
- **70-79%**: Very Compatible - Strong foundation
- **60-69%**: Moderately Compatible - Good potential
- **50-59%**: Some Compatibility - Requires effort
- **<50%**: Limited Compatibility - Friendship potential

## 🔍 Social Media Integration

### Supported Platforms

- **LinkedIn**: Professional profiles, work history, skills
- **Instagram**: Lifestyle content, interests, personality
- **Facebook**: Planned for future versions

### Privacy & Ethics

- Only analyzes publicly available information
- Respects platform terms of service
- No unauthorized data access
- User consent and transparency

### Mock Implementation

Current version includes mock social media APIs for demonstration. Production deployment would integrate with:
- LinkedIn API (with proper authorization)
- Instagram Basic Display API
- Custom web scraping (where legally permitted)

## 📈 Performance & Scalability

### Optimization Features

- **Async Processing**: Non-blocking agent execution
- **Parallel Workflows**: Concurrent agent processing (configurable)
- **Caching**: Vector embeddings and LLM responses
- **Batch Processing**: Multiple user queries simultaneously

### Resource Requirements

- **Memory**: 4-8GB for local LLM inference
- **GPU**: Optional, improves LLM performance
- **Storage**: Vector database scales with user data
- **Network**: Social media API calls require stable connection

## 🔒 Security & Privacy

### Data Protection

- No permanent storage of social media data
- Temporary analysis only
- User consent required for social search
- GDPR/CCPA compliance considerations

### Model Security

- Local LLM inference option (no cloud dependency)
- Open-source models (transparent and auditable)
- No personal data sent to external APIs
- Secure configuration management

## 🛠️ Development & Customization

### Adding New Agents

```python
from agentic_processing.agents.base_agent import BaseAgent
from agentic_processing.models.agentic_models import AgentTask, AgentResponse

class CustomAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return "Your custom agent prompt..."
    
    async def process_task(self, task: AgentTask) -> AgentResponse:
        # Your custom logic
        return AgentResponse(...)
```

### Custom Scoring Algorithms

```python
from agentic_processing.agents.compatibility_scorer import CompatibilityScorerAgent

class CustomScorerAgent(CompatibilityScorerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom scoring weights
        self.scoring_weights = {
            "custom_factor": 0.3,
            "professional_alignment": 0.2,
            # ... other factors
        }
```

### Integration with Existing Systems

```python
# Integrate with your user database
async def integrate_with_user_db():
    from your_app.models import User
    from agentic_processing import AgenticBioMatcher
    
    bio_matcher = AgenticBioMatcher()
    
    user = User.objects.get(id=123)
    user_bio = {
        "name": user.name,
        "age": user.age,
        # Map your user model to bio format
    }
    
    matches = await bio_matcher.find_matches(
        user_query=user.preferences,
        user_bio=user_bio
    )
    
    # Save results to your database
    for match in matches["final_results"]["match_summaries"]:
        # Store match results
        pass
```

## 📚 API Reference

### AgenticBioMatcher

Main interface for the bio matching system.

#### Methods

##### `find_matches(user_query, user_bio, **kwargs)`
- **user_query** (str): Natural language description of preferences
- **user_bio** (dict): User's bio data
- **include_social_search** (bool): Enable social media search
- **max_results** (int): Maximum matches to return
- **detailed_summaries** (bool): Generate detailed vs brief summaries

##### `simple_bio_matching(user_bio, max_results)`
- **user_bio** (dict): User's bio data  
- **max_results** (int): Maximum matches to return

### Bio Data Format

```python
user_bio = {
    "id": "unique_identifier",
    "name": "Full Name",
    "age": 29,
    "location": "City, State/Country", 
    "occupation": "Job Title",
    "education": "Degree - Institution",
    "interests": ["Interest1", "Interest2", ...],
    "bio": "Free text biography",
    "looking_for": "What they're seeking (optional)"
}
```

### Response Format

```python
{
    "workflow_id": "uuid",
    "status": "completed",
    "total_matches": 10,
    "execution_time": 15.3,
    "final_results": {
        "match_summaries": [
            {
                "bio_data_id": "match_id",
                "rank": 1,
                "overall_score": 0.87,
                "summary_text": "Personalized summary...",
                "key_strengths": ["Professional alignment", ...],
                "conversation_starters": ["Question 1", ...],
                "next_steps": ["Suggestion 1", ...],
                "confidence_level": 0.9
            }
        ],
        "overall_recommendations": [...],
        "summary_statistics": {
            "average_compatibility": 0.73,
            "high_quality_count": 3,
            "matches_to_focus_on": 5
        }
    }
}
```

## 🧪 Testing & Demo

### Run Demo

```bash
cd agentic_processing
python main.py
```

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
pytest tests/integration/
```

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python -m agentic_processing.main
```

### Docker Deployment

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "agentic_processing.main"]
```

### Cloud Deployment

- **AWS/Azure/GCP**: Container deployment
- **Model Hosting**: Use managed LLM services or local inference
- **Vector Database**: Managed vector databases or self-hosted FAISS
- **Social APIs**: Configure with proper credentials

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include docstrings for public methods
- Write unit tests for new features
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: Open-source model ecosystem
- **FAISS**: Efficient vector similarity search
- **Pydantic**: Data validation and settings management
- **Open Source LLM Community**: Phi-3, Llama, Mistral, Falcon models

---

**Ready to find meaningful connections through intelligent bio matching! 💕**
