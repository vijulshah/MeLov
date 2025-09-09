"""
Pydantic models for agentic bio matching system.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class AgentRole(str, Enum):
    """Agent roles in the multi-agent system."""
    QUERY_PROCESSOR = "query_processor"
    BIO_MATCHER = "bio_matcher"
    SOCIAL_FINDER = "social_finder"
    PROFILE_ANALYZER = "profile_analyzer"
    COMPATIBILITY_SCORER = "compatibility_scorer"
    SUMMARY_GENERATOR = "summary_generator"


class ModelProvider(str, Enum):
    """LLM model providers."""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class SocialPlatform(str, Enum):
    """Social media platforms."""
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"


class MatchQuality(str, Enum):
    """Match quality levels."""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class LLMConfig(BaseModel):
    """Configuration for LLM models."""
    model_name: str = Field(..., description="Model name/path")
    provider: ModelProvider = ModelProvider.HUGGINGFACE
    max_tokens: int = Field(4096, ge=512, le=32768)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    use_case: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    name: str
    model: str
    role: str
    capabilities: List[str]
    max_retries: int = Field(3, ge=1, le=10)
    timeout_seconds: int = Field(60, ge=10, le=300)


class UserQuery(BaseModel):
    """User query for bio matching."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_bio_id: str = Field(..., description="User's bio data ID")
    query_text: str = Field(..., min_length=5, max_length=1000)
    max_results: int = Field(10, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Search preferences
    age_range: Optional[Dict[str, int]] = None  # {"min": 25, "max": 35}
    location_preference: Optional[str] = None
    education_level: Optional[str] = None
    profession_preference: Optional[str] = None
    interests: Optional[List[str]] = None


class ProcessedQuery(BaseModel):
    """Processed and structured user query."""
    query_id: str
    original_query: str
    structured_requirements: Dict[str, Any]
    search_keywords: List[str]
    priority_factors: List[str]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_agent: str
    processing_time_ms: float


class BioMatch(BaseModel):
    """Bio data match from vector store."""
    bio_data_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    personal_info: Dict[str, Any]
    education: Optional[Dict[str, Any]] = None
    professional: Optional[Dict[str, Any]] = None
    interests: Optional[Dict[str, Any]] = None
    lifestyle: Optional[Dict[str, Any]] = None
    relationship: Optional[Dict[str, Any]] = None
    source_file: str
    match_reasons: List[str] = Field(default_factory=list)


class SocialProfile(BaseModel):
    """Social media profile information."""
    platform: SocialPlatform
    profile_url: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    verified: bool = False
    profile_image_url: Optional[str] = None
    last_activity: Optional[datetime] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class LinkedInProfile(BaseModel):
    """LinkedIn-specific profile data."""
    profile_url: str
    headline: Optional[str] = None
    summary: Optional[str] = None
    current_position: Optional[Dict[str, Any]] = None
    experience: List[Dict[str, Any]] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    certifications: List[Dict[str, Any]] = Field(default_factory=list)
    connections_count: Optional[int] = None
    recent_posts: List[Dict[str, Any]] = Field(default_factory=list)
    activity_score: float = Field(0.0, ge=0.0, le=1.0)


class SocialAnalysis(BaseModel):
    """Analysis of social media profiles."""
    bio_data_id: str
    profiles: List[SocialProfile] = Field(default_factory=list)
    linkedin_data: Optional[LinkedInProfile] = None
    
    # Extracted insights
    professional_insights: Dict[str, Any] = Field(default_factory=dict)
    interest_insights: Dict[str, Any] = Field(default_factory=dict)
    personality_traits: List[str] = Field(default_factory=list)
    activity_level: str = "unknown"  # low, medium, high
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    data_freshness: str = "unknown"  # recent, moderate, stale


class CompatibilityScore(BaseModel):
    """Compatibility scoring between user and potential match."""
    bio_data_id: str
    overall_score: float = Field(..., ge=0.0, le=1.0)
    
    # Detailed scores
    bio_similarity: float = Field(..., ge=0.0, le=1.0)
    professional_match: float = Field(..., ge=0.0, le=1.0)
    interests_alignment: float = Field(..., ge=0.0, le=1.0)
    education_compatibility: float = Field(..., ge=0.0, le=1.0)
    social_activity: float = Field(..., ge=0.0, le=1.0)
    
    # Factors
    age_compatibility: float = Field(..., ge=0.0, le=1.0)
    location_compatibility: float = Field(..., ge=0.0, le=1.0)
    lifestyle_compatibility: float = Field(..., ge=0.0, le=1.0)
    
    # Explanations
    positive_factors: List[str] = Field(default_factory=list)
    negative_factors: List[str] = Field(default_factory=list)
    match_quality: MatchQuality
    
    # Reasoning
    score_explanation: str
    recommendation: str


class MatchResult(BaseModel):
    """Complete match result with all analysis."""
    query_id: str
    bio_data_id: str
    rank: int
    
    # Core data
    bio_match: BioMatch
    social_analysis: Optional[SocialAnalysis] = None
    compatibility_score: CompatibilityScore
    
    # Generated content
    profile_summary: str
    match_explanation: str
    why_good_match: List[str] = Field(default_factory=list)
    potential_concerns: List[str] = Field(default_factory=list)
    
    # Metadata
    processing_time_ms: float
    data_completeness: float = Field(..., ge=0.0, le=1.0)


class AgentResponse(BaseModel):
    """Response from an individual agent."""
    agent_name: str
    agent_role: AgentRole
    task_id: str
    success: bool
    
    # Response data
    response_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = None
    
    # Metadata
    processing_time_ms: float
    token_usage: Optional[Dict[str, int]] = None
    model_used: str
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Reasoning (for debugging/explanation)
    reasoning: Optional[str] = None
    intermediate_steps: List[str] = Field(default_factory=list)


class WorkflowStage(BaseModel):
    """Workflow stage configuration."""
    name: str
    agents: List[str]
    dependencies: List[str] = Field(default_factory=list)
    parallel: bool = False
    timeout_seconds: int = 120
    required: bool = True


class WorkflowExecution(BaseModel):
    """Workflow execution tracking."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    user_bio_id: str
    
    # Execution state
    current_stage: str
    completed_stages: List[str] = Field(default_factory=list)
    failed_stages: List[str] = Field(default_factory=list)
    
    # Timing
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    
    # Agent responses
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    
    # Final results
    matches: List[MatchResult] = Field(default_factory=list)
    summary: Optional[str] = None
    
    # Status
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    error_message: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat message in the conversation."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Associated data
    query_id: Optional[str] = None
    execution_id: Optional[str] = None
    
    # Message metadata
    message_type: Literal["query", "response", "clarification", "error", "info"] = "query"
    confidence_score: Optional[float] = None
    
    # Rich content
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)


class ChatSession(BaseModel):
    """Chat session with user."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_bio_id: str
    
    # Conversation history
    messages: List[ChatMessage] = Field(default_factory=list)
    
    # Session state
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    
    # Context
    current_query: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Statistics
    total_queries: int = 0
    successful_matches: int = 0


class SystemMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Agent metrics
    agent_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    model_usage: Dict[str, int] = Field(default_factory=dict)
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # API usage
    social_api_calls: Dict[str, int] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)


class AgentTask(BaseModel):
    """Task for an individual agent."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str
    task_type: str
    
    # Input data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Task configuration
    priority: int = Field(1, ge=1, le=10)
    timeout_seconds: int = Field(60, ge=10, le=300)
    max_retries: int = Field(3, ge=1, le=10)
    
    # Status
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[AgentResponse] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class SystemConfig(BaseModel):
    """System configuration."""
    # LLM models
    llm_models: Dict[str, LLMConfig] = Field(default_factory=dict)
    
    # Agents
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    
    # Workflow
    workflow_stages: List[WorkflowStage] = Field(default_factory=list)
    max_concurrent_agents: int = Field(3, ge=1, le=10)
    
    # Performance
    cache_enabled: bool = True
    cache_ttl_hours: int = Field(24, ge=1, le=168)
    
    # Rate limiting
    requests_per_minute: int = Field(60, ge=1, le=1000)
    requests_per_hour: int = Field(1000, ge=10, le=10000)
    
    # Security
    data_anonymization: bool = True
    pii_filtering: bool = True
    consent_required: bool = True
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
