"""
MCP Tool definitions for the agentic bio matching system.
"""
from typing import Dict, Any, List
from .mcp_client import MCPTool, MCPResource, MCPPrompt


# Bio Processing Tools
BIO_EXTRACTION_TOOL = MCPTool(
    name="extract_bio_data",
    description="Extract structured bio data from raw text",
    input_schema={
        "type": "object",
        "properties": {
            "raw_text": {"type": "string", "description": "Raw bio text to extract"},
            "source": {"type": "string", "description": "Source of the bio data"},
            "format": {"type": "string", "enum": ["matrimony", "dating", "social"], "default": "matrimony"}
        },
        "required": ["raw_text"]
    }
)

BIO_STANDARDIZATION_TOOL = MCPTool(
    name="standardize_bio_data",
    description="Standardize bio data format across different sources",
    input_schema={
        "type": "object",
        "properties": {
            "bio_data": {"type": "object", "description": "Raw bio data to standardize"},
            "target_schema": {"type": "string", "description": "Target schema version", "default": "v1.0"}
        },
        "required": ["bio_data"]
    }
)

BIO_VALIDATION_TOOL = MCPTool(
    name="validate_bio_data",
    description="Validate bio data completeness and accuracy",
    input_schema={
        "type": "object",
        "properties": {
            "bio_data": {"type": "object", "description": "Bio data to validate"},
            "strict_mode": {"type": "boolean", "default": False, "description": "Enable strict validation"}
        },
        "required": ["bio_data"]
    }
)

# Vector Search Tools
VECTOR_SEARCH_TOOL = MCPTool(
    name="search_similar_profiles",
    description="Search for similar profiles using vector similarity",
    input_schema={
        "type": "object",
        "properties": {
            "query_vector": {"type": "array", "items": {"type": "number"}, "description": "Query vector"},
            "query_text": {"type": "string", "description": "Alternative text query"},
            "top_k": {"type": "integer", "default": 10, "description": "Number of results to return"},
            "filters": {"type": "object", "description": "Additional filters to apply"},
            "similarity_threshold": {"type": "number", "default": 0.7, "description": "Minimum similarity score"}
        }
    }
)

VECTOR_INDEX_TOOL = MCPTool(
    name="index_bio_profile",
    description="Index a bio profile in the vector store",
    input_schema={
        "type": "object",
        "properties": {
            "bio_data": {"type": "object", "description": "Bio data to index"},
            "metadata": {"type": "object", "description": "Additional metadata"},
            "index_name": {"type": "string", "default": "bio_profiles", "description": "Index to use"}
        },
        "required": ["bio_data"]
    }
)

VECTOR_EMBEDDING_TOOL = MCPTool(
    name="generate_embeddings",
    description="Generate embeddings for bio data",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to embed"},
            "model": {"type": "string", "default": "sentence-transformers/all-MiniLM-L6-v2", "description": "Embedding model to use"}
        },
        "required": ["text"]
    }
)

# Social Analysis Tools
SOCIAL_PROFILE_SEARCH_TOOL = MCPTool(
    name="search_social_profiles",
    description="Search for social media profiles based on bio data",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Person's name"},
            "email": {"type": "string", "description": "Email address"},
            "phone": {"type": "string", "description": "Phone number"},
            "location": {"type": "string", "description": "Location information"},
            "platforms": {"type": "array", "items": {"type": "string"}, "default": ["linkedin"], "description": "Platforms to search"},
            "confidence_threshold": {"type": "number", "default": 0.8, "description": "Minimum confidence for matches"}
        },
        "required": ["name"]
    }
)

LINKEDIN_ANALYSIS_TOOL = MCPTool(
    name="analyze_linkedin_profile",
    description="Analyze LinkedIn profile for professional insights",
    input_schema={
        "type": "object",
        "properties": {
            "profile_url": {"type": "string", "description": "LinkedIn profile URL"},
            "profile_data": {"type": "object", "description": "Pre-fetched profile data"},
            "analysis_depth": {"type": "string", "enum": ["basic", "detailed", "comprehensive"], "default": "detailed"}
        }
    }
)

SOCIAL_ACTIVITY_TOOL = MCPTool(
    name="analyze_social_activity",
    description="Analyze social media activity patterns",
    input_schema={
        "type": "object",
        "properties": {
            "profiles": {"type": "array", "items": {"type": "object"}, "description": "Social profiles to analyze"},
            "timeframe_days": {"type": "integer", "default": 90, "description": "Analysis timeframe in days"},
            "activity_types": {"type": "array", "items": {"type": "string"}, "default": ["posts", "comments", "likes"]}
        },
        "required": ["profiles"]
    }
)

# Compatibility Scoring Tools
COMPATIBILITY_SCORING_TOOL = MCPTool(
    name="calculate_compatibility",
    description="Calculate compatibility score between two profiles",
    input_schema={
        "type": "object",
        "properties": {
            "user_profile": {"type": "object", "description": "User's profile data"},
            "candidate_profile": {"type": "object", "description": "Candidate's profile data"},
            "weights": {"type": "object", "description": "Scoring weights for different factors"},
            "scoring_model": {"type": "string", "default": "comprehensive", "description": "Scoring model to use"}
        },
        "required": ["user_profile", "candidate_profile"]
    }
)

BATCH_SCORING_TOOL = MCPTool(
    name="batch_calculate_compatibility",
    description="Calculate compatibility scores for multiple candidates",
    input_schema={
        "type": "object",
        "properties": {
            "user_profile": {"type": "object", "description": "User's profile data"},
            "candidate_profiles": {"type": "array", "items": {"type": "object"}, "description": "List of candidate profiles"},
            "max_candidates": {"type": "integer", "default": 50, "description": "Maximum candidates to score"},
            "parallel_processing": {"type": "boolean", "default": True, "description": "Enable parallel processing"}
        },
        "required": ["user_profile", "candidate_profiles"]
    }
)

FACTOR_ANALYSIS_TOOL = MCPTool(
    name="analyze_compatibility_factors",
    description="Analyze individual compatibility factors in detail",
    input_schema={
        "type": "object",
        "properties": {
            "user_profile": {"type": "object", "description": "User's profile data"},
            "candidate_profile": {"type": "object", "description": "Candidate's profile data"},
            "factors": {"type": "array", "items": {"type": "string"}, "description": "Specific factors to analyze"}
        },
        "required": ["user_profile", "candidate_profile"]
    }
)

# Summary Generation Tools
PROFILE_SUMMARY_TOOL = MCPTool(
    name="generate_profile_summary",
    description="Generate concise profile summary",
    input_schema={
        "type": "object",
        "properties": {
            "profile_data": {"type": "object", "description": "Complete profile data"},
            "summary_type": {"type": "string", "enum": ["brief", "detailed", "highlights"], "default": "detailed"},
            "focus_areas": {"type": "array", "items": {"type": "string"}, "description": "Areas to focus on"},
            "max_length": {"type": "integer", "default": 200, "description": "Maximum summary length in words"}
        },
        "required": ["profile_data"]
    }
)

MATCH_EXPLANATION_TOOL = MCPTool(
    name="generate_match_explanation",
    description="Generate explanation for why profiles are compatible",
    input_schema={
        "type": "object",
        "properties": {
            "compatibility_data": {"type": "object", "description": "Compatibility analysis results"},
            "explanation_style": {"type": "string", "enum": ["casual", "formal", "detailed"], "default": "casual"},
            "include_concerns": {"type": "boolean", "default": True, "description": "Include potential concerns"}
        },
        "required": ["compatibility_data"]
    }
)

# Resources
BIO_SCHEMAS_RESOURCE = MCPResource(
    uri="file:///schemas/bio_data_schema.json",
    name="bio_data_schema",
    description="JSON schema for standardized bio data format",
    mime_type="application/json"
)

SCORING_WEIGHTS_RESOURCE = MCPResource(
    uri="file:///config/scoring_weights.yaml",
    name="scoring_weights",
    description="Default weights for compatibility scoring",
    mime_type="application/yaml"
)

SOCIAL_PLATFORMS_RESOURCE = MCPResource(
    uri="file:///config/social_platforms.json",
    name="social_platforms",
    description="Configuration for social media platforms",
    mime_type="application/json"
)

# Prompts
QUERY_ANALYSIS_PROMPT = MCPPrompt(
    name="analyze_user_query",
    description="Analyze and structure user query for bio matching",
    arguments=[
        {"name": "user_query", "description": "Raw user query", "required": True},
        {"name": "user_context", "description": "Additional user context", "required": False}
    ]
)

BIO_COMPARISON_PROMPT = MCPPrompt(
    name="compare_bio_profiles",
    description="Compare two bio profiles for compatibility",
    arguments=[
        {"name": "profile1", "description": "First profile", "required": True},
        {"name": "profile2", "description": "Second profile", "required": True},
        {"name": "focus_areas", "description": "Areas to focus on", "required": False}
    ]
)

PROFESSIONAL_ANALYSIS_PROMPT = MCPPrompt(
    name="analyze_professional_compatibility",
    description="Analyze professional compatibility between profiles",
    arguments=[
        {"name": "user_career", "description": "User's career information", "required": True},
        {"name": "candidate_career", "description": "Candidate's career information", "required": True}
    ]
)

INTEREST_MATCHING_PROMPT = MCPPrompt(
    name="match_interests_hobbies",
    description="Analyze interest and hobby compatibility",
    arguments=[
        {"name": "user_interests", "description": "User's interests and hobbies", "required": True},
        {"name": "candidate_interests", "description": "Candidate's interests and hobbies", "required": True}
    ]
)


def get_all_tools() -> List[MCPTool]:
    """Get all available MCP tools."""
    return [
        BIO_EXTRACTION_TOOL,
        BIO_STANDARDIZATION_TOOL,
        BIO_VALIDATION_TOOL,
        VECTOR_SEARCH_TOOL,
        VECTOR_INDEX_TOOL,
        VECTOR_EMBEDDING_TOOL,
        SOCIAL_PROFILE_SEARCH_TOOL,
        LINKEDIN_ANALYSIS_TOOL,
        SOCIAL_ACTIVITY_TOOL,
        COMPATIBILITY_SCORING_TOOL,
        BATCH_SCORING_TOOL,
        FACTOR_ANALYSIS_TOOL,
        PROFILE_SUMMARY_TOOL,
        MATCH_EXPLANATION_TOOL
    ]


def get_all_resources() -> List[MCPResource]:
    """Get all available MCP resources."""
    return [
        BIO_SCHEMAS_RESOURCE,
        SCORING_WEIGHTS_RESOURCE,
        SOCIAL_PLATFORMS_RESOURCE
    ]


def get_all_prompts() -> List[MCPPrompt]:
    """Get all available MCP prompts."""
    return [
        QUERY_ANALYSIS_PROMPT,
        BIO_COMPARISON_PROMPT,
        PROFESSIONAL_ANALYSIS_PROMPT,
        INTEREST_MATCHING_PROMPT
    ]


def get_tools_by_category(category: str) -> List[MCPTool]:
    """Get tools by category."""
    categories = {
        "bio_processing": [BIO_EXTRACTION_TOOL, BIO_STANDARDIZATION_TOOL, BIO_VALIDATION_TOOL],
        "vector_search": [VECTOR_SEARCH_TOOL, VECTOR_INDEX_TOOL, VECTOR_EMBEDDING_TOOL],
        "social_analysis": [SOCIAL_PROFILE_SEARCH_TOOL, LINKEDIN_ANALYSIS_TOOL, SOCIAL_ACTIVITY_TOOL],
        "compatibility": [COMPATIBILITY_SCORING_TOOL, BATCH_SCORING_TOOL, FACTOR_ANALYSIS_TOOL],
        "summary": [PROFILE_SUMMARY_TOOL, MATCH_EXPLANATION_TOOL]
    }
    return categories.get(category, [])
