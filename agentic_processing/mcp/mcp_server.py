"""
MCP Server implementation for bio processing tools.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from ..mcp.mcp_tools import get_all_tools, get_all_resources, get_all_prompts
from ..data_processing.extraction.bio_extractor import BioExtractor
from ..data_vector_store.faiss_store import FAISSVectorStore


class MCPRequest(BaseModel):
    """MCP request model."""
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP response model."""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None


class BioProcessingMCPServer:
    """MCP Server for bio processing tools."""
    
    def __init__(self):
        """Initialize the MCP server."""
        self.app = FastAPI(title="Bio Processing MCP Server")
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.bio_extractor = None
        self.vector_store = None
        
        # Setup routes
        self._setup_routes()
    
    async def initialize(self):
        """Initialize server components."""
        try:
            # Initialize bio extractor
            self.bio_extractor = BioExtractor()
            
            # Initialize vector store
            self.vector_store = FAISSVectorStore()
            await self.vector_store.initialize()
            
            self.logger.info("MCP server initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes for MCP endpoints."""
        
        @self.app.post("/tools/list")
        async def list_tools(request: MCPRequest) -> MCPResponse:
            """List available tools."""
            try:
                tools = get_all_tools()
                tool_list = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema
                    }
                    for tool in tools
                ]
                
                return MCPResponse(result={"tools": tool_list})
                
            except Exception as e:
                return MCPResponse(error={"code": "INTERNAL_ERROR", "message": str(e)})
        
        @self.app.post("/tools/call")
        async def call_tool(request: MCPRequest) -> MCPResponse:
            """Call a specific tool."""
            try:
                if not request.params:
                    raise ValueError("Missing tool parameters")
                
                tool_name = request.params.get("name")
                arguments = request.params.get("arguments", {})
                
                result = await self._execute_tool(tool_name, arguments)
                
                return MCPResponse(result={"content": result})
                
            except Exception as e:
                return MCPResponse(error={"code": "TOOL_ERROR", "message": str(e)})
        
        @self.app.post("/resources/list")
        async def list_resources(request: MCPRequest) -> MCPResponse:
            """List available resources."""
            try:
                resources = get_all_resources()
                resource_list = [
                    {
                        "uri": resource.uri,
                        "name": resource.name,
                        "description": resource.description,
                        "mime_type": resource.mime_type
                    }
                    for resource in resources
                ]
                
                return MCPResponse(result={"resources": resource_list})
                
            except Exception as e:
                return MCPResponse(error={"code": "INTERNAL_ERROR", "message": str(e)})
        
        @self.app.post("/resources/read")
        async def read_resource(request: MCPRequest) -> MCPResponse:
            """Read a specific resource."""
            try:
                if not request.params:
                    raise ValueError("Missing resource parameters")
                
                uri = request.params.get("uri")
                content = await self._read_resource(uri)
                
                return MCPResponse(result={
                    "contents": [{"uri": uri, "text": content}]
                })
                
            except Exception as e:
                return MCPResponse(error={"code": "RESOURCE_ERROR", "message": str(e)})
        
        @self.app.post("/prompts/list")
        async def list_prompts(request: MCPRequest) -> MCPResponse:
            """List available prompts."""
            try:
                prompts = get_all_prompts()
                prompt_list = [
                    {
                        "name": prompt.name,
                        "description": prompt.description,
                        "arguments": prompt.arguments
                    }
                    for prompt in prompts
                ]
                
                return MCPResponse(result={"prompts": prompt_list})
                
            except Exception as e:
                return MCPResponse(error={"code": "INTERNAL_ERROR", "message": str(e)})
        
        @self.app.post("/prompts/get")
        async def get_prompt(request: MCPRequest) -> MCPResponse:
            """Get a specific prompt."""
            try:
                if not request.params:
                    raise ValueError("Missing prompt parameters")
                
                prompt_name = request.params.get("name")
                arguments = request.params.get("arguments", {})
                
                content = await self._get_prompt(prompt_name, arguments)
                
                return MCPResponse(result={
                    "messages": [
                        {
                            "role": "user",
                            "content": {"type": "text", "text": content}
                        }
                    ]
                })
                
            except Exception as e:
                return MCPResponse(error={"code": "PROMPT_ERROR", "message": str(e)})
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a specific tool."""
        
        if tool_name == "extract_bio_data":
            return await self._extract_bio_data(arguments)
        elif tool_name == "standardize_bio_data":
            return await self._standardize_bio_data(arguments)
        elif tool_name == "validate_bio_data":
            return await self._validate_bio_data(arguments)
        elif tool_name == "search_similar_profiles":
            return await self._search_similar_profiles(arguments)
        elif tool_name == "index_bio_profile":
            return await self._index_bio_profile(arguments)
        elif tool_name == "generate_embeddings":
            return await self._generate_embeddings(arguments)
        elif tool_name == "generate_profile_summary":
            return await self._generate_profile_summary(arguments)
        elif tool_name == "analyze_user_query":
            return await self._analyze_user_query(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _extract_bio_data(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract bio data from raw text."""
        raw_text = arguments.get("raw_text", "")
        source = arguments.get("source", "unknown")
        format_type = arguments.get("format", "matrimony")
        
        if not self.bio_extractor:
            raise RuntimeError("Bio extractor not initialized")
        
        try:
            extracted_data = await self.bio_extractor.extract_bio_data(
                raw_text, source, format_type
            )
            
            return [{
                "type": "text",
                "text": json.dumps(extracted_data, indent=2)
            }]
            
        except Exception as e:
            raise RuntimeError(f"Bio extraction failed: {e}")
    
    async def _standardize_bio_data(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Standardize bio data format."""
        bio_data = arguments.get("bio_data", {})
        target_schema = arguments.get("target_schema", "v1.0")
        
        # Basic standardization logic
        standardized_data = {
            "version": target_schema,
            "personal_info": bio_data.get("personal_info", {}),
            "education": bio_data.get("education", {}),
            "professional": bio_data.get("professional", {}),
            "interests": bio_data.get("interests", {}),
            "lifestyle": bio_data.get("lifestyle", {}),
            "relationship": bio_data.get("relationship", {}),
            "standardized_at": asyncio.get_event_loop().time()
        }
        
        return [{
            "type": "text",
            "text": json.dumps(standardized_data, indent=2)
        }]
    
    async def _validate_bio_data(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate bio data completeness."""
        bio_data = arguments.get("bio_data", {})
        strict_mode = arguments.get("strict_mode", False)
        
        # Basic validation
        required_fields = ["personal_info"]
        if strict_mode:
            required_fields.extend(["education", "professional"])
        
        validation_result = {
            "valid": True,
            "missing_fields": [],
            "warnings": [],
            "completeness_score": 0.0
        }
        
        total_sections = 6
        completed_sections = 0
        
        for field in ["personal_info", "education", "professional", "interests", "lifestyle", "relationship"]:
            if field in bio_data and bio_data[field]:
                completed_sections += 1
            elif field in required_fields:
                validation_result["valid"] = False
                validation_result["missing_fields"].append(field)
        
        validation_result["completeness_score"] = completed_sections / total_sections
        
        return [{
            "type": "text",
            "text": json.dumps(validation_result, indent=2)
        }]
    
    async def _search_similar_profiles(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for similar profiles."""
        query_text = arguments.get("query_text", "")
        top_k = arguments.get("top_k", 10)
        similarity_threshold = arguments.get("similarity_threshold", 0.7)
        filters = arguments.get("filters", {})
        
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        try:
            results = await self.vector_store.search(
                query_text=query_text,
                top_k=top_k,
                filters=filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.get("similarity_score", 0) >= similarity_threshold
            ]
            
            return [{
                "type": "text",
                "text": json.dumps({
                    "matches": filtered_results,
                    "total_found": len(filtered_results),
                    "query": query_text
                }, indent=2)
            }]
            
        except Exception as e:
            raise RuntimeError(f"Vector search failed: {e}")
    
    async def _index_bio_profile(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Index a bio profile."""
        bio_data = arguments.get("bio_data", {})
        metadata = arguments.get("metadata", {})
        
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        try:
            profile_id = await self.vector_store.add_profile(bio_data, metadata)
            
            return [{
                "type": "text",
                "text": json.dumps({
                    "profile_id": profile_id,
                    "status": "indexed",
                    "message": "Profile successfully indexed"
                }, indent=2)
            }]
            
        except Exception as e:
            raise RuntimeError(f"Profile indexing failed: {e}")
    
    async def _generate_embeddings(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate embeddings for text."""
        text = arguments.get("text", "")
        model = arguments.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        try:
            embeddings = await self.vector_store.embedding_manager.get_embedding(text)
            
            return [{
                "type": "text",
                "text": json.dumps({
                    "embeddings": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                    "model": model,
                    "text_length": len(text)
                }, indent=2)
            }]
            
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    async def _generate_profile_summary(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate profile summary."""
        user_bio_data = arguments.get("user_bio_data", {})
        compatibility_scores = arguments.get("compatibility_scores", [])
        top_k = arguments.get("top_k", 10)
        summary_type = arguments.get("summary_type", "detailed")
        
        # Sort by compatibility score and take top k
        sorted_scores = sorted(
            compatibility_scores,
            key=lambda x: x.get("overall_score", 0),
            reverse=True
        )[:top_k]
        
        summaries = []
        for score in sorted_scores:
            summary = {
                "bio_data_id": score.get("bio_data_id"),
                "compatibility_score": score.get("overall_score", 0),
                "match_quality": score.get("match_quality", "unknown"),
                "summary": f"Match with {score.get('overall_score', 0):.1%} compatibility",
                "positive_factors": score.get("positive_factors", []),
                "potential_concerns": score.get("negative_factors", [])
            }
            summaries.append(summary)
        
        return [{
            "type": "text", 
            "text": json.dumps({
                "match_summaries": summaries,
                "total_matches": len(summaries),
                "summary_type": summary_type
            }, indent=2)
        }]
    
    async def _analyze_user_query(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze user query for requirements."""
        user_query = arguments.get("user_query", "")
        user_context = arguments.get("user_context", {})
        
        # Basic query analysis - in production this would use NLP
        analysis = {
            "query": user_query,
            "requirements": {},
            "filters": {},
            "max_matches": 20,
            "confidence_score": 0.8
        }
        
        # Extract basic requirements from query
        query_lower = user_query.lower()
        
        if "age" in query_lower:
            analysis["requirements"]["age_preference"] = "mentioned"
        if "education" in query_lower:
            analysis["requirements"]["education_preference"] = "mentioned"
        if "profession" in query_lower or "job" in query_lower:
            analysis["requirements"]["profession_preference"] = "mentioned"
        if "location" in query_lower or "city" in query_lower:
            analysis["requirements"]["location_preference"] = "mentioned"
        
        return [{
            "type": "text",
            "text": json.dumps(analysis, indent=2)
        }]
    
    async def _read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        # Basic resource reading - in production this would handle various URIs
        if "bio_data_schema" in uri:
            return json.dumps({
                "type": "object",
                "properties": {
                    "personal_info": {"type": "object"},
                    "education": {"type": "object"},
                    "professional": {"type": "object"},
                    "interests": {"type": "object"},
                    "lifestyle": {"type": "object"},
                    "relationship": {"type": "object"}
                }
            }, indent=2)
        else:
            raise ValueError(f"Resource not found: {uri}")
    
    async def _get_prompt(self, prompt_name: str, arguments: Dict[str, Any]) -> str:
        """Get a prompt template."""
        if prompt_name == "analyze_user_query":
            user_query = arguments.get("user_query", "[USER_QUERY]")
            return f"""
            Analyze the following user query for bio matching requirements:
            
            Query: {user_query}
            
            Extract:
            1. Age preferences
            2. Education requirements
            3. Professional preferences
            4. Location preferences
            5. Interest areas
            6. Lifestyle preferences
            
            Provide structured output with confidence scores.
            """
        elif prompt_name == "compare_bio_profiles":
            return """
            Compare the following two bio profiles for compatibility:
            
            Profile 1: {profile1}
            Profile 2: {profile2}
            
            Analyze compatibility in:
            1. Personal values
            2. Educational background
            3. Professional goals
            4. Interests and hobbies
            5. Lifestyle preferences
            
            Provide compatibility score and explanation.
            """
        else:
            raise ValueError(f"Prompt not found: {prompt_name}")
    
    def run(self, host: str = "localhost", port: int = 8001):
        """Run the MCP server."""
        uvicorn.run(self.app, host=host, port=port)


async def main():
    """Main function to run the MCP server."""
    server = BioProcessingMCPServer()
    await server.initialize()
    server.run()


if __name__ == "__main__":
    asyncio.run(main())
