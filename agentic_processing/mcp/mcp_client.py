"""
Model Context Protocol (MCP) client for agentic workflow.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
import httpx
from pydantic import BaseModel, Field

from ..models.agentic_models import AgentTask, AgentResponse


class MCPTool(BaseModel):
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None


class MCPResource(BaseModel):
    """MCP Resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str


class MCPPrompt(BaseModel):
    """MCP Prompt template."""
    name: str
    description: str
    arguments: List[Dict[str, Any]] = Field(default_factory=list)


class MCPServer(BaseModel):
    """MCP Server configuration."""
    name: str
    endpoint: str
    capabilities: List[str] = Field(default_factory=list)
    tools: List[MCPTool] = Field(default_factory=list)
    resources: List[MCPResource] = Field(default_factory=list)
    prompts: List[MCPPrompt] = Field(default_factory=list)


class MCPClient:
    """MCP Client for managing tools, resources, and prompts."""
    
    def __init__(self, server_configs: List[MCPServer]):
        """
        Initialize MCP client.
        
        Args:
            server_configs: List of MCP server configurations
        """
        self.servers = {server.name: server for server in server_configs}
        self.client = httpx.AsyncClient(timeout=30.0)
        self.logger = logging.getLogger(__name__)
        
        # Cache for discovered capabilities
        self._tools_cache: Dict[str, MCPTool] = {}
        self._resources_cache: Dict[str, MCPResource] = {}
        self._prompts_cache: Dict[str, MCPPrompt] = {}
    
    async def initialize(self):
        """Initialize MCP client and discover capabilities."""
        self.logger.info("Initializing MCP client...")
        
        for server_name, server in self.servers.items():
            try:
                await self._discover_server_capabilities(server_name, server)
            except Exception as e:
                self.logger.error(f"Failed to discover capabilities for {server_name}: {e}")
    
    async def _discover_server_capabilities(self, server_name: str, server: MCPServer):
        """Discover capabilities from an MCP server."""
        try:
            # Discover tools
            tools_response = await self.client.post(
                f"{server.endpoint}/tools/list",
                json={"method": "tools/list"}
            )
            if tools_response.status_code == 200:
                tools_data = tools_response.json()
                for tool_data in tools_data.get("tools", []):
                    tool = MCPTool(**tool_data)
                    self._tools_cache[f"{server_name}.{tool.name}"] = tool
                    server.tools.append(tool)
            
            # Discover resources
            resources_response = await self.client.post(
                f"{server.endpoint}/resources/list",
                json={"method": "resources/list"}
            )
            if resources_response.status_code == 200:
                resources_data = resources_response.json()
                for resource_data in resources_data.get("resources", []):
                    resource = MCPResource(**resource_data)
                    self._resources_cache[f"{server_name}.{resource.name}"] = resource
                    server.resources.append(resource)
            
            # Discover prompts
            prompts_response = await self.client.post(
                f"{server.endpoint}/prompts/list",
                json={"method": "prompts/list"}
            )
            if prompts_response.status_code == 200:
                prompts_data = prompts_response.json()
                for prompt_data in prompts_data.get("prompts", []):
                    prompt = MCPPrompt(**prompt_data)
                    self._prompts_cache[f"{server_name}.{prompt.name}"] = prompt
                    server.prompts.append(prompt)
            
            self.logger.info(f"Discovered capabilities for {server_name}: "
                           f"{len(server.tools)} tools, {len(server.resources)} resources, "
                           f"{len(server.prompts)} prompts")
            
        except Exception as e:
            self.logger.error(f"Error discovering capabilities for {server_name}: {e}")
    
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        server_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            server_name: Specific server to use (if None, searches all servers)
            
        Returns:
            Tool execution result
        """
        # Find the tool
        full_tool_name = None
        target_server = None
        
        if server_name:
            full_tool_name = f"{server_name}.{tool_name}"
            target_server = self.servers.get(server_name)
        else:
            # Search all servers
            for srv_name, server in self.servers.items():
                candidate_name = f"{srv_name}.{tool_name}"
                if candidate_name in self._tools_cache:
                    full_tool_name = candidate_name
                    target_server = server
                    break
        
        if not full_tool_name or not target_server:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool = self._tools_cache[full_tool_name]
        
        try:
            # Validate arguments against schema
            self._validate_tool_arguments(tool, arguments)
            
            # Call the tool
            response = await self.client.post(
                f"{target_server.endpoint}/tools/call",
                json={
                    "method": "tools/call",
                    "params": {
                        "name": tool.name,
                        "arguments": arguments
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("content", [])
            else:
                raise Exception(f"Tool call failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            raise
    
    async def get_resource(
        self, 
        resource_uri: str,
        server_name: Optional[str] = None
    ) -> str:
        """
        Get content from an MCP resource.
        
        Args:
            resource_uri: URI of the resource
            server_name: Specific server to use
            
        Returns:
            Resource content
        """
        # Find the server that has this resource
        target_server = None
        
        if server_name:
            target_server = self.servers.get(server_name)
        else:
            # Search all servers for this resource
            for server in self.servers.values():
                for resource in server.resources:
                    if resource.uri == resource_uri:
                        target_server = server
                        break
                if target_server:
                    break
        
        if not target_server:
            raise ValueError(f"Resource '{resource_uri}' not found")
        
        try:
            response = await self.client.post(
                f"{target_server.endpoint}/resources/read",
                json={
                    "method": "resources/read",
                    "params": {
                        "uri": resource_uri
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("contents", [{}])[0].get("text", "")
            else:
                raise Exception(f"Resource read failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error reading resource {resource_uri}: {e}")
            raise
    
    async def get_prompt(
        self, 
        prompt_name: str, 
        arguments: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None
    ) -> str:
        """
        Get a prompt template from MCP server.
        
        Args:
            prompt_name: Name of the prompt
            arguments: Prompt arguments
            server_name: Specific server to use
            
        Returns:
            Rendered prompt
        """
        # Find the prompt
        full_prompt_name = None
        target_server = None
        
        if server_name:
            full_prompt_name = f"{server_name}.{prompt_name}"
            target_server = self.servers.get(server_name)
        else:
            # Search all servers
            for srv_name, server in self.servers.items():
                candidate_name = f"{srv_name}.{prompt_name}"
                if candidate_name in self._prompts_cache:
                    full_prompt_name = candidate_name
                    target_server = server
                    break
        
        if not full_prompt_name or not target_server:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        
        try:
            response = await self.client.post(
                f"{target_server.endpoint}/prompts/get",
                json={
                    "method": "prompts/get",
                    "params": {
                        "name": prompt_name,
                        "arguments": arguments or {}
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                messages = result.get("messages", [])
                # Combine all message content
                return "\n".join(msg.get("content", {}).get("text", "") for msg in messages)
            else:
                raise Exception(f"Prompt get failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error getting prompt {prompt_name}: {e}")
            raise
    
    def _validate_tool_arguments(self, tool: MCPTool, arguments: Dict[str, Any]):
        """Validate tool arguments against schema."""
        # Basic validation - in production you'd want more comprehensive validation
        required_fields = tool.input_schema.get("required", [])
        
        for field in required_fields:
            if field not in arguments:
                raise ValueError(f"Required argument '{field}' missing for tool '{tool.name}'")
    
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """List all available tools across all servers."""
        tools = []
        for tool_name, tool in self._tools_cache.items():
            server_name = tool_name.split('.')[0]
            tools.append({
                "name": tool.name,
                "full_name": tool_name,
                "server": server_name,
                "description": tool.description,
                "input_schema": tool.input_schema
            })
        return tools
    
    def list_available_resources(self) -> List[Dict[str, Any]]:
        """List all available resources across all servers."""
        resources = []
        for resource_name, resource in self._resources_cache.items():
            server_name = resource_name.split('.')[0]
            resources.append({
                "name": resource.name,
                "full_name": resource_name,
                "server": server_name,
                "uri": resource.uri,
                "description": resource.description,
                "mime_type": resource.mime_type
            })
        return resources
    
    def list_available_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts across all servers."""
        prompts = []
        for prompt_name, prompt in self._prompts_cache.items():
            server_name = prompt_name.split('.')[0]
            prompts.append({
                "name": prompt.name,
                "full_name": prompt_name,
                "server": server_name,
                "description": prompt.description,
                "arguments": prompt.arguments
            })
        return prompts
    
    async def close(self):
        """Close the MCP client."""
        await self.client.aclose()


# Predefined MCP server configurations for bio matching system
def get_default_mcp_servers() -> List[MCPServer]:
    """Get default MCP server configurations."""
    return [
        MCPServer(
            name="bio_processing",
            endpoint="http://localhost:8001/mcp",
            capabilities=["tools", "resources", "prompts"]
        ),
        MCPServer(
            name="social_analysis",
            endpoint="http://localhost:8002/mcp", 
            capabilities=["tools", "resources"]
        ),
        MCPServer(
            name="vector_search",
            endpoint="http://localhost:8003/mcp",
            capabilities=["tools", "resources"]
        ),
        MCPServer(
            name="compatibility_scoring",
            endpoint="http://localhost:8004/mcp",
            capabilities=["tools", "prompts"]
        )
    ]
