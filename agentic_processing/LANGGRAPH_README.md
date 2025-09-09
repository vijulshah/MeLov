# Enhanced Agentic Bio Matching System

## LangGraph + MCP Integration

This enhanced version of the agentic bio matching system integrates **LangGraph** for advanced workflow orchestration and **Model Context Protocol (MCP)** for standardized tool and resource management.

## 🚀 New Features

### LangGraph Workflow Orchestration
- **State-based workflow management** with automatic checkpointing
- **Conditional routing** and parallel execution
- **Visual workflow graphs** for debugging and monitoring
- **Fault tolerance** with automatic retries and error handling
- **Streaming execution** with real-time progress tracking

### Model Context Protocol (MCP) Integration
- **Standardized tool interface** for bio processing, vector search, and social analysis
- **Resource management** for schemas, configurations, and data
- **Prompt templates** for consistent AI interactions
- **Server-based architecture** for scalable tool execution
- **Cross-platform compatibility** with standard MCP specification

### Hybrid Orchestration Modes
- **Legacy Mode**: Traditional sequential agent execution
- **LangGraph Mode**: Advanced graph-based workflow execution
- **Hybrid Mode**: LangGraph with legacy fallback for maximum reliability

## 📋 Prerequisites

```bash
# Install enhanced dependencies
pip install -r requirements.txt

# Additional dependencies for LangGraph and MCP
pip install langgraph>=0.0.60
pip install langchain>=0.1.0
pip install langchain-core>=0.1.0
pip install mcp>=0.5.0
pip install fastapi>=0.104.0
pip install uvicorn>=0.24.0
```

## 🛠️ Quick Start

### Basic Usage

```python
from agentic_processing.enhanced_main import EnhancedAgenticBioMatcher
from agentic_processing.workflow_orchestrator import WorkflowOrchestrationMode

# Initialize with hybrid mode (recommended)
bio_matcher = EnhancedAgenticBioMatcher(
    orchestration_mode=WorkflowOrchestrationMode.HYBRID,
    enable_mcp=True
)

await bio_matcher.initialize()

# Execute bio matching workflow
result = await bio_matcher.find_matches(
    user_query="Looking for a tech professional who enjoys outdoor activities",
    user_bio={
        "personal_info": {"name": "John", "age": 30, "location": "SF"},
        "interests": {"hobbies": ["hiking", "tech", "reading"]}
    },
    include_social_search=True,
    max_results=10
)

await bio_matcher.close()
```

### Running Demos

```bash
# Run example workflow with all features
python -m agentic_processing.enhanced_main example

# Compare different orchestration modes
python -m agentic_processing.enhanced_main compare

# Interactive demo
python -m agentic_processing.enhanced_main demo
```

## 🏗️ Architecture Overview

### LangGraph Workflow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Initialize      │───▶│ Query Processing │───▶│ Bio Matching    │
│ Workflow        │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                              ┌─────────────────┐      │
                              │ Route Workflow  │◀─────┘
                              │                 │
                              └─────────┬───────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │ Social Finding  │    │ Compatibility   │    │ End Workflow    │
        │                 │    │ Scoring         │    │                 │
        └─────────┬───────┘    │                 │    └─────────────────┘
                  │            └─────────┬───────┘
                  ▼                      ▲
        ┌─────────────────┐              │
        │ Profile Analysis│──────────────┘
        │                 │
        └─────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Summary         │
        │ Generation      │
        └─────────┬───────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Finalize        │
        │ Workflow        │
        └─────────────────┘
```

### MCP Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────│
│  │Bio Processing│  │Social       │  │Vector       │  │Compat│
│  │Server        │  │Analysis     │  │Search       │  │Score │
│  │:8001         │  │Server:8002  │  │Server:8003  │  │:8004 │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────│
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Enhanced Configuration File

The system uses `langgraph_agentic_config.yaml` for comprehensive configuration:

```yaml
# LangGraph Configuration
langgraph:
  enable_checkpointing: true
  checkpoint_storage: "memory"
  max_workflow_steps: 50
  step_timeout_seconds: 300

# MCP Configuration
mcp:
  enabled: true
  servers:
    bio_processing:
      endpoint: "http://localhost:8001/mcp"
      capabilities: ["tools", "resources", "prompts"]

# Workflow Configuration
workflow:
  orchestration_mode: "hybrid"
  enable_parallel_processing: true
  max_concurrent_agents: 3
```

### Orchestration Modes

1. **Legacy Mode**
   - Traditional sequential execution
   - No LangGraph or MCP dependencies
   - Maximum compatibility

2. **LangGraph Mode**
   - Advanced workflow orchestration
   - State management and checkpointing
   - Requires LangGraph installation

3. **Hybrid Mode** (Recommended)
   - Attempts LangGraph execution
   - Falls back to legacy on failure
   - Best reliability and features

## 🛠️ MCP Server Setup

### Starting MCP Servers

```bash
# Bio Processing Server
python -m agentic_processing.mcp.mcp_server

# Or use the convenience script (when available)
python scripts/start_mcp_servers.py
```

### Available MCP Tools

#### Bio Processing Tools
- `extract_bio_data`: Extract structured data from raw text
- `standardize_bio_data`: Standardize bio data format
- `validate_bio_data`: Validate data completeness

#### Vector Search Tools
- `search_similar_profiles`: Vector similarity search
- `index_bio_profile`: Index new profiles
- `generate_embeddings`: Generate text embeddings

#### Social Analysis Tools
- `search_social_profiles`: Find social media profiles
- `analyze_linkedin_profile`: Analyze LinkedIn profiles
- `analyze_social_activity`: Analyze activity patterns

#### Compatibility Tools
- `calculate_compatibility`: Score compatibility
- `batch_calculate_compatibility`: Batch scoring
- `factor_analysis`: Detailed factor analysis

## 📊 Monitoring and Debugging

### Workflow Visualization

LangGraph provides built-in visualization for workflow execution:

```python
# Enable visualization in config
langgraph:
  enable_visualization: true
  enable_tracing: true

# Access workflow graph
orchestrator = LangGraphWorkflowOrchestrator()
graph = orchestrator.compiled_workflow
graph.get_graph().draw_mermaid()  # Generate Mermaid diagram
```

### Performance Monitoring

```python
# Get workflow status
status = await orchestrator.get_workflow_status(execution_id)

# Get agent performance metrics
for agent_name, agent in orchestrator.agents.items():
    metrics = agent.get_performance_metrics()
    print(f"{agent_name}: {metrics}")

# Get orchestrator capabilities
capabilities = orchestrator.get_capabilities()
```

## 🔍 Troubleshooting

### Common Issues

1. **LangGraph Import Errors**
   ```bash
   pip install langgraph>=0.0.60 langchain>=0.1.0
   ```

2. **MCP Connection Failures**
   - Check MCP server endpoints in config
   - Ensure MCP servers are running
   - Verify network connectivity

3. **Workflow Execution Failures**
   - Check logs for detailed error messages
   - Use hybrid mode for automatic fallback
   - Verify agent configurations

### Debug Mode

```python
# Enable debug mode in configuration
development:
  enable_debug_mode: true
  save_intermediate_results: true
  enable_workflow_visualization: true
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_langgraph_orchestrator.py
python -m pytest tests/test_mcp_integration.py
```

### Integration Tests

```bash
# Test workflow execution
python -m agentic_processing.enhanced_main compare

# Test MCP servers
python tests/test_mcp_servers.py
```

## 📈 Performance Optimization

### Recommendations

1. **Use Hybrid Mode**: Best balance of features and reliability
2. **Enable Caching**: Reduces redundant computations
3. **Parallel Processing**: Enable for faster execution
4. **Resource Limits**: Configure appropriate memory/CPU limits
5. **MCP Connection Pooling**: Reuse connections for better performance

### Performance Metrics

The system tracks comprehensive performance metrics:
- Workflow execution time
- Agent processing time
- MCP tool call latency
- Memory and CPU usage
- Success/failure rates

## 🤝 Contributing

When contributing to the enhanced system:

1. Follow the MCP specification for new tools
2. Add LangGraph nodes for new workflow steps
3. Update configuration schemas
4. Add comprehensive tests
5. Update documentation

## 📄 License

This enhanced version maintains the same license as the original project.

## 🔗 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Original Bio Matching System README](./README.md)
