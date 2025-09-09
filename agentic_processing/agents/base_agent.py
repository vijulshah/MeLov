"""
Base agent class and LLM interface for the multi-agent bio matching system.
"""
import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator
import json
import uuid

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..models.agentic_models import (
    AgentResponse, AgentRole, LLMConfig, AgentTask, ModelProvider
)
from ..prompts.prompt_manager import prompt_manager


class LLMInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text completion."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text completion."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class HuggingFaceLLM(LLMInterface):
    """Hugging Face model interface."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize Hugging Face LLM.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the model and tokenizer."""
        try:
            self.logger.info(f"Loading model: {self.config.model_name}")
            
            # Login to Hugging Face if API key provided
            if self.config.api_key:
                login(token=self.config.api_key)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            self.logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text completion."""
        try:
            # Format prompt for instruction models
            formatted_prompt = self._format_prompt(prompt)
            
            # Generate response
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                **kwargs
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Clean up the response
            return self._clean_response(generated_text)
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text completion."""
        # For simplicity, we'll simulate streaming by yielding chunks
        # In a production system, you'd implement proper streaming
        full_response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        
        # Yield response in chunks
        chunk_size = 10
        words = full_response.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            yield chunk + " "
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for the specific model."""
        # Different models may need different prompt formats
        if "Phi-3" in self.config.model_name:
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif "Llama-3" in self.config.model_name:
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "Mistral" in self.config.model_name:
            return f"[INST] {prompt} [/INST]"
        elif "falcon" in self.config.model_name:
            return f"User: {prompt}\nAssistant: "
        else:
            # Generic format
            return f"Human: {prompt}\n\nAssistant: "
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response."""
        # Remove common artifacts
        response = response.strip()
        
        # Remove repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() and line not in cleaned_lines[-3:]:  # Avoid immediate repetition
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "max_tokens": self.config.max_tokens,
            "device": next(self.model.parameters()).device.type if self.model else "unknown",
            "parameters": self.model.num_parameters() if hasattr(self.model, 'num_parameters') else "unknown"
        }


class MockLLM(LLMInterface):
    """Mock LLM for testing without actual model loading."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate mock response."""
        await asyncio.sleep(0.5)  # Simulate processing time
        return f"Mock response for prompt: {prompt[:50]}... (Generated by {self.config.model_name})"
    
    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate mock streaming response."""
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        words = response.split()
        
        for word in words:
            yield word + " "
            await asyncio.sleep(0.1)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model_name": self.config.model_name,
            "provider": "mock",
            "max_tokens": self.config.max_tokens,
            "device": "cpu",
            "parameters": "mock"
        }


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create_llm(config: LLMConfig, use_mock: bool = False) -> LLMInterface:
        """
        Create LLM instance based on configuration.
        
        Args:
            config: LLM configuration
            use_mock: Whether to use mock LLM for testing
            
        Returns:
            LLM interface instance
        """
        if use_mock or not TRANSFORMERS_AVAILABLE:
            return MockLLM(config)
        
        if config.provider == ModelProvider.HUGGINGFACE:
            return HuggingFaceLLM(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(
        self, 
        name: str,
        role: AgentRole,
        llm_config: LLMConfig,
        use_mock_llm: bool = False
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            role: Agent role
            llm_config: LLM configuration
            use_mock_llm: Whether to use mock LLM
        """
        self.name = name
        self.role = role
        self.llm_config = llm_config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Initialize LLM
        self.llm = LLMFactory.create_llm(llm_config, use_mock_llm)
        
        # Performance tracking
        self.task_count = 0
        self.total_processing_time = 0.0
        self.success_count = 0
        self.error_count = 0
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> AgentResponse:
        """
        Process a task assigned to this agent.
        
        Args:
            task: Task to process
            
        Returns:
            Agent response
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent from prompt manager."""
        agent_name = self.name.lower().replace("agent", "").replace("_", "")
        return prompt_manager.get_agent_prompt(agent_name)
    
    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """
        Execute a task with error handling and metrics.
        
        Args:
            task: Task to execute
            
        Returns:
            Agent response
        """
        start_time = time.time()
        self.task_count += 1
        
        try:
            self.logger.info(f"Executing task {task.task_id} of type {task.task_type}")
            
            # Process the task
            response = await self.process_task(task)
            
            # Update success metrics
            self.success_count += 1
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            # Update response metadata
            response.processing_time_ms = processing_time
            response.model_used = self.llm_config.model_name
            
            self.logger.info(f"Task {task.task_id} completed successfully in {processing_time:.2f}ms")
            
            return response
            
        except Exception as e:
            # Update error metrics
            self.error_count += 1
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            # Return error response
            return AgentResponse(
                agent_name=self.name,
                agent_role=self.role,
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time,
                model_used=self.llm_config.model_name
            )
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate LLM response with system prompt.
        
        Args:
            prompt: User prompt
            context: Additional context
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Generated response
        """
        # Build full prompt with system prompt and context
        system_prompt = self.get_system_prompt()
        
        if context:
            context_str = f"Context: {json.dumps(context, indent=2)}\n\n"
        else:
            context_str = ""
        
        full_prompt = f"{system_prompt}\n\n{context_str}User Request: {prompt}"
        
        # Generate response
        response = await self.llm.generate(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        avg_processing_time = (
            self.total_processing_time / self.task_count 
            if self.task_count > 0 else 0.0
        )
        
        success_rate = (
            self.success_count / self.task_count 
            if self.task_count > 0 else 0.0
        )
        
        return {
            "agent_name": self.name,
            "role": self.role,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "average_processing_time_ms": avg_processing_time,
            "total_processing_time_ms": self.total_processing_time
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.role})"
    
    def __repr__(self) -> str:
        return f"BaseAgent(name='{self.name}', role='{self.role}', model='{self.llm_config.model_name}')"
