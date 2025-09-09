#!/usr/bin/env python3
"""
Setup script for enhanced agentic processing with LangGraph and MCP support.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n🚀 Installing Enhanced Agentic Processing Dependencies")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Update pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Updating pip"):
        return False
    
    # Install base dependencies
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing base dependencies"):
        return False
    
    # Install additional LangGraph and MCP dependencies
    additional_packages = [
        "langgraph>=0.0.60",
        "langchain>=0.1.0", 
        "langchain-core>=0.1.0",
        "langchain-experimental>=0.0.50",
        "langsmith>=0.0.80",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "httpx>=0.24.0",
        "websockets>=11.0.0"
    ]
    
    for package in additional_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}")
    
    return True


def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    directories = [
        "logs",
        "data/test_data", 
        "config/mcp_servers",
        "scripts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_mcp_server_script():
    """Create MCP server startup script."""
    script_content = '''#!/usr/bin/env python3
"""
Start MCP servers for the agentic bio matching system.
"""
import asyncio
import subprocess
import sys
import time
from pathlib import Path

servers = [
    {
        "name": "bio_processing",
        "port": 8001,
        "module": "agentic_processing.mcp.mcp_server"
    },
    # Add more servers as they become available
    # {
    #     "name": "social_analysis", 
    #     "port": 8002,
    #     "module": "agentic_processing.mcp.social_server"
    # },
    # {
    #     "name": "vector_search",
    #     "port": 8003, 
    #     "module": "agentic_processing.mcp.vector_server"
    # },
    # {
    #     "name": "compatibility_scoring",
    #     "port": 8004,
    #     "module": "agentic_processing.mcp.compatibility_server"
    # }
]

def start_server(server_config):
    """Start a single MCP server."""
    print(f"Starting {server_config['name']} server on port {server_config['port']}...")
    
    cmd = [
        sys.executable, "-m", server_config["module"],
        "--port", str(server_config["port"])
    ]
    
    try:
        process = subprocess.Popen(cmd)
        time.sleep(2)  # Give server time to start
        
        if process.poll() is None:
            print(f"✅ {server_config['name']} server started (PID: {process.pid})")
            return process
        else:
            print(f"❌ {server_config['name']} server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {server_config['name']}: {e}")
        return None

def main():
    """Main function to start all MCP servers."""
    print("🚀 Starting MCP Servers for Agentic Bio Matching")
    print("=" * 50)
    
    processes = []
    
    for server_config in servers:
        process = start_server(server_config)
        if process:
            processes.append((server_config["name"], process))
    
    if not processes:
        print("❌ No servers started successfully")
        return
    
    print(f"\\n✅ Started {len(processes)} MCP servers")
    print("\\n📋 Running servers:")
    for name, process in processes:
        print(f"   • {name} (PID: {process.pid})")
    
    print("\\n🔧 Press Ctrl+C to stop all servers")
    
    try:
        # Keep script running
        while True:
            time.sleep(1)
            
            # Check if any processes have died
            for name, process in processes[:]:
                if process.poll() is not None:
                    print(f"⚠️  {name} server stopped (PID: {process.pid})")
                    processes.remove((name, process))
            
            if not processes:
                print("❌ All servers have stopped")
                break
                
    except KeyboardInterrupt:
        print("\\n🛑 Stopping all servers...")
        
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {name} server")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔨 Force killed {name} server")
            except Exception as e:
                print(f"⚠️  Error stopping {name}: {e}")
        
        print("✅ All servers stopped")

if __name__ == "__main__":
    main()
'''
    
    script_path = Path("scripts/start_mcp_servers.py")
    script_path.parent.mkdir(exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable on Unix systems
    if os.name != 'nt':
        os.chmod(script_path, 0o755)
    
    print(f"✅ Created MCP server script: {script_path}")


def verify_installation():
    """Verify the installation by testing imports."""
    print("\n🔍 Verifying installation...")
    
    test_imports = [
        ("agentic_processing", "Basic agentic processing"),
        ("agentic_processing.workflow_orchestrator", "Workflow orchestrator"),
        ("agentic_processing.models.agentic_models", "Data models"),
    ]
    
    # Test enhanced features
    enhanced_imports = [
        ("langgraph", "LangGraph"),
        ("langchain", "LangChain"),
        ("fastapi", "FastAPI"),
        ("httpx", "HTTPX")
    ]
    
    all_success = True
    
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description} - OK")
        except ImportError as e:
            print(f"❌ {description} - FAILED: {e}")
            all_success = False
    
    print("\n🔧 Testing enhanced features:")
    enhanced_available = 0
    
    for module, description in enhanced_imports:
        try:
            __import__(module)
            print(f"✅ {description} - Available")
            enhanced_available += 1
        except ImportError:
            print(f"⚠️  {description} - Not available")
    
    try:
        from agentic_processing.enhanced_main import EnhancedAgenticBioMatcher
        print("✅ Enhanced features - Available")
    except ImportError:
        print("⚠️  Enhanced features - Limited (some dependencies missing)")
    
    return all_success, enhanced_available


def main():
    """Main setup function."""
    print("🚀 Enhanced Agentic Processing Setup")
    print("=" * 50)
    print("This script will install LangGraph and MCP dependencies")
    print("for the enhanced agentic bio matching system.")
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the errors above.")
        return 1
    
    # Setup directories
    setup_directories()
    
    # Create helper scripts
    create_mcp_server_script()
    
    # Verify installation
    success, enhanced_count = verify_installation()
    
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    if success:
        print("✅ Core installation: SUCCESS")
    else:
        print("❌ Core installation: FAILED")
    
    print(f"🔧 Enhanced features: {enhanced_count}/4 available")
    
    if enhanced_count >= 3:
        print("✅ Enhanced features: READY")
        print("\n🚀 You can now use:")
        print("   • LangGraph workflow orchestration")
        print("   • MCP tool integration") 
        print("   • Hybrid orchestration modes")
        print("   • Enhanced monitoring and debugging")
    else:
        print("⚠️  Enhanced features: LIMITED")
        print("   Some dependencies missing - legacy mode available")
    
    print("\n📖 Next steps:")
    print("1. Review the configuration in config/langgraph_agentic_config.yaml")
    print("2. Start MCP servers: python scripts/start_mcp_servers.py")
    print("3. Run demo: python -m agentic_processing.enhanced_main demo")
    print("4. See LANGGRAPH_README.md for detailed documentation")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
