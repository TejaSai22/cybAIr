#!/usr/bin/env python3
"""
AgentChain Startup Script
Helps you get the system running quickly
"""

import os
import subprocess
import sys
import time

def check_docker():
    """Check if Docker is running"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return True
    return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_content = """# General
APP_NAME=AgentChain
ENVIRONMENT=development
DEBUG=True

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_DETECTION_TOPIC=detection-events
KAFKA_CLASSIFICATION_TOPIC=classification-events
KAFKA_TRIAGE_TOPIC=triage-events
KAFKA_MITIGATION_TOPIC=mitigation-events

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Optional: OpenAI Configuration (fallback)
OPENAI_API_KEY=your_openai_api_key_here

# RAG/Embedding
FAISS_INDEX_PATH=./data/faiss.index
PINECONE_API_KEY=
PINECONE_ENV=
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("⚠️  Please ensure Ollama is running on http://localhost:11434")
        print("⚠️  Optional: Update OPENAI_API_KEY in .env file for fallback")
    else:
        print("✅ .env file already exists")

def main():
    print("🚀 AgentChain Startup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        print("❌ Python 3.9+ required")
        print(f"   Current version: {sys.version}")
        return
    
    print("✅ Python version OK")
    
    # Check Docker
    if not check_docker():
        print("❌ Docker not found or not running")
        print("   Please install Docker Desktop and start it")
        return
    
    print("✅ Docker found")
    
    # Create .env file
    create_env_file()
    
    print("\n📋 Next steps:")
    print("1. Ensure Ollama is running on http://localhost:11434")
    print("2. Optional: Update OPENAI_API_KEY in .env file for fallback")
    print("3. Run: docker-compose up -d")
    print("4. Run: python scripts/setup_pipeline.py")
    print("5. Run: uvicorn agentchain.api.main:app --reload")
    print("6. Run: python scripts/test_pipeline.py")
    
    print("\n🎉 Setup complete! Follow the steps above to start AgentChain.")

if __name__ == "__main__":
    main() 