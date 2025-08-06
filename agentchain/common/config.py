from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # General
    app_name: str = "AgentChain"
    environment: str = "development"
    debug: bool = True

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_detection_topic: str = "detection-events"
    kafka_classification_topic: str = "classification-events"
    kafka_triage_topic: str = "triage-events"
    kafka_mitigation_topic: str = "mitigation-events"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # LLM Configuration
    llm_provider: str = "ollama"  # "openai" or "ollama"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"  # or "mistral", "codellama", etc.

    # RAG/Embedding
    faiss_index_path: str = "./data/faiss.index"
    pinecone_api_key: Optional[str] = None
    pinecone_env: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings() 