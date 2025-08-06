import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentchain.common.neo4j_utils import get_neo4j_session
from agentchain.common.kafka_utils import get_kafka_producer
from agentchain.common.config import settings
import time

def setup_neo4j():
    """Initialize Neo4j with sample data"""
    try:
        with get_neo4j_session() as session:
            # Create sample assets
            session.run("""
                CREATE (a1:Asset {id: 'asset-1', name: 'Web Server', ip: '192.168.1.100', type: 'server'})
                CREATE (a2:Asset {id: 'asset-2', name: 'Database Server', ip: '192.168.1.101', type: 'database'})
                CREATE (a3:Asset {id: 'asset-3', name: 'User Workstation', ip: '192.168.1.50', type: 'workstation'})
            """)
            
            # Create sample threats
            session.run("""
                CREATE (t1:Threat {id: 'threat-1', name: 'SQL Injection', type: 'malware', severity: 'high'})
                CREATE (t2:Threat {id: 'threat-2', name: 'DDoS Attack', type: 'network', severity: 'critical'})
                CREATE (t3:Threat {id: 'threat-3', name: 'Phishing Email', type: 'social', severity: 'medium'})
            """)
            
            print("✅ Neo4j initialized with sample data")
            
    except Exception as e:
        print(f"❌ Error setting up Neo4j: {e}")

def setup_kafka_topics():
    """Create Kafka topics if they don't exist"""
    try:
        producer = get_kafka_producer()
        
        # Send a test message to each topic to create them
        topics = [
            settings.kafka_detection_topic,
            settings.kafka_classification_topic,
            settings.kafka_triage_topic,
            settings.kafka_mitigation_topic
        ]
        
        for topic in topics:
            producer.send(topic, {"test": "setup"})
            print(f"✅ Created Kafka topic: {topic}")
            
        producer.flush()
        print("✅ Kafka topics initialized")
        
    except Exception as e:
        print(f"❌ Error setting up Kafka topics: {e}")

def main():
    print("🚀 Setting up AgentChain Pipeline...")
    
    # Wait for services to be ready
    print("⏳ Waiting for services to be ready...")
    time.sleep(10)
    
    setup_neo4j()
    setup_kafka_topics()
    
    print("🎉 Pipeline setup complete!")
    print("\n📋 Next steps:")
    print("1. Start your FastAPI app: uvicorn agentchain.api.main:app --reload")
    print("2. Run the test script: python scripts/test_pipeline.py")
    print("3. Visit Neo4j browser: http://localhost:7474")

if __name__ == "__main__":
    main() 