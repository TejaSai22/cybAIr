from agentchain.common.kafka_utils import get_kafka_producer, get_kafka_consumer
from agentchain.common.config import settings
import json
import threading
import time

class AgentPipeline:
    def __init__(self):
        self._producer = None
        
    @property
    def producer(self):
        """Lazy initialization of Kafka producer"""
        if self._producer is None:
            try:
                self._producer = get_kafka_producer()
            except Exception as e:
                print(f"⚠️  Kafka producer not available: {e}")
                return None
        return self._producer
        
    def send_detection_event(self, data):
        """Send detection result to classification agent"""
        if self.producer is None:
            print("⚠️  Kafka not available, skipping detection event")
            return
            
        message = {
            "timestamp": time.time(),
            "source": "detection",
            "data": data
        }
        self.producer.send(settings.kafka_classification_topic, message)
        
    def send_classification_event(self, data):
        """Send classification result to triage agent"""
        if self.producer is None:
            print("⚠️  Kafka not available, skipping classification event")
            return
            
        message = {
            "timestamp": time.time(),
            "source": "classification", 
            "data": data
        }
        self.producer.send(settings.kafka_triage_topic, message)
        
    def send_triage_event(self, data):
        """Send triage result to mitigation agent"""
        if self.producer is None:
            print("⚠️  Kafka not available, skipping triage event")
            return
            
        message = {
            "timestamp": time.time(),
            "source": "triage",
            "data": data
        }
        self.producer.send(settings.kafka_mitigation_topic, message)
        
    def send_mitigation_event(self, data):
        """Send mitigation result to graph agent"""
        if self.producer is None:
            print("⚠️  Kafka not available, skipping mitigation event")
            return
            
        message = {
            "timestamp": time.time(),
            "source": "mitigation",
            "data": data
        }
        # Could send to a graph topic or directly update Neo4j
        
    def start_pipeline_consumer(self, topic, handler_func):
        """Start a consumer for a specific topic"""
        try:
            consumer = get_kafka_consumer(topic, group_id=f"{topic}-group")
        except Exception as e:
            print(f"⚠️  Kafka consumer not available: {e}")
            return None
        
        def consume():
            for message in consumer:
                try:
                    data = message.value
                    handler_func(data)
                except Exception as e:
                    print(f"Error processing message: {e}")
                    
        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        return thread

# Global pipeline instance
pipeline = AgentPipeline() 