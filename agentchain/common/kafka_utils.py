from kafka import KafkaProducer, KafkaConsumer
from agentchain.common.config import settings
import json
import logging

logger = logging.getLogger(__name__)

def get_kafka_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=3,
            acks='all'
        )
        logger.info("Kafka producer created successfully")
        return producer
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {e}")
        raise

def get_kafka_consumer(topic, group_id=None, auto_offset_reset='earliest'):
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True,
            auto_commit_interval_ms=1000
        )
        logger.info(f"Kafka consumer created successfully for topic: {topic}")
        return consumer
    except Exception as e:
        logger.error(f"Failed to create Kafka consumer: {e}")
        raise 