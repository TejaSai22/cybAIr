from kafka import KafkaProducer, KafkaConsumer
from agentchain.common.config import settings
import json

def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def get_kafka_consumer(topic, group_id=None, auto_offset_reset='earliest'):
    return KafkaConsumer(
        topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=group_id,
        auto_offset_reset=auto_offset_reset,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    ) 