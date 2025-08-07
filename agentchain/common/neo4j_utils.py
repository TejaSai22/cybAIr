from neo4j import GraphDatabase
from agentchain.common.config import settings
import logging

logger = logging.getLogger(__name__)
_driver = None

def get_neo4j_driver():
    global _driver
    if _driver is None:
        try:
            _driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            # Test the connection
            _driver.verify_connectivity()
            logger.info("Neo4j connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            _driver = None
    return _driver

def get_neo4j_session():
    driver = get_neo4j_driver()
    if driver is None:
        raise ConnectionError("Neo4j driver not available")
    return driver.session() 