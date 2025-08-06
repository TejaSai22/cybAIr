from neo4j import GraphDatabase
from agentchain.common.config import settings

_driver = None

def get_neo4j_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver

def get_neo4j_session():
    driver = get_neo4j_driver()
    return driver.session() 