from agentchain.common.neo4j_utils import get_neo4j_session

def list_assets():
    try:
        with get_neo4j_session() as session:
            result = session.run("MATCH (a:Asset) RETURN a")
            assets = [record["a"]._properties for record in result]
            result = session.run("MATCH (t:Threat) RETURN t")
            threats = [record["t"]._properties for record in result]
        return {"assets": assets, "threats": threats}
    except Exception as e:
        return {"assets": [], "threats": [], "error": str(e)}

def update_asset_threat(asset_id: str, threat_id: str, relationship: str):
    try:
        with get_neo4j_session() as session:
            # Validate relationship type to prevent injection
            valid_relationships = ["AFFECTS", "TARGETS", "EXPLOITS", "COMPROMISES"]
            if relationship.upper() not in valid_relationships:
                return {"status": "error", "details": "Invalid relationship type"}
            
            cypher = (
                "MATCH (a:Asset {id: $asset_id}), (t:Threat {id: $threat_id}) "
                "MERGE (a)-[r:" + relationship.upper() + "]->(t) RETURN r"
            )
            session.run(cypher, asset_id=asset_id, threat_id=threat_id)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "details": str(e)} 