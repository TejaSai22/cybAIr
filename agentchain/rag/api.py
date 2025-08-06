"""
RAG API - Threat Intelligence Management and Retrieval
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from agentchain.rag.knowledge_base import knowledge_base

logger = logging.getLogger(__name__)
router = APIRouter(tags=["RAG System"])

class ThreatIntelligence(BaseModel):
    threat_type: str
    description: str
    severity: str = "medium"
    indicators: List[str] = []
    attack_patterns: List[str] = []
    mitigation: List[str] = []
    source: str = "manual"
    cve_info: Optional[Dict[str, Any]] = None

class SearchQuery(BaseModel):
    query: str
    k: int = 5

class CVEInfo(BaseModel):
    id: str
    description: str
    severity: str
    cvss_score: Optional[float] = None
    affected_products: List[str] = []

@router.get("/status")
async def get_rag_status():
    """Get RAG system status and statistics"""
    try:
        stats = knowledge_base.get_threat_statistics()
        return {
            "status": "operational",
            "vector_store": stats.get("vector_store", "unknown"),
            "total_documents": stats.get("total_documents", 0),
            "embedding_model": stats.get("embedding_model", "OpenAI")
        }
    except Exception as e:
        logger.error(f"Failed to get RAG status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-threat")
async def add_threat_intelligence(threat: ThreatIntelligence):
    """Add threat intelligence to the knowledge base"""
    try:
        threat_data = threat.dict()
        success = knowledge_base.add_threat_intelligence(threat_data)
        
        if success:
            return {"status": "success", "message": f"Added threat: {threat.threat_type}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add threat intelligence")
    except Exception as e:
        logger.error(f"Failed to add threat intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_threats(query: SearchQuery):
    """Search for relevant threat intelligence"""
    try:
        results = knowledge_base.search_threats(query.query, query.k)
        return {
            "status": "success",
            "query": query.query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to search threats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/common-threats")
async def get_common_threats():
    """Get most common threat types"""
    try:
        threats = knowledge_base.get_common_threats()
        return {
            "status": "success",
            "threats": threats
        }
    except Exception as e:
        logger.error(f"Failed to get common threats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/threat-trends")
async def get_threat_trends():
    """Get threat trends over time"""
    try:
        trends = knowledge_base.get_threat_trends()
        return {
            "status": "success",
            "trends": trends
        }
    except Exception as e:
        logger.error(f"Failed to get threat trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-cve")
async def add_cve_info(cve: CVEInfo):
    """Add CVE information to the knowledge base"""
    try:
        cve_data = cve.dict()
        knowledge_base.add_cve_data(cve_data)
        return {"status": "success", "message": f"Added CVE: {cve.id}"}
    except Exception as e:
        logger.error(f"Failed to add CVE: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/demo/initialize")
async def initialize_demo_data():
    """Initialize demo threat intelligence data"""
    try:
        demo_threats = [
            {
                "threat_type": "DDoS Attack",
                "description": "Distributed Denial of Service attack targeting web servers",
                "severity": "high",
                "indicators": ["High traffic volume", "Multiple source IPs", "Service unavailability"],
                "attack_patterns": ["T1498", "T1499"],
                "mitigation": ["Rate limiting", "DDoS protection", "Traffic filtering"],
                "source": "demo"
            },
            {
                "threat_type": "SQL Injection",
                "description": "Malicious SQL code injection in web applications",
                "severity": "high",
                "indicators": ["Database errors", "Unauthorized access", "Data exfiltration"],
                "attack_patterns": ["T1190", "T1505"],
                "mitigation": ["Input validation", "Prepared statements", "WAF"],
                "source": "demo"
            },
            {
                "threat_type": "Phishing",
                "description": "Social engineering attack via deceptive emails",
                "severity": "medium",
                "indicators": ["Suspicious emails", "Fake domains", "Urgent requests"],
                "attack_patterns": ["T1566", "T1598"],
                "mitigation": ["Email filtering", "User training", "MFA"],
                "source": "demo"
            }
        ]
        
        for threat in demo_threats:
            knowledge_base.add_threat_intelligence(threat)
        
        return {
            "status": "success",
            "message": f"Initialized {len(demo_threats)} demo threats",
            "threats_added": [t["threat_type"] for t in demo_threats]
        }
    except Exception as e:
        logger.error(f"Failed to initialize demo data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 