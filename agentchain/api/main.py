from fastapi import FastAPI
from agentchain.common.config import settings

# Import agent routers
from agentchain.detection.api import router as detection_router
from agentchain.classification.api import router as classification_router
from agentchain.triage.api import router as triage_router
from agentchain.mitigation.api import router as mitigation_router
from agentchain.graph.api import router as graph_router
from agentchain.data_collectors.api import router as collectors_router
from agentchain.ml_enhancements.api import router as ml_enhancements_router
from agentchain.api.web_interface import router as web_interface_router
from agentchain.rag.api import router as rag_router

app = FastAPI(
    title="AgentChain Cyber Threat Response API",
    description="Multi-agent cybersecurity system for threat detection, classification, and response",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AgentChain", "version": "1.0.0"}

# Register agent routers
app.include_router(detection_router, prefix="/detection", tags=["Detection"])
app.include_router(classification_router, prefix="/classification", tags=["Classification"])
app.include_router(triage_router, prefix="/triage", tags=["Triage"])
app.include_router(mitigation_router, prefix="/mitigation", tags=["Mitigation"])
app.include_router(graph_router, prefix="/graph", tags=["Graph"])
app.include_router(collectors_router, prefix="/collectors", tags=["Data Collectors"])
app.include_router(ml_enhancements_router, prefix="/ml", tags=["Enhanced ML Models"])
app.include_router(rag_router, prefix="/rag", tags=["RAG System"])
app.include_router(web_interface_router, tags=["Web Interface"])

@app.get("/")
def root():
    return {
        "message": "AgentChain Multi-Agent Cybersecurity System",
        "docs": "/docs",
        "health": "/health"
    } 