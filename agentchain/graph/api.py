from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List
from agentchain.graph.model import list_assets, update_asset_threat

router = APIRouter()

class AssetThreatUpdate(BaseModel):
    asset_id: str
    threat_id: str
    relationship: str

@router.get("/assets")
def list_assets_endpoint():
    return list_assets()

@router.post("/update")
def update_asset_threat_endpoint(data: AssetThreatUpdate):
    return update_asset_threat(data.asset_id, data.threat_id, data.relationship) 