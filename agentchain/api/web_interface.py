"""
Web Interface for AgentChain Dashboard
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

router = APIRouter(tags=["Web Interface"])

# Create templates directory if it doesn't exist
templates_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "templates")
os.makedirs(templates_dir, exist_ok=True)

# Setup templates
templates = Jinja2Templates(directory=templates_dir)

@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_alt(request: Request):
    """Alternative dashboard route"""
    return templates.TemplateResponse("index.html", {"request": request}) 