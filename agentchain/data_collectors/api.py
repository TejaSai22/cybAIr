"""
Data Collectors API
FastAPI endpoints for network and security log collection
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any
import json
import time

from agentchain.data_collectors.network_collector import NetworkCollector
from agentchain.data_collectors.security_log_collector import SecurityLogCollector

router = APIRouter(tags=["Data Collectors"])

# Global collector instances
network_collector = None
security_collector = None

@router.post("/network/start")
async def start_network_capture(
    interface: str = None,
    duration: int = 60,
    packet_count: int = 100
):
    """Start network packet capture"""
    global network_collector
    
    try:
        network_collector = NetworkCollector(interface, packet_count)
        capture_thread = network_collector.start_capture(duration)
        
        return {
            "status": "success",
            "message": f"Network capture started on interface: {interface}",
            "duration": duration,
            "thread_id": capture_thread.ident if capture_thread else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start network capture: {str(e)}")

@router.post("/network/stop")
async def stop_network_capture():
    """Stop network packet capture"""
    global network_collector
    
    if network_collector:
        network_collector.is_capturing = False
        return {
            "status": "success",
            "message": "Network capture stopped"
        }
    else:
        raise HTTPException(status_code=404, detail="No active network capture")

@router.get("/network/stats")
async def get_network_stats():
    """Get network capture statistics"""
    global network_collector
    
    if not network_collector:
        raise HTTPException(status_code=404, detail="No network collector active")
    
    stats = network_collector.get_capture_stats()
    return {
        "collector_type": "network",
        "is_capturing": network_collector.is_capturing,
        "stats": stats
    }

@router.post("/security/start")
async def start_security_monitoring(log_paths: List[str]):
    """Start security log monitoring"""
    global security_collector
    
    try:
        security_collector = SecurityLogCollector(log_paths)
        monitoring_threads = security_collector.start_monitoring()
        
        return {
            "status": "success",
            "message": f"Security monitoring started for {len(log_paths)} log files",
            "monitored_files": log_paths,
            "thread_count": len(monitoring_threads)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start security monitoring: {str(e)}")

@router.post("/security/stop")
async def stop_security_monitoring():
    """Stop security log monitoring"""
    global security_collector
    
    if security_collector:
        security_collector.stop_monitoring()
        return {
            "status": "success",
            "message": "Security monitoring stopped"
        }
    else:
        raise HTTPException(status_code=404, detail="No active security monitoring")

@router.get("/security/stats")
async def get_security_stats():
    """Get security monitoring statistics"""
    global security_collector
    
    if not security_collector:
        raise HTTPException(status_code=404, detail="No security collector active")
    
    stats = security_collector.get_monitoring_stats()
    return {
        "collector_type": "security",
        "is_monitoring": security_collector.is_monitoring,
        "stats": stats
    }

@router.post("/security/add-log")
async def add_log_file(log_path: str):
    """Add a log file to security monitoring"""
    global security_collector
    
    if not security_collector:
        security_collector = SecurityLogCollector()
    
    security_collector.add_log_path(log_path)
    
    return {
        "status": "success",
        "message": f"Added log file: {log_path}",
        "total_monitored": len(security_collector.log_paths)
    }

@router.get("/status")
async def get_collectors_status():
    """Get status of all data collectors"""
    return {
        "network_collector": {
            "active": network_collector is not None,
            "is_capturing": network_collector.is_capturing if network_collector else False
        },
        "security_collector": {
            "active": security_collector is not None,
            "is_monitoring": security_collector.is_monitoring if security_collector else False
        }
    }

@router.post("/demo/start")
async def start_demo_collection():
    """Start demo data collection with sample data"""
    try:
        # Start network capture (will use default interface)
        global network_collector
        network_collector = NetworkCollector()
        network_thread = network_collector.start_capture(30)  # 30 seconds
        
        # Start security monitoring with sample log
        global security_collector
        security_collector = SecurityLogCollector()
        
        # Create a sample security log for demo
        sample_log_content = [
            "2024-01-15 10:30:15 Failed password for user admin from 192.168.1.100",
            "2024-01-15 10:30:16 Failed password for user admin from 192.168.1.100",
            "2024-01-15 10:30:17 Failed password for user admin from 192.168.1.100",
            "2024-01-15 10:31:20 port scan detected from 10.0.0.50",
            "2024-01-15 10:32:45 virus detected in file /tmp/suspicious.exe",
            "2024-01-15 10:33:10 firewall block connection from 203.0.113.0"
        ]
        
        # Write sample log
        with open("data/sample_security.log", "w") as f:
            for line in sample_log_content:
                f.write(line + "\n")
        
        security_collector.add_log_path("data/sample_security.log")
        security_threads = security_collector.start_monitoring()
        
        return {
            "status": "success",
            "message": "Demo collection started",
            "network_capture": "30 seconds",
            "security_log": "data/sample_security.log",
            "network_thread": network_thread.ident if network_thread else None,
            "security_threads": len(security_threads)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start demo: {str(e)}") 