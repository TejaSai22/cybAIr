"""
Security Log Collector
Parses various security log formats and extracts threat indicators
"""
import re
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import threading

from agentchain.common.pipeline import AgentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLogCollector:
    """Security log parser and threat indicator extractor"""
    
    def __init__(self, log_paths: List[str] = None):
        self.log_paths = log_paths or []
        self.pipeline = AgentPipeline()
        self.is_monitoring = False
        self.parsed_events = []
        
        # Common threat patterns
        self.threat_patterns = {
            "brute_force": [
                r"Failed password for",
                r"authentication failure",
                r"invalid login attempt"
            ],
            "port_scan": [
                r"port scan detected",
                r"connection attempt to port",
                r"multiple connection attempts"
            ],
            "malware": [
                r"virus detected",
                r"malware found",
                r"suspicious file",
                r"trojan detected"
            ],
            "ddos": [
                r"flood attack",
                r"rate limit exceeded",
                r"too many requests"
            ]
        }
    
    def add_log_path(self, log_path: str):
        """Add a log file path to monitor"""
        if Path(log_path).exists():
            self.log_paths.append(log_path)
            logger.info(f"Added log path: {log_path}")
        else:
            logger.warning(f"Log path does not exist: {log_path}")
    
    def parse_log_line(self, line: str, log_type: str = "generic") -> Optional[Dict[str, Any]]:
        """Parse a single log line and extract threat indicators"""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "log_type": log_type,
                "raw_line": line.strip(),
                "threat_indicators": [],
                "severity": "low"
            }
            
            # Extract timestamp if present
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if timestamp_match:
                event["log_timestamp"] = timestamp_match.group(1)
            
            # Extract IP addresses
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ips = re.findall(ip_pattern, line)
            if ips:
                event["source_ips"] = list(set(ips))
            
            # Extract usernames
            user_pattern = r'user[:\s]+([^\s]+)'
            user_match = re.search(user_pattern, line, re.IGNORECASE)
            if user_match:
                event["username"] = user_match.group(1)
            
            # Check for threat patterns
            for threat_type, patterns in self.threat_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        event["threat_indicators"].append(threat_type)
                        event["severity"] = "high" if threat_type in ["malware", "ddos"] else "medium"
            
            # Additional parsing based on log type
            if log_type == "ssh":
                event.update(self._parse_ssh_log(line))
            elif log_type == "firewall":
                event.update(self._parse_firewall_log(line))
            elif log_type == "antivirus":
                event.update(self._parse_antivirus_log(line))
            
            return event if event["threat_indicators"] else None
            
        except Exception as e:
            logger.error(f"Error parsing log line: {e}")
            return None
    
    def _parse_ssh_log(self, line: str) -> Dict[str, Any]:
        """Parse SSH-specific log entries"""
        ssh_data = {}
        
        # SSH authentication failures
        if "Failed password" in line:
            ssh_data["event_type"] = "ssh_auth_failure"
            ssh_data["severity"] = "medium"
        
        # SSH brute force detection
        if line.count("Failed password") > 3:
            ssh_data["event_type"] = "ssh_brute_force"
            ssh_data["severity"] = "high"
        
        return ssh_data
    
    def _parse_firewall_log(self, line: str) -> Dict[str, Any]:
        """Parse firewall-specific log entries"""
        fw_data = {}
        
        # Blocked connections
        if "BLOCK" in line or "DENY" in line:
            fw_data["event_type"] = "firewall_block"
            fw_data["severity"] = "medium"
        
        # Port scanning
        if "port scan" in line.lower():
            fw_data["event_type"] = "port_scan"
            fw_data["severity"] = "high"
        
        return fw_data
    
    def _parse_antivirus_log(self, line: str) -> Dict[str, Any]:
        """Parse antivirus-specific log entries"""
        av_data = {}
        
        # Malware detection
        if any(keyword in line.lower() for keyword in ["virus", "malware", "trojan"]):
            av_data["event_type"] = "malware_detected"
            av_data["severity"] = "high"
        
        return av_data
    
    def monitor_log_file(self, log_path: str, log_type: str = "generic"):
        """Monitor a log file for new entries"""
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Seek to end of file
                f.seek(0, 2)
                
                while self.is_monitoring:
                    line = f.readline()
                    if line:
                        event = self.parse_log_line(line, log_type)
                        if event:
                            self.parsed_events.append(event)
                            self._send_event_to_pipeline(event)
                    else:
                        time.sleep(1)  # Wait for new content
                        
        except Exception as e:
            logger.error(f"Error monitoring log file {log_path}: {e}")
    
    def _send_event_to_pipeline(self, event: Dict[str, Any]):
        """Send parsed security event to the pipeline"""
        try:
            # Send to detection agent
            self.pipeline.send_detection_event({
                "source": "security_log",
                "event": event,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Sent security event to pipeline: {event.get('threat_indicators', [])}")
            
        except Exception as e:
            logger.error(f"Error sending event to pipeline: {e}")
    
    def start_monitoring(self):
        """Start monitoring all configured log files"""
        logger.info("Starting security log monitoring...")
        self.is_monitoring = True
        
        threads = []
        for log_path in self.log_paths:
            # Determine log type from path
            log_type = self._determine_log_type(log_path)
            
            thread = threading.Thread(
                target=self.monitor_log_file,
                args=(log_path, log_type),
                daemon=False  # Non-daemon to prevent crashes
            )
            thread.start()
            threads.append(thread)
        
        return threads
    
    def _determine_log_type(self, log_path: str) -> str:
        """Determine log type from file path"""
        path_lower = log_path.lower()
        
        if "ssh" in path_lower or "auth" in path_lower:
            return "ssh"
        elif "firewall" in path_lower or "iptables" in path_lower:
            return "firewall"
        elif "antivirus" in path_lower or "av" in path_lower:
            return "antivirus"
        else:
            return "generic"
    
    def stop_monitoring(self):
        """Stop monitoring log files"""
        self.is_monitoring = False
        logger.info("Security log monitoring stopped")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get statistics about monitored events"""
        if not self.parsed_events:
            return {"total_events": 0, "threat_types": {}}
        
        threat_counts = {}
        for event in self.parsed_events:
            for indicator in event.get("threat_indicators", []):
                threat_counts[indicator] = threat_counts.get(indicator, 0) + 1
        
        return {
            "total_events": len(self.parsed_events),
            "threat_types": threat_counts,
            "monitored_files": len(self.log_paths),
            "is_monitoring": self.is_monitoring
        } 