"""
Network Traffic Collector using Scapy
Captures real-time network packets for threat detection
"""
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
from scapy.layers.inet import Ether
import threading
from datetime import datetime
import logging

from agentchain.common.pipeline import AgentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkCollector:
    """Real-time network traffic collector and analyzer"""
    
    def __init__(self, interface: str = None, packet_count: int = 100):
        self.interface = interface
        self.packet_count = packet_count
        self.pipeline = AgentPipeline()
        self.is_capturing = False
        self.captured_packets = []
        
    def packet_callback(self, packet):
        """Process each captured packet"""
        try:
            packet_data = self._extract_packet_features(packet)
            if packet_data:
                self.captured_packets.append(packet_data)
                
                # Send to detection agent if we have enough data
                if len(self.captured_packets) >= 10:
                    self._send_batch_for_detection()
                    
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    def _extract_packet_features(self, packet) -> Optional[Dict[str, Any]]:
        """Extract relevant features from network packet"""
        try:
            # Basic packet info
            packet_info = {
                "timestamp": datetime.now().isoformat(),
                "length": len(packet),
                "protocol": "unknown"
            }
            
            # IP layer analysis
            if IP in packet:
                packet_info.update({
                    "src_ip": str(packet[IP].src),
                    "dst_ip": str(packet[IP].dst),
                    "protocol": int(packet[IP].proto),
                    "ttl": int(packet[IP].ttl),
                    "flags": int(packet[IP].flags)
                })
                
                # TCP analysis
                if TCP in packet:
                    packet_info.update({
                        "src_port": int(packet[TCP].sport),
                        "dst_port": int(packet[TCP].dport),
                        "flags": int(packet[TCP].flags),  # Convert FlagValue to int
                        "window": int(packet[TCP].window),
                        "seq": int(packet[TCP].seq),
                        "ack": int(packet[TCP].ack)
                    })
                    
                    # Detect suspicious patterns
                    packet_info["suspicious"] = self._detect_suspicious_tcp(packet_info)
                    
                # UDP analysis
                elif UDP in packet:
                    packet_info.update({
                        "src_port": int(packet[UDP].sport),
                        "dst_port": int(packet[UDP].dport),
                        "length": int(packet[UDP].len)
                    })
                    
                # ICMP analysis
                elif ICMP in packet:
                    packet_info.update({
                        "type": int(packet[ICMP].type),
                        "code": int(packet[ICMP].code)
                    })
                    
            # Payload analysis
            if Raw in packet:
                payload = packet[Raw].load
                packet_info["payload_size"] = len(payload)
                packet_info["payload_preview"] = payload[:50].hex()
                
            return packet_info
            
        except Exception as e:
            logger.error(f"Error extracting packet features: {e}")
            return None
    
    def _detect_suspicious_tcp(self, packet_info: Dict[str, Any]) -> bool:
        """Detect suspicious TCP patterns"""
        suspicious_indicators = [
            # Port scanning
            packet_info.get("dst_port", 0) in [22, 23, 80, 443, 3389, 8080],
            # SYN flood
            packet_info.get("flags", 0) == 2,  # SYN flag only
            # Unusual window size
            packet_info.get("window", 0) < 1000,
            # High TTL variation
            packet_info.get("ttl", 0) < 32
        ]
        
        return any(suspicious_indicators)
    
    def _send_batch_for_detection(self):
        """Send batch of packets to detection agent"""
        try:
            batch_data = {
                "packets": self.captured_packets[-10:],  # Last 10 packets
                "batch_id": f"batch_{int(time.time())}",
                "total_packets": len(self.captured_packets),
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to detection agent via pipeline
            self.pipeline.send_detection_event(batch_data)
            logger.info(f"Sent batch {batch_data['batch_id']} to detection agent")
            
        except Exception as e:
            logger.error(f"Error sending batch to detection: {e}")
    
    def start_capture(self, duration: int = 60):
        """Start packet capture for specified duration"""
        logger.info(f"Starting network capture on interface: {self.interface}")
        self.is_capturing = True
        
        try:
            # Start capture in a separate thread (non-daemon to prevent crashes)
            capture_thread = threading.Thread(
                target=self._capture_packets,
                args=(duration,),
                daemon=False
            )
            capture_thread.start()
            
            return capture_thread
            
        except Exception as e:
            logger.error(f"Error starting capture: {e}")
            return None
    
    def _capture_packets(self, duration: int):
        """Internal packet capture method"""
        try:
            sniff(
                iface=self.interface,
                prn=self.packet_callback,
                store=0,
                timeout=duration
            )
        except Exception as e:
            logger.error(f"Error during packet capture: {e}")
        finally:
            self.is_capturing = False
            logger.info("Network capture completed")
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get statistics about captured packets"""
        if not self.captured_packets:
            return {"total_packets": 0, "suspicious_count": 0}
        
        suspicious_count = sum(1 for p in self.captured_packets if p.get("suspicious", False))
        
        return {
            "total_packets": len(self.captured_packets),
            "suspicious_count": suspicious_count,
            "suspicious_percentage": (suspicious_count / len(self.captured_packets)) * 100,
            "protocols": self._get_protocol_distribution(),
            "top_ips": self._get_top_ips()
        }
    
    def _get_protocol_distribution(self) -> Dict[str, int]:
        """Get distribution of protocols"""
        protocols = {}
        for packet in self.captured_packets:
            protocol = packet.get("protocol", "unknown")
            protocols[protocol] = protocols.get(protocol, 0) + 1
        return protocols
    
    def _get_top_ips(self) -> List[Dict[str, Any]]:
        """Get top source and destination IPs"""
        ip_counts = {}
        for packet in self.captured_packets:
            for ip_field in ["src_ip", "dst_ip"]:
                ip = packet.get(ip_field)
                if ip:
                    ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        # Return top 5 IPs
        sorted_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"ip": ip, "count": count} for ip, count in sorted_ips[:5]] 