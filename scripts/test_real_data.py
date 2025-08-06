#!/usr/bin/env python3
"""
Test Real Data Integration
Tests network capture and security log monitoring with real data
"""
import sys
import os
import time
import json
import httpx
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_collectors():
    """Test the data collectors with real data"""
    print("🚀 Testing Real Data Integration")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        with httpx.Client(timeout=30.0) as client:
            
            # 1. Test collectors status
            print("\n1️⃣ Checking collectors status...")
            response = client.get(f"{base_url}/collectors/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✅ Collectors status: {json.dumps(status, indent=2)}")
            else:
                print(f"❌ Failed to get status: {response.status_code}")
                return
            
            # 2. Start demo collection
            print("\n2️⃣ Starting demo data collection...")
            response = client.post(f"{base_url}/collectors/demo/start")
            if response.status_code == 200:
                demo_result = response.json()
                print(f"✅ Demo started: {json.dumps(demo_result, indent=2)}")
            else:
                print(f"❌ Failed to start demo: {response.status_code}")
                return
            
            # 3. Wait for data collection
            print("\n3️⃣ Waiting for data collection (10 seconds)...")
            time.sleep(10)
            
            # 4. Check network stats
            print("\n4️⃣ Checking network capture stats...")
            response = client.get(f"{base_url}/collectors/network/stats")
            if response.status_code == 200:
                network_stats = response.json()
                print(f"✅ Network stats: {json.dumps(network_stats, indent=2)}")
            else:
                print(f"⚠️  Network stats not available: {response.status_code}")
            
            # 5. Check security stats
            print("\n5️⃣ Checking security monitoring stats...")
            response = client.get(f"{base_url}/collectors/security/stats")
            if response.status_code == 200:
                security_stats = response.json()
                print(f"✅ Security stats: {json.dumps(security_stats, indent=2)}")
            else:
                print(f"⚠️  Security stats not available: {response.status_code}")
            
            # 6. Test full pipeline with collected data
            print("\n6️⃣ Testing full pipeline with collected data...")
            test_pipeline_with_real_data(client, base_url)
            
            # 7. Stop collectors
            print("\n7️⃣ Stopping data collectors...")
            client.post(f"{base_url}/collectors/network/stop")
            client.post(f"{base_url}/collectors/security/stop")
            print("✅ Collectors stopped")
            
    except httpx.ConnectError:
        print("❌ Cannot connect to FastAPI server. Make sure it's running!")
        print("💡 Run: uvicorn agentchain.api.main:app --reload --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_pipeline_with_real_data(client, base_url):
    """Test the full pipeline with real collected data"""
    
    # Create sample threat data based on real patterns
    sample_threats = [
        {
            "source": "network_capture",
            "packets": [
                {
                    "src_ip": "192.168.1.100",
                    "dst_ip": "10.0.0.1",
                    "src_port": 12345,
                    "dst_port": 22,
                    "protocol": 6,  # TCP
                    "suspicious": True,
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "threat_indicators": ["port_scan", "brute_force"],
            "severity": "high"
        },
        {
            "source": "security_log",
            "event": {
                "log_type": "ssh",
                "source_ips": ["203.0.113.0"],
                "threat_indicators": ["brute_force"],
                "severity": "high",
                "timestamp": datetime.now().isoformat()
            }
        }
    ]
    
    for i, threat_data in enumerate(sample_threats, 1):
        print(f"\n   Testing threat {i}: {threat_data.get('threat_indicators', ['unknown'])}")
        
        # 1. Detection
        try:
            response = client.post(f"{base_url}/detection/detect", json=threat_data)
            if response.status_code == 200:
                detection_result = response.json()
                print(f"   ✅ Detection: {detection_result.get('anomaly_score', 'N/A')}")
            else:
                print(f"   ❌ Detection failed: {response.status_code}")
                continue
        except Exception as e:
            print(f"   ❌ Detection error: {e}")
            continue
        
        # 2. Classification
        try:
            response = client.post(f"{base_url}/classification/classify", json=threat_data)
            if response.status_code == 200:
                classification_result = response.json()
                print(f"   ✅ Classification: {classification_result.get('threat_type', 'N/A')}")
            else:
                print(f"   ❌ Classification failed: {response.status_code}")
                continue
        except Exception as e:
            print(f"   ❌ Classification error: {e}")
            continue
        
        # 3. Triage
        try:
            triage_data = {
                "threat_type": classification_result.get("threat_type", "unknown"),
                "anomaly_score": detection_result.get("anomaly_score", 0.5),
                "source_data": threat_data
            }
            response = client.post(f"{base_url}/triage/triage", json=triage_data)
            if response.status_code == 200:
                triage_result = response.json()
                print(f"   ✅ Triage: {triage_result.get('severity', 'N/A')} - {triage_result.get('action', 'N/A')}")
            else:
                print(f"   ❌ Triage failed: {response.status_code}")
                continue
        except Exception as e:
            print(f"   ❌ Triage error: {e}")
            continue
        
        # 4. Mitigation
        try:
            mitigation_data = {
                "threat_type": classification_result.get("threat_type", "unknown"),
                "severity": triage_result.get("severity", "medium"),
                "source_data": threat_data
            }
            response = client.post(f"{base_url}/mitigation/mitigate", json=mitigation_data)
            if response.status_code == 200:
                mitigation_result = response.json()
                print(f"   ✅ Mitigation: {mitigation_result.get('action', 'N/A')} (confidence: {mitigation_result.get('confidence', 'N/A')})")
            else:
                print(f"   ❌ Mitigation failed: {response.status_code}")
                continue
        except Exception as e:
            print(f"   ❌ Mitigation error: {e}")
            continue
        
        # 5. Graph update
        try:
            graph_data = {
                "asset_id": "web_server_01",
                "threat_data": {
                    "type": classification_result.get("threat_type", "unknown"),
                    "severity": triage_result.get("severity", "medium"),
                    "mitigation": mitigation_result.get("action", "none"),
                    "timestamp": datetime.now().isoformat()
                }
            }
            response = client.post(f"{base_url}/graph/update", json=graph_data)
            if response.status_code == 200:
                graph_result = response.json()
                print(f"   ✅ Graph updated: {graph_result.get('status', 'N/A')}")
            else:
                print(f"   ❌ Graph update failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Graph update error: {e}")

def test_individual_collectors():
    """Test individual collector features"""
    print("\n🔧 Testing Individual Collector Features")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        with httpx.Client(timeout=30.0) as client:
            
            # Test network collector
            print("\n📡 Testing Network Collector...")
            response = client.post(f"{base_url}/collectors/network/start", params={
                "duration": 15,
                "packet_count": 50
            })
            if response.status_code == 200:
                print(f"✅ Network capture started: {response.json()}")
                
                # Wait and check stats
                time.sleep(5)
                stats_response = client.get(f"{base_url}/collectors/network/stats")
                if stats_response.status_code == 200:
                    print(f"✅ Network stats: {stats_response.json()}")
                
                # Stop capture
                client.post(f"{base_url}/collectors/network/stop")
                print("✅ Network capture stopped")
            else:
                print(f"❌ Network capture failed: {response.status_code}")
            
            # Test security collector
            print("\n🔒 Testing Security Log Collector...")
            
            # Create a test log file
            test_log_content = [
                "2024-01-15 11:00:01 Failed password for user root from 192.168.1.50",
                "2024-01-15 11:00:02 Failed password for user root from 192.168.1.50",
                "2024-01-15 11:00:03 Failed password for user root from 192.168.1.50",
                "2024-01-15 11:01:15 port scan detected from 10.0.0.100",
                "2024-01-15 11:02:30 malware detected in file /var/tmp/suspicious.bin"
            ]
            
            with open("data/test_security.log", "w") as f:
                for line in test_log_content:
                    f.write(line + "\n")
            
            # Add log file
            response = client.post(f"{base_url}/collectors/security/add-log", params={
                "log_path": "data/test_security.log"
            })
            if response.status_code == 200:
                print(f"✅ Log file added: {response.json()}")
                
                # Start monitoring
                response = client.post(f"{base_url}/collectors/security/start", json=["data/test_security.log"])
                if response.status_code == 200:
                    print(f"✅ Security monitoring started: {response.json()}")
                    
                    # Wait and check stats
                    time.sleep(3)
                    stats_response = client.get(f"{base_url}/collectors/security/stats")
                    if stats_response.status_code == 200:
                        print(f"✅ Security stats: {stats_response.json()}")
                    
                    # Stop monitoring
                    client.post(f"{base_url}/collectors/security/stop")
                    print("✅ Security monitoring stopped")
                else:
                    print(f"❌ Security monitoring failed: {response.status_code}")
            else:
                print(f"❌ Log file addition failed: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Individual collector test failed: {e}")

if __name__ == "__main__":
    print("🚀 AgentChain Real Data Integration Test")
    print("=" * 60)
    
    # Test individual collectors first
    test_individual_collectors()
    
    # Test full integration
    test_data_collectors()
    
    print("\n🎉 Real Data Integration Test Complete!")
    print("=" * 60) 