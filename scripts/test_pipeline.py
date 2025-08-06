import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import time
import json

BASE_URL = "http://localhost:8000"

def test_full_pipeline():
    """Test the complete automated pipeline"""
    print("🧪 Testing AgentChain Full Pipeline")
    print("=" * 50)
    
    with httpx.Client() as client:
        # 1. Train Detection Agent
        print("1️⃣ Training Detection Agent...")
        train_data = {
            "samples": [
                {"src_port": 443, "dst_port": 8080, "packet_size": 512, "tcp_flags": 18, "duration": 0.5},
                {"src_port": 80, "dst_port": 443, "packet_size": 256, "tcp_flags": 16, "duration": 0.2},
                {"src_port": 22, "dst_port": 2222, "packet_size": 128, "tcp_flags": 2, "duration": 0.1}
            ]
        }
        r = client.post(f"{BASE_URL}/detection/train", json=train_data)
        print(f"   ✅ Training result: {r.json()}")
        
        # 2. Test Normal Traffic (should not trigger pipeline)
        print("\n2️⃣ Testing Normal Traffic...")
        normal_data = {"payload": {"src_port": 443, "dst_port": 8080, "packet_size": 512, "tcp_flags": 18, "duration": 0.5}}
        r = client.post(f"{BASE_URL}/detection/detect", json=normal_data)
        result = r.json()
        print(f"   📊 Detection result: {result}")
        
        if result.get("anomaly"):
            print("   ⚠️  Normal traffic flagged as anomaly - pipeline will trigger!")
        else:
            print("   ✅ Normal traffic correctly classified")
        
        # 3. Test Anomalous Traffic (should trigger full pipeline)
        print("\n3️⃣ Testing Anomalous Traffic (Pipeline Trigger)...")
        anomalous_data = {"payload": {"src_port": 9999, "dst_port": 9999, "packet_size": 9999, "tcp_flags": 999, "duration": 999}}
        r = client.post(f"{BASE_URL}/detection/detect", json=anomalous_data)
        result = r.json()
        print(f"   📊 Detection result: {result}")
        
        if result.get("anomaly"):
            print("   🚨 Anomaly detected! Pipeline should be running...")
            print("   ⏳ Waiting for pipeline to complete...")
            time.sleep(3)  # Give time for async processing
        else:
            print("   ⚠️  Anomalous traffic not detected")
        
        # 4. Check Graph Database
        print("\n4️⃣ Checking Graph Database...")
        r = client.get(f"{BASE_URL}/graph/assets")
        graph_data = r.json()
        print(f"   📊 Graph data: {json.dumps(graph_data, indent=2)}")
        
        # 5. Health Check
        print("\n5️⃣ Health Check...")
        r = client.get(f"{BASE_URL}/health")
        print(f"   ✅ Health status: {r.json()}")

def test_individual_agents():
    """Test individual agents manually"""
    print("\n🔧 Testing Individual Agents")
    print("=" * 50)
    
    with httpx.Client() as client:
        # Test Classification
        print("🔍 Testing Classification Agent...")
        classify_data = {"payload": {"anomaly": True, "score": -0.8, "features": {"src_port": 9999}}}
        r = client.post(f"{BASE_URL}/classification/classify", json=classify_data)
        print(f"   📊 Classification: {r.json()}")
        
        # Test Triage
        print("⚖️ Testing Triage Agent...")
        triage_data = {"payload": {"threat_type": "malware", "llm_result": "Detected malware"}}
        r = client.post(f"{BASE_URL}/triage/triage", json=triage_data)
        print(f"   📊 Triage: {r.json()}")
        
        # Test Mitigation
        print("🛡️ Testing Mitigation Agent...")
        mitigate_data = {"payload": {"severity": "high", "action": "block"}}
        r = client.post(f"{BASE_URL}/mitigation/mitigate", json=mitigate_data)
        print(f"   📊 Mitigation: {r.json()}")

if __name__ == "__main__":
    print("🚀 AgentChain Pipeline Test Suite")
    print("=" * 60)
    
    try:
        test_full_pipeline()
        test_individual_agents()
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("💡 Make sure:")
        print("   - FastAPI app is running (uvicorn agentchain.api.main:app --reload)")
        print("   - Kafka and Neo4j are running (docker-compose up)")
        print("   - Setup script has been run (python scripts/setup_pipeline.py)") 