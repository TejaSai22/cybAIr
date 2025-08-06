#!/usr/bin/env python3
"""
Test Enhanced ML Models
Tests the advanced ML models for detection, classification, and RL
"""
import sys
import os
import time
import json
import random
import httpx
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_enhanced_ml_models():
    """Test the enhanced ML models"""
    print("🚀 Testing Enhanced ML Models")
    print("=" * 60)

    base_url = "http://127.0.0.1:8000"

    try:
        with httpx.Client(timeout=30.0) as client:

            # 1. Check ML models status
            print("\n1️⃣ Checking enhanced ML models status...")
            response = client.get(f"{base_url}/ml/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✅ ML models status: {json.dumps(status, indent=2)}")
            else:
                print(f"❌ Failed to get ML status: {response.status_code}")
                return

            # 2. Demo train all models
            print("\n2️⃣ Training all enhanced ML models with demo data...")
            response = client.post(f"{base_url}/ml/demo/train-all")
            if response.status_code == 200:
                training_result = response.json()
                print(f"✅ Training completed: {json.dumps(training_result, indent=2)}")
            else:
                print(f"❌ Training failed: {response.status_code}")
                return

            # 3. Test individual models
            print("\n3️⃣ Testing individual enhanced models...")
            test_individual_models(client, base_url)

            # 4. Test full enhanced pipeline
            print("\n4️⃣ Testing full enhanced ML pipeline...")
            test_enhanced_pipeline(client, base_url)

            # 5. Save models
            print("\n5️⃣ Saving enhanced models...")
            response = client.post(f"{base_url}/ml/models/save")
            if response.status_code == 200:
                save_result = response.json()
                print(f"✅ Models saved: {json.dumps(save_result, indent=2)}")
            else:
                print(f"❌ Model saving failed: {response.status_code}")

    except httpx.ConnectError:
        print("❌ Cannot connect to FastAPI server. Make sure it's running!")
        print("💡 Run: uvicorn agentchain.api.main:app --reload --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def test_individual_models(client, base_url):
    """Test individual enhanced ML models"""
    
    # Test advanced detection
    print("\n   📊 Testing Advanced Detection...")
    sample_packets = [
        {
            "length": 1500,
            "src_port": 12345,
            "dst_port": 22,
            "protocol": 6,  # TCP
            "ttl": 64,
            "window": 8192,
            "payload_size": 100,
            "suspicious": True,
            "timestamp": datetime.now().isoformat()
        },
        {
            "length": 64,
            "src_port": 54321,
            "dst_port": 80,
            "protocol": 6,  # TCP
            "ttl": 128,
            "window": 65535,
            "payload_size": 0,
            "suspicious": False,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    response = client.post(f"{base_url}/ml/detection/predict", json=sample_packets)
    if response.status_code == 200:
        detection_result = response.json()
        print(f"   ✅ Advanced detection: {detection_result['prediction']['anomaly_score']:.3f}")
    else:
        print(f"   ❌ Advanced detection failed: {response.status_code}")

    # Test advanced classification
    print("\n   🏷️  Testing Advanced Classification...")
    sample_threat = {
        "threat_type": "brute_force",
        "packets": sample_packets,
        "threat_indicators": ["brute_force", "port_scan"],
        "severity": "high",
        "anomaly_score": 0.85,
        "confidence": 0.92
    }
    
    response = client.post(f"{base_url}/ml/classification/predict", json=sample_threat)
    if response.status_code == 200:
        classification_result = response.json()
        print(f"   ✅ Advanced classification: {classification_result['prediction']['threat_type']}")
    else:
        print(f"   ❌ Advanced classification failed: {response.status_code}")

    # Test advanced RL
    print("\n   🤖 Testing Advanced RL...")
    sample_threat_data = {
        "severity": "high",
        "threat_indicators": ["brute_force", "port_scan"],
        "anomaly_score": 0.85,
        "confidence": 0.92
    }
    
    response = client.post(f"{base_url}/ml/rl/predict", json=sample_threat_data)
    if response.status_code == 200:
        rl_result = response.json()
        print(f"   ✅ Advanced RL action: {rl_result['prediction']['action']}")
    else:
        print(f"   ❌ Advanced RL failed: {response.status_code}")

def test_enhanced_pipeline(client, base_url):
    """Test the full enhanced ML pipeline"""
    
    # Create comprehensive threat data
    threat_data = {
        "source": "enhanced_test",
        "packets": [
            {
                "length": 1500,
                "src_port": 12345,
                "dst_port": 22,
                "protocol": 6,
                "ttl": 64,
                "window": 8192,
                "payload_size": 100,
                "suspicious": True,
                "timestamp": datetime.now().isoformat()
            }
        ],
        "threat_indicators": ["brute_force", "port_scan", "ddos"],
        "severity": "high",
        "anomaly_score": 0.9,
        "confidence": 0.95,
        "event": {
            "log_type": "ssh",
            "threat_indicators": ["brute_force"],
            "severity": "high",
            "source_ips": ["192.168.1.100"]
        }
    }
    
    response = client.post(f"{base_url}/ml/demo/test-pipeline", json=threat_data)
    if response.status_code == 200:
        pipeline_result = response.json()
        print(f"   ✅ Enhanced pipeline test completed")
        
        # Display results
        results = pipeline_result['results']
        if 'detection' in results and 'anomaly_score' in results['detection']:
            print(f"   📊 Detection Score: {results['detection']['anomaly_score']:.3f}")
        
        if 'classification' in results and 'threat_type' in results['classification']:
            print(f"   🏷️  Classification: {results['classification']['threat_type']}")
        
        if 'rl' in results and 'action' in results['rl']:
            print(f"   🤖 RL Action: {results['rl']['action']}")
        
    else:
        print(f"   ❌ Enhanced pipeline test failed: {response.status_code}")

def test_model_management():
    """Test model management features"""
    print("\n🔧 Testing Model Management Features")
    print("=" * 50)

    base_url = "http://127.0.0.1:8000"

    try:
        with httpx.Client(timeout=30.0) as client:

            # Test model initialization
            print("\n1️⃣ Testing model initialization...")
            
            # Initialize detection
            response = client.post(f"{base_url}/ml/detection/initialize", params={"model_type": "ensemble"})
            if response.status_code == 200:
                print("   ✅ Advanced detection initialized")
            else:
                print(f"   ❌ Detection initialization failed: {response.status_code}")

            # Initialize classification
            response = client.post(f"{base_url}/ml/classification/initialize", params={"model_type": "ensemble"})
            if response.status_code == 200:
                print("   ✅ Advanced classification initialized")
            else:
                print(f"   ❌ Classification initialization failed: {response.status_code}")

            # Initialize RL
            response = client.post(f"{base_url}/ml/rl/initialize", params={"agent_type": "q_learning"})
            if response.status_code == 200:
                print("   ✅ Advanced RL initialized")
            else:
                print(f"   ❌ RL initialization failed: {response.status_code}")

            # Test knowledge base building
            print("\n2️⃣ Testing knowledge base building...")
            threat_docs = [
                "Brute force attacks involve repeated login attempts with different credentials.",
                "Port scanning is a technique to identify open ports and services on a target system.",
                "DDoS attacks flood systems with traffic to make them unavailable.",
                "Malware is malicious software designed to harm systems or steal data."
            ]
            
            response = client.post(f"{base_url}/ml/classification/build-knowledge-base", json=threat_docs)
            if response.status_code == 200:
                print("   ✅ Knowledge base built successfully")
            else:
                print(f"   ❌ Knowledge base building failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Model management test failed: {e}")

def compare_basic_vs_enhanced():
    """Compare basic vs enhanced ML models"""
    print("\n⚖️  Comparing Basic vs Enhanced ML Models")
    print("=" * 50)

    base_url = "http://127.0.0.1:8000"

    try:
        with httpx.Client(timeout=30.0) as client:

            # Sample data for comparison
            sample_data = {
                "packets": [
                    {
                        "length": 1500,
                        "src_port": 12345,
                        "dst_port": 22,
                        "protocol": 6,
                        "suspicious": True
                    }
                ],
                "threat_indicators": ["brute_force"],
                "severity": "high"
            }

            print("\n1️⃣ Basic Detection vs Enhanced Detection...")
            
            # Basic detection
            response = client.post(f"{base_url}/detection/detect", json=sample_data)
            if response.status_code == 200:
                basic_result = response.json()
                print(f"   📊 Basic Detection Score: {basic_result.get('anomaly_score', 'N/A')}")
            else:
                print(f"   ❌ Basic detection failed: {response.status_code}")

            # Enhanced detection
            response = client.post(f"{base_url}/ml/detection/predict", json=sample_data["packets"])
            if response.status_code == 200:
                enhanced_result = response.json()
                print(f"   🚀 Enhanced Detection Score: {enhanced_result['prediction'].get('anomaly_score', 'N/A')}")
                print(f"   🎯 Models Used: {enhanced_result['prediction'].get('models_used', [])}")
            else:
                print(f"   ❌ Enhanced detection failed: {response.status_code}")

            print("\n2️⃣ Basic Classification vs Enhanced Classification...")
            
            # Basic classification
            response = client.post(f"{base_url}/classification/classify", json=sample_data)
            if response.status_code == 200:
                basic_result = response.json()
                print(f"   🏷️  Basic Classification: {basic_result.get('threat_type', 'N/A')}")
            else:
                print(f"   ❌ Basic classification failed: {response.status_code}")

            # Enhanced classification
            response = client.post(f"{base_url}/ml/classification/predict", json=sample_data)
            if response.status_code == 200:
                enhanced_result = response.json()
                print(f"   🚀 Enhanced Classification: {enhanced_result['prediction'].get('threat_type', 'N/A')}")
                print(f"   🎯 Confidence: {enhanced_result['prediction'].get('confidence', 'N/A')}")
            else:
                print(f"   ❌ Enhanced classification failed: {response.status_code}")

            print("\n3️⃣ Basic Mitigation vs Enhanced RL...")
            
            # Basic mitigation
            response = client.post(f"{base_url}/mitigation/mitigate", json=sample_data)
            if response.status_code == 200:
                basic_result = response.json()
                print(f"   🤖 Basic Action: {basic_result.get('action', 'N/A')}")
            else:
                print(f"   ❌ Basic mitigation failed: {response.status_code}")

            # Enhanced RL
            response = client.post(f"{base_url}/ml/rl/predict", json=sample_data)
            if response.status_code == 200:
                enhanced_result = response.json()
                print(f"   🚀 Enhanced RL Action: {enhanced_result['prediction'].get('action', 'N/A')}")
                print(f"   🎯 Confidence: {enhanced_result['prediction'].get('confidence', 'N/A')}")
            else:
                print(f"   ❌ Enhanced RL failed: {response.status_code}")

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")

if __name__ == "__main__":
    print("🚀 AgentChain Enhanced ML Models Test")
    print("=" * 60)

    # Test model management
    test_model_management()

    # Test enhanced ML models
    test_enhanced_ml_models()

    # Compare basic vs enhanced
    compare_basic_vs_enhanced()

    print("\n🎉 Enhanced ML Models Test Complete!")
    print("=" * 60)
    print("\n📋 Summary of Enhanced Features:")
    print("✅ Advanced Anomaly Detection (Ensemble: Isolation Forest, One-Class SVM, DBSCAN, MLP)")
    print("✅ Advanced Classification (TF-IDF + Numerical features + LLM enhancement)")
    print("✅ Advanced RL (Q-Learning + Policy Gradient for optimal actions)")
    print("✅ Model persistence and management")
    print("✅ Knowledge base building for RAG")
    print("✅ Comprehensive API endpoints")
    print("✅ Demo training and testing capabilities") 