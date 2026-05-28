# 🛡️ AgentChain: Multi-Agent Cyber Threat Response System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-Powered Cybersecurity Platform** - Real-time threat detection, classification, and automated response using multi-agent systems and local LLMs.

## 🎯 Overview

AgentChain is a **production-ready cybersecurity system** that combines multiple autonomous AI agents to detect, analyze, and respond to cyber threats in real-time. Built with cutting-edge technologies including local LLMs (Ollama), machine learning, reinforcement learning, and graph databases.

### 🌟 Key Features

- **🤖 Multi-Agent Architecture** - Detection, Classification, Triage, Mitigation, and Graph agents
- **🧠 Local AI Processing** - Ollama integration for privacy-focused threat analysis
- **📊 Real-time Monitoring** - Network packet analysis and security log processing
- **🔍 Advanced ML Models** - Ensemble anomaly detection and threat classification
- **🕸️ Graph Intelligence** - Neo4j for asset-threat relationship mapping
- **📈 Interactive Dashboard** - Modern web interface with real-time metrics
- **🚀 Production Ready** - Scalable, fault-tolerant architecture

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Detection     │    │ Classification  │    │     Triage      │
│     Agent       │───▶│     Agent       │───▶│     Agent       │
│  (ML Models)    │    │   (Ollama LLM)  │    │ (Severity)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mitigation    │    │      Graph      │    │      RAG        │
│     Agent       │    │     Agent       │    │   Knowledge     │
│   (RLlib)       │    │    (Neo4j)      │    │     Base        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Ollama** (with llama3 model)
- **Git**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TejaSai22/cybAIr.git
   cd cybAIr
   ```

2. **Create virtual environment**
   ```bash
   python -m venv agentchain_env
   source agentchain_env/bin/activate  # Linux/Mac
   # or
   agentchain_env\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start infrastructure services**
   ```bash
   docker-compose up -d
   ```

6. **Initialize the system**
   ```bash
   python scripts/setup_pipeline.py
   ```

7. **Start the application**
   ```bash
   uvicorn agentchain.api.main:app --reload --host 127.0.0.1 --port 8000
   ```

8. **Access the dashboard**
   ```
   http://127.0.0.1:8000
   ```

## 🛠️ Technology Stack

### **Backend & AI**
- **FastAPI** - High-performance web framework
- **Ollama** - Local LLM for threat classification
- **scikit-learn** - Machine learning models
- **Custom Q-Learning** - Reinforcement learning for mitigation
- **FAISS** - Vector similarity search
- **LangChain** - LLM orchestration

### **Data & Storage**
- **Neo4j** - Graph database for relationships
- **Kafka** - Real-time data streaming
- **Pinecone** - Cloud vector database (optional)

### **Network & Security**
- **Scapy** - Network packet manipulation
- **Wireshark** - Network protocol analyzer
- **Security Log Parsing** - Multi-format log analysis

### **Frontend**
- **HTML/CSS/JavaScript** - Modern web interface
- **Bootstrap** - Responsive design
- **Chart.js** - Data visualization
- **Font Awesome** - Icons

## 📊 System Components

### **🤖 AI Agents**

#### **Detection Agent**
- **Ensemble ML Models**: Isolation Forest, One-Class SVM, DBSCAN
- **Real-time Anomaly Detection**: Network traffic analysis
- **Feature Extraction**: Advanced packet analysis
- **Model Persistence**: Save/load trained models

#### **Classification Agent**
- **Ollama LLM**: Local threat classification
- **RAG Integration**: Context-aware analysis
- **Threat Categories**: DDoS, SQL Injection, Phishing, Malware, etc.
- **Detailed Analysis**: Comprehensive threat reports

#### **Triage Agent**
- **Severity Assessment**: Risk scoring and prioritization
- **Response Planning**: Automated action recommendations
- **Escalation Logic**: Human intervention triggers

#### **Mitigation Agent**
- **Reinforcement Learning**: Custom Tabular Q-Learning Implementation
- **Optimal Response Selection**: AI-driven mitigation strategies
- **Action Execution**: Automated threat containment

#### **Graph Agent**
- **Neo4j Integration**: Asset-threat relationship mapping
- **Real-time Updates**: Dynamic graph maintenance
- **Query Interface**: Advanced relationship queries

### **📈 Web Dashboard**

- **Real-time Metrics**: Live threat monitoring
- **Interactive Charts**: Data visualization
- **Control Panel**: System management
- **Alert Management**: Threat notifications
- **Log Viewer**: System activity monitoring
- **RAG Interface**: Knowledge base management

## 🔧 API Endpoints

### **Core Agents**
```
POST /detection/detect      - Detect anomalies
POST /detection/train       - Train detection models
POST /classification/classify - Classify threats
POST /triage/triage         - Assess severity
POST /mitigation/mitigate   - Execute responses
GET  /graph/assets          - List assets
POST /graph/update          - Update relationships
```

### **RAG System**
```
GET  /rag/status            - System status
POST /rag/add               - Add threat intelligence
GET  /rag/search            - Search knowledge base
GET  /rag/common-threats    - Common threats
GET  /rag/trends            - Threat trends
```

### **Data Collectors**
```
POST /collectors/network    - Network packet capture
POST /collectors/logs       - Security log parsing
GET  /collectors/status     - Collector status
```

### **Enhanced ML**
```
POST /ml/detection/predict  - Advanced anomaly detection
POST /ml/classification     - Enhanced classification
POST /ml/rl/train           - RL model training
GET  /ml/status             - ML system status
```

## 📋 Configuration

### **Environment Variables (.env)**
```env
# General
APP_NAME=AgentChain
ENVIRONMENT=development
DEBUG=true

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_DETECTION_TOPIC=detection-events
KAFKA_CLASSIFICATION_TOPIC=classification-events
KAFKA_TRIAGE_TOPIC=triage-events
KAFKA_MITIGATION_TOPIC=mitigation-events

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Ollama Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# RAG Configuration
FAISS_INDEX_PATH=./data/faiss.index

# Optional: Pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_env
```

## 🧪 Testing

### **Run Automated Tests**
```bash
# Test full pipeline
python scripts/test_pipeline.py

# Test with real data
python scripts/test_real_data.py

# Test enhanced ML models
python scripts/test_enhanced_ml.py
```

### **Manual Testing**
```bash
# Test classification
curl -X POST "http://127.0.0.1:8000/classification/classify" \
     -H "Content-Type: application/json" \
     -d '{"payload": {"source_ip": "192.168.1.100", "dest_ip": "10.0.0.1", "port": 80}}'

# Test RAG system
curl -X GET "http://127.0.0.1:8000/rag/status"
```

## 📚 Documentation

- **API Documentation**: http://127.0.0.1:8000/docs
- **System Architecture**: See architecture diagram above
- **Agent Details**: Check individual agent modules
- **Configuration Guide**: See .env.example

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for local LLM capabilities
- **FastAPI** for the excellent web framework
- **Neo4j** for graph database technology
- **Apache Kafka** for real-time streaming
- **scikit-learn** for machine learning tools

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/TejaSai22/cybAIr/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TejaSai22/cybAIr/discussions)
- **Documentation**: Check the `/docs` folder

---

**⭐ Star this repository if you find it useful!**

**🛡️ Built with ❤️ for cybersecurity professionals** 