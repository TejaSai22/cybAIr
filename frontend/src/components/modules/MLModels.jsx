import React from 'react';
import { motion } from 'framer-motion';

const MLModels = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="glass-panel p-8 mt-10"
    >
      <h2 className="text-glow-primary text-3xl mb-6" style={{ marginBottom: '24px' }}>Machine Learning Models</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginBottom: '16px', fontSize: '1.25rem', color: 'var(--success)' }}>Anomaly Detection (Isolation Forest)</h3>
          <p className="text-muted" style={{ marginBottom: '16px' }}>Identifies zero-day threats and abnormal network behaviors by isolating outliers in the feature space.</p>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span>Status</span> <span style={{ color: 'var(--success)' }}>Trained & Active</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span>Contamination Rate</span> <span>0.05</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span>Last Retrained</span> <span>10 mins ago</span>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginBottom: '16px', fontSize: '1.25rem', color: 'var(--primary)' }}>Threat Classification (LLM)</h3>
          <p className="text-muted" style={{ marginBottom: '16px' }}>Uses advanced LLMs (Ollama/Llama3) via LangChain to classify specific attack vectors (e.g., SQLi, XSS, DDoS).</p>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span>Status</span> <span style={{ color: 'var(--success)' }}>Online</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span>Accuracy Score</span> <span style={{ color: 'var(--primary)' }}>98.5%</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span>Latency</span> <span>450ms</span>
          </div>
        </div>

      </div>
    </motion.div>
  );
};

export default MLModels;
