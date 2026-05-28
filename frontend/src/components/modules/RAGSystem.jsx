import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search } from 'lucide-react';
import axios from 'axios';

const RAGSystem = () => {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query) return;
    setLoading(true);
    try {
      // Mock API call to RAG endpoint
      await axios.get('http://localhost:8000/health');
      setTimeout(() => {
        setResult(`Found 3 relevant threat intelligence reports in ChromaDB for: "${query}". \n\n1. Associated IP 192.168.1.45 has a history of SQL Injection attempts (Confidence: 94%).\n2. Similar attack vectors observed in recent CVE-2023-XXXX exploits.\n3. Recommended Mitigation: Block IP at the WAF level and rotate database credentials.`);
        setLoading(false);
      }, 1500);
    } catch (error) {
      setResult("Error querying RAG system.");
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="glass-panel p-8 mt-10"
    >
      <h2 className="text-glow-primary text-3xl mb-6" style={{ marginBottom: '24px' }}>RAG Threat Intelligence</h2>
      
      <form onSubmit={handleSearch} style={{ display: 'flex', gap: '16px', marginBottom: '32px' }}>
        <input 
          type="text" 
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search knowledge base for threat vectors (e.g., 'SQLi patterns')..." 
          style={{ flex: 1, padding: '16px 24px', borderRadius: '12px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', fontSize: '1rem', outline: 'none' }}
        />
        <button type="submit" className="btn-glow btn-primary" disabled={loading}>
          <Search size={20} />
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {result && (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-panel" 
          style={{ padding: '24px', borderLeft: '4px solid var(--primary)' }}
        >
          <h3 style={{ marginBottom: '16px', color: 'var(--primary)' }}>Retrieval Results</h3>
          <p style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }} className="text-muted">
            {result}
          </p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default RAGSystem;
