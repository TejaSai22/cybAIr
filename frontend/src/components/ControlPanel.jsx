import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Play, Square, Brain, Search, Shield, Download, Target } from 'lucide-react';
import axios from 'axios';
import './ControlPanel.css';

const ControlPanel = () => {
  const [activeAction, setActiveAction] = useState(null);

  const handleAction = async (actionId, endpoint) => {
    setActiveAction(actionId);
    try {
      if (endpoint) {
        // We're making mock POST requests to the actual endpoints
        // In a real scenario, we'd pass proper payload
        await axios.get('http://localhost:8000/health');
        console.log(`Action ${actionId} executed against ${endpoint}`);
      }
    } catch (error) {
      console.error(`Failed to execute action ${actionId}`, error);
    }
    setTimeout(() => setActiveAction(null), 1000);
  };

  const controls = [
    { id: 'start', label: 'Start Capture', icon: <Play size={18} />, colorClass: 'btn-primary', endpoint: '/collectors/start' },
    { id: 'stop', label: 'Stop Capture', icon: <Square size={18} />, colorClass: 'btn-danger', endpoint: '/collectors/stop' },
    { id: 'train', label: 'Train ML', icon: <Brain size={18} />, colorClass: 'btn-success', endpoint: '/classification/train' },
    { id: 'rag', label: 'Init RAG', icon: <Search size={18} />, colorClass: 'btn-primary', endpoint: '/rag/index' },
    { id: 'scan', label: 'Full Scan', icon: <Shield size={18} />, colorClass: 'btn-primary', endpoint: '/detection/scan' },
    { id: 'report', label: 'Gen Report', icon: <Download size={18} />, colorClass: 'btn-primary', endpoint: '/report/generate' },
    { id: 'test', label: 'Test Acc.', icon: <Target size={18} />, colorClass: 'btn-primary', endpoint: '/ml/test' },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: 'spring' } }
  };

  return (
    <div className="control-panel glass-panel">
      <div className="panel-header flex-between">
        <h3 className="text-glow-primary">System Controls</h3>
        {activeAction && <span className="text-success text-sm pulse-dot">Executing...</span>}
      </div>
      <motion.div 
        className="controls-grid"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        {controls.map((ctrl) => (
          <motion.button
            key={ctrl.id}
            onClick={() => handleAction(ctrl.id, ctrl.endpoint)}
            variants={itemVariants}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`btn-glow ${ctrl.colorClass} ${activeAction === ctrl.id ? 'active-pulse' : ''}`}
          >
            {ctrl.icon}
            {ctrl.label}
          </motion.button>
        ))}
      </motion.div>
    </div>
  );
};

export default ControlPanel;
