import React from 'react';
import { motion } from 'framer-motion';
import { Play, Square, Brain, Search, Shield, Download, Target } from 'lucide-react';
import './ControlPanel.css';

const ControlPanel = () => {
  const controls = [
    { id: 'start', label: 'Start Capture', icon: <Play size={18} />, colorClass: 'btn-primary' },
    { id: 'stop', label: 'Stop Capture', icon: <Square size={18} />, colorClass: 'btn-danger' },
    { id: 'train', label: 'Train ML', icon: <Brain size={18} />, colorClass: 'btn-success' },
    { id: 'rag', label: 'Init RAG', icon: <Search size={18} />, colorClass: 'btn-primary' },
    { id: 'scan', label: 'Full Scan', icon: <Shield size={18} />, colorClass: 'btn-primary' },
    { id: 'report', label: 'Gen Report', icon: <Download size={18} />, colorClass: 'btn-primary' },
    { id: 'test', label: 'Test Acc.', icon: <Target size={18} />, colorClass: 'btn-primary' },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: 'spring' } }
  };

  return (
    <div className="control-panel glass-panel">
      <div className="panel-header">
        <h3 className="text-glow-primary">System Controls</h3>
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
            variants={itemVariants}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`btn-glow ${ctrl.colorClass}`}
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
