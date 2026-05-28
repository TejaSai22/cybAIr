import React from 'react';
import { motion } from 'framer-motion';
import { Shield, Activity, AlertTriangle, Brain, Search, FileText, BarChart2 } from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ currentSection, setCurrentSection }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <Activity size={20} /> },
    { id: 'network', label: 'Network Monitor', icon: <Shield size={20} /> },
    { id: 'threats', label: 'Threat Analysis', icon: <AlertTriangle size={20} /> },
    { id: 'ml', label: 'ML Models', icon: <Brain size={20} /> },
    { id: 'rag', label: 'RAG System', icon: <Search size={20} /> },
    { id: 'logs', label: 'Security Logs', icon: <FileText size={20} /> },
    { id: 'analytics', label: 'Analytics', icon: <BarChart2 size={20} /> },
  ];

  return (
    <motion.div 
      initial={{ x: -280 }}
      animate={{ x: 0 }}
      transition={{ type: 'spring', stiffness: 200, damping: 25 }}
      className="sidebar glass-panel"
    >
      <div className="sidebar-header">
        <Shield size={28} className="logo-icon text-primary" />
        <h2 className="text-glow-primary">AgentChain</h2>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <motion.button
            key={item.id}
            onClick={() => setCurrentSection(item.id)}
            className={`nav-item ${currentSection === item.id ? 'active' : ''}`}
            whileHover={{ x: 10, backgroundColor: 'rgba(0, 240, 255, 0.1)' }}
            whileTap={{ scale: 0.95 }}
          >
            <span className="icon">{item.icon}</span>
            <span className="label">{item.label}</span>
            
            {currentSection === item.id && (
              <motion.div 
                layoutId="active-indicator"
                className="active-indicator"
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
              />
            )}
          </motion.button>
        ))}
      </nav>
    </motion.div>
  );
};

export default Sidebar;
