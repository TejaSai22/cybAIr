import React from 'react';
import { motion } from 'framer-motion';
import './MetricCard.css';

const MetricCard = ({ title, value, color, icon: Icon, delay }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, type: 'spring' }}
      whileHover={{ y: -5, scale: 1.02 }}
      className="metric-card glass-panel"
      style={{ '--accent-color': color }}
    >
      <div className="metric-header">
        <span className="metric-title">{title}</span>
        <div className="metric-icon-wrapper" style={{ color: color }}>
          {Icon && <Icon size={24} />}
        </div>
      </div>
      <div className="metric-content">
        <motion.h3 
          className="metric-value"
          style={{ color: color }}
          initial={{ scale: 0.5 }}
          animate={{ scale: 1 }}
          transition={{ delay: delay + 0.2, type: 'spring' }}
        >
          {value}
        </motion.h3>
        <div className="metric-glow" style={{ background: color }}></div>
      </div>
    </motion.div>
  );
};

export default MetricCard;
