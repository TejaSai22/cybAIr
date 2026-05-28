import React from 'react';
import { motion } from 'framer-motion';

const NetworkMonitor = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="glass-panel p-8 mt-10"
    >
      <h2 className="text-glow-primary text-3xl mb-6" style={{ marginBottom: '24px' }}>Network Monitor</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginBottom: '16px', fontSize: '1.25rem' }}>Active Connections</h3>
          <ul className="text-muted" style={{ listStyle: 'none', padding: 0 }}>
            <li style={{ padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>192.168.1.45:443 - ESTABLISHED (TLSv1.3)</li>
            <li style={{ padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>10.0.0.12:80 - HTTP (Proxy)</li>
            <li className="text-warning" style={{ padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>203.0.113.4:22 - SSH (Multiple attempts)</li>
          </ul>
        </div>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ marginBottom: '16px', fontSize: '1.25rem' }}>Traffic Distribution</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div>
              <div className="flex-between" style={{ marginBottom: '4px' }}><span>HTTPS</span><span>65%</span></div>
              <div style={{ width: '100%', background: 'rgba(255,255,255,0.1)', borderRadius: '10px', height: '8px' }}>
                <div style={{ background: 'var(--primary)', height: '8px', width: '65%', borderRadius: '10px' }}></div>
              </div>
            </div>
            <div>
              <div className="flex-between" style={{ marginBottom: '4px' }}><span>DNS</span><span>15%</span></div>
              <div style={{ width: '100%', background: 'rgba(255,255,255,0.1)', borderRadius: '10px', height: '8px' }}>
                <div style={{ background: 'var(--success)', height: '8px', width: '15%', borderRadius: '10px' }}></div>
              </div>
            </div>
            <div>
              <div className="flex-between" style={{ marginBottom: '4px' }}><span>SSH</span><span>20%</span></div>
              <div style={{ width: '100%', background: 'rgba(255,255,255,0.1)', borderRadius: '10px', height: '8px' }}>
                <div style={{ background: 'var(--warning)', height: '8px', width: '20%', borderRadius: '10px' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default NetworkMonitor;
