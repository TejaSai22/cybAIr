import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Network, Activity, ShieldAlert, Cpu, CheckCircle } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

import Sidebar from './components/Sidebar';
import MetricCard from './components/MetricCard';
import ControlPanel from './components/ControlPanel';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const App = () => {
  const [currentSection, setCurrentSection] = useState('dashboard');

  const pageVariants = {
    initial: { opacity: 0, x: 20 },
    in: { opacity: 1, x: 0 },
    out: { opacity: 0, x: -20 }
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { grid: { color: 'rgba(255,255,255,0.05)' } }
    },
    elements: {
      line: { tension: 0.4 }
    }
  };

  const chartData = {
    labels: ['10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30'],
    datasets: [
      {
        fill: true,
        label: 'Network Traffic',
        data: [150, 230, 180, 420, 190, 500, 310],
        borderColor: '#00f0ff',
        backgroundColor: 'rgba(0, 240, 255, 0.1)',
      }
    ]
  };

  return (
    <div className="app-container">
      <Sidebar currentSection={currentSection} setCurrentSection={setCurrentSection} />
      
      <main className="main-content">
        <header className="top-header glass-panel mb-6">
          <div className="flex-between">
            <div>
              <h1 className="text-2xl font-bold">Cyber Threat Response</h1>
              <p className="text-muted">Real-time Multi-Agent Intelligence</p>
            </div>
            <div className="status-badge">
              <span className="pulse-dot"></span>
              System Online
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {currentSection === 'dashboard' && (
            <motion.div
              key="dashboard"
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={{ duration: 0.4 }}
            >
              <div className="metrics-grid">
                <MetricCard title="Packets Captured" value="1.2M" color="var(--primary)" icon={Network} delay={0.1} />
                <MetricCard title="Threats Detected" value="432" color="var(--warning)" icon={Activity} delay={0.2} />
                <MetricCard title="ML Accuracy" value="98.5%" color="var(--secondary)" icon={Cpu} delay={0.3} />
                <MetricCard title="Active Threats" value="12" color="var(--danger)" icon={ShieldAlert} delay={0.4} />
                <MetricCard title="Mitigation Success" value="99.1%" color="var(--success)" icon={CheckCircle} delay={0.5} />
              </div>

              <ControlPanel />

              <div className="charts-container mt-6">
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                  className="chart-card glass-panel"
                >
                  <h3 className="mb-4">Live Network Traffic</h3>
                  <div className="chart-wrapper">
                    <Line options={chartOptions} data={chartData} />
                  </div>
                </motion.div>
                
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7 }}
                  className="chart-card glass-panel"
                >
                  <h3 className="mb-4">System Logs</h3>
                  <div className="logs-container">
                    <div className="log-item error">
                      <span className="time">10:30:14</span> [Detection] Anomalous packet detected (Score: -0.84)
                    </div>
                    <div className="log-item info">
                      <span className="time">10:30:15</span> [Classification] Threat classified as SQL Injection
                    </div>
                    <div className="log-item warning">
                      <span className="time">10:30:15</span> [Triage] Severity marked as HIGH
                    </div>
                    <div className="log-item success">
                      <span className="time">10:30:16</span> [Mitigation] IP 192.168.1.105 blocked successfully
                    </div>
                  </div>
                </motion.div>
              </div>
            </motion.div>
          )}

          {currentSection !== 'dashboard' && (
            <motion.div
              key="other"
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              className="glass-panel p-8 text-center mt-10"
            >
              <h2 className="text-glow-primary text-4xl mb-4">Module Initializing</h2>
              <p className="text-muted text-xl">The {currentSection} module is being connected to the Python backend...</p>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};

export default App;
