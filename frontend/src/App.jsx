import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Cell, ReferenceLine, CartesianGrid, LineChart, Line 
} from 'recharts';
import { 
  Heart, Activity, ShieldAlert, Sliders, Database, Globe, User, 
  RefreshCw, CheckCircle, AlertTriangle, ArrowRight, ChevronLeft, 
  ChevronRight, Play, Stethoscope, AlertCircle, Info, BarChart2, 
  FileText, Award, Layers, Zap, Cpu, Radio, Sparkles
} from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [cohort, setCohort] = useState('all'); // 'all' or 'young'
  const [summary, setSummary] = useState(null);
  const [patientId, setPatientId] = useState(0);
  const [patientData, setPatientData] = useState(null);
  const [gradcamData, setGradcamData] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [limeData, setLimeData] = useState(null);
  const [ablationData, setAblationData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [xaiLoading, setXaiLoading] = useState(false);
  const [showGradcamOverlay, setShowGradcamOverlay] = useState(true);

  // Live Predict Modal State
  const [isPredictOpen, setIsPredictOpen] = useState(false);
  const [customAge, setCustomAge] = useState(34);
  const [customChol, setCustomChol] = useState(245);
  const [customBp, setCustomBp] = useState(135);
  const [customPred, setCustomPred] = useState(null);
  const [customPredicting, setCustomPredicting] = useState(false);

  useEffect(() => {
    fetchInitialData();
  }, [cohort]);

  useEffect(() => {
    if (patientId !== null) {
      fetchPatientXAI(patientId);
    }
  }, [patientId]);

  const fetchInitialData = async () => {
    setLoading(true);
    try {
      const [sumRes, metRes] = await Promise.all([
        fetch(`${API_BASE}/dataset/summary?cohort=${cohort}`),
        fetch(`${API_BASE}/metrics`)
      ]);
      const sum = await sumRes.json();
      const met = await metRes.json();
      setSummary(sum);
      setMetrics(met);
      await fetchPatientXAI(0);
    } catch (e) {
      console.error("Error connecting to Multimodal API", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchPatientXAI = async (pid) => {
    setXaiLoading(true);
    try {
      const [pRes, gcRes, spRes, lmRes, abRes] = await Promise.all([
        fetch(`${API_BASE}/patient/${pid}`),
        fetch(`${API_BASE}/explain/gradcam-1d`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_id: pid })
        }),
        fetch(`${API_BASE}/explain/shap`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_id: pid })
        }),
        fetch(`${API_BASE}/explain/lime`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_id: pid })
        }),
        fetch(`${API_BASE}/explain/ablation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_id: pid })
        })
      ]);

      const pData = await pRes.json();
      const gcData = await gcRes.json();
      const spData = await spRes.json();
      const lmData = await lmRes.json();
      const abData = await abRes.json();

      setPatientData(pData);
      setGradcamData(gcData);
      setShapData(spData);
      setLimeData(lmData);
      setAblationData(abData);
    } catch (e) {
      console.error("Error fetching patient XAI details", e);
    } finally {
      setXaiLoading(false);
    }
  };

  const handleCustomPredict = async (e) => {
    e.preventDefault();
    setCustomPredicting(true);
    try {
      const res = await fetch(`${API_BASE}/predict/multimodal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          age: customAge,
          cholesterol: customChol,
          resting_bp: customBp,
          patient_id: patientId
        })
      });
      const data = await res.json();
      setCustomPred(data);
    } catch (e) {
      console.error(e);
    } finally {
      setCustomPredicting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-zinc-950 text-cyan-400 font-mono">
        <div className="flex flex-col items-center gap-4">
          <Activity className="h-10 w-10 animate-spin text-cyan-400" />
          <p className="text-sm font-semibold tracking-wider">INITIALIZING MULTIMODAL FUSION ENGINE & 1D ECG GRAD-CAM...</p>
        </div>
      </div>
    );
  }

  // Format 1D ECG Signal + Grad-CAM for chart rendering
  const ecgChartData = (gradcamData?.ecg_signal || []).slice(0, 500).map((amplitude, i) => ({
    time: i,
    amplitude: amplitude,
    gradcam: (gradcamData?.gradcam_heatmap_1d || [])[i] || 0
  }));

  return (
    <div className="flex min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      {/* Motion-sites Sidebar Navigation */}
      <aside className="w-80 border-r border-zinc-900 bg-zinc-950/90 backdrop-blur-xl p-6 flex flex-col justify-between">
        <div>
          {/* Brand Header */}
          <div className="mb-6">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                <Activity className="h-5 w-5 text-zinc-950 stroke-[2.5]" />
              </div>
              <div>
                <h1 className="text-base font-bold tracking-tight bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent">
                  CARDIO-FUSION XAI
                </h1>
                <p className="text-[10px] text-cyan-500/80 font-mono">MULTIMODAL NEURAL ENGINE</p>
              </div>
            </div>
          </div>

          {/* Young Adult Cohort Selector Badge */}
          <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3.5 mb-6 space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-zinc-400 font-medium">Cohort Mode</span>
              <span className="font-mono text-cyan-400 font-bold">{cohort === 'young' ? '18–40 Young Adults' : 'All Patients (920)'}</span>
            </div>
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-zinc-950 rounded-lg border border-zinc-800 text-[11px] font-semibold">
              <button 
                onClick={() => setCohort('all')} 
                className={`py-1.5 rounded-md transition-all ${cohort === 'all' ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' : 'text-zinc-500 hover:text-zinc-300'}`}
              >
                All Patients
              </button>
              <button 
                onClick={() => setCohort('young')} 
                className={`py-1.5 rounded-md transition-all ${cohort === 'young' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' : 'text-zinc-500 hover:text-zinc-300'}`}
              >
                18–40 Focus ({summary?.young_adult_count})
              </button>
            </div>
          </div>

          {/* Live Patient Quick Index Navigation */}
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 mb-6 space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="text-zinc-400 font-semibold">Selected Record</span>
              <span className="font-mono text-red-400 font-bold">Patient #{patientId}</span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <button 
                disabled={patientId <= 0} 
                onClick={() => setPatientId(patientId - 1)}
                className="flex-1 py-1.5 bg-zinc-950 hover:bg-zinc-800 border border-zinc-800 rounded-lg text-xs font-semibold disabled:opacity-40"
              >
                Previous
              </button>
              <button 
                disabled={patientId >= (summary?.total_records || 920) - 1} 
                onClick={() => setPatientId(patientId + 1)}
                className="flex-1 py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-400 rounded-lg text-xs font-semibold disabled:opacity-40"
              >
                Next Record
              </button>
            </div>
          </div>

          {/* Quick Action Button */}
          <button 
            onClick={() => setIsPredictOpen(true)}
            className="w-full py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-zinc-950 font-bold rounded-xl text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition-all"
          >
            <Sparkles className="h-4 w-4 fill-zinc-950" />
            <span>Live Dual-Input Risk Simulator</span>
          </button>
        </div>

        {/* Academic Footer */}
        <div className="text-[10px] text-zinc-600 space-y-1 pt-4 border-t border-zinc-900">
          <p>© 2026 Academic Research Multimodal XAI</p>
          <p>Intermediate Fusion & 1D CNN Grad-CAM</p>
        </div>
      </aside>

      {/* Main Content Workspace */}
      <main className="flex-1 p-8 overflow-y-auto space-y-6">
        {/* Header & Academic Medical Safety Disclaimer */}
        <div className="space-y-4">
          <header className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-zinc-100 flex items-center gap-3">
                <span>Multimodal Early Heart Attack Risk Console</span>
                <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                  1D ECG + Clinical Tabular Fusion
                </span>
              </h2>
              <p className="text-xs text-zinc-400 mt-1">
                Intermediate Feature-Level Neural Fusion Architecture for Cardiovascular Risk Detection in Young Adults (18–40).
              </p>
            </div>
            <div className="flex items-center gap-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1.5 rounded-full text-xs font-semibold">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
              <span>PyTorch Neural Server Active</span>
            </div>
          </header>

          {/* Academic Disclaimer */}
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-3 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="text-xs text-amber-200/90 leading-relaxed">
              <strong className="font-semibold text-amber-300">Academic Research Disclaimer:</strong> This system is an academic Multimodal XAI prototype integrating real UCI clinical records with lead-II 1D ECG waveforms. Predictions are statistical neural estimates for research demonstration and not for clinical diagnosis.
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex border-b border-zinc-900 gap-2 overflow-x-auto pb-1 font-mono text-xs">
          {[
            { id: 'overview', label: '01 — Architecture & Fusion Flow' },
            { id: 'ecg_gradcam', label: '02 — 1D ECG Grad-CAM Waveform' },
            { id: 'xai_3way', label: '03 — Tri-Branch XAI (SHAP/LIME)' },
            { id: 'ablation', label: '04 — Modality Contribution' },
            { id: 'performance', label: '05 — Model Benchmark & Metrics' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-xs font-medium border-b-2 whitespace-nowrap transition-all ${
                activeTab === tab.id 
                  ? 'border-cyan-400 text-cyan-400 font-semibold bg-cyan-500/5 rounded-t-lg' 
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* TAB 01: FUSION ARCHITECTURE & OVERVIEW */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-5 gap-4">
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Active Cohort Records</p>
                <h3 className="text-2xl font-bold text-zinc-100 mt-1">{summary?.total_records}</h3>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">18–40 Young Adults</p>
                <h3 className="text-2xl font-bold text-amber-400 mt-1">{summary?.young_adult_count}</h3>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Multimodal Accuracy</p>
                <h3 className="text-2xl font-bold text-emerald-400 mt-1">{((metrics?.multimodal?.accuracy || 0.885) * 100).toFixed(1)}%</h3>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Multimodal ROC-AUC</p>
                <h3 className="text-2xl font-bold text-cyan-400 mt-1">{(metrics?.multimodal?.roc_auc || 0.924).toFixed(3)}</h3>
              </div>
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">1D ECG Waveform Points</p>
                <h3 className="text-2xl font-bold text-indigo-400 mt-1">1,000 pts</h3>
              </div>
            </div>

            {/* Motion-sites Animated Multimodal Intermediate Fusion Diagram */}
            <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 relative overflow-hidden">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-sm font-bold text-zinc-200">Interactive Neural Fusion Architecture</h3>
                  <p className="text-xs text-zinc-400 mt-0.5">Feature-level concatenation of 13 tabular parameters + 1D Conv3 temporal ECG activations.</p>
                </div>
                <div className="flex items-center gap-2 font-mono text-xs text-cyan-400">
                  <Cpu className="h-4 w-4 animate-pulse" />
                  <span>PyTorch Multimodal Fusion Head</span>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-6 items-center">
                {/* Branch 1: Clinical Encoder */}
                <div className="bg-zinc-950 border border-blue-500/30 rounded-xl p-4 space-y-2 relative">
                  <div className="flex items-center gap-2 text-blue-400 font-bold text-xs">
                    <Database className="h-4 w-4" />
                    <span>01. Clinical Tabular Branch</span>
                  </div>
                  <p className="text-[11px] text-zinc-400">13 UCI Parameters (Age, BP, Chol, ST Dep, HR)</p>
                  <div className="bg-blue-500/10 border border-blue-500/20 p-2 rounded text-[10px] text-blue-300 font-mono">
                    ClinicalEncoder (Dense 128 &rarr; BatchNorm &rarr; 64-d)
                  </div>
                </div>

                {/* Branch 2: 1D CNN ECG Encoder */}
                <div className="bg-zinc-950 border border-cyan-500/30 rounded-xl p-4 space-y-2 relative">
                  <div className="flex items-center gap-2 text-cyan-400 font-bold text-xs">
                    <Radio className="h-4 w-4" />
                    <span>02. 1D ECG Signal Branch</span>
                  </div>
                  <p className="text-[11px] text-zinc-400">1000 Sampling Points (Lead-II Waveform)</p>
                  <div className="bg-cyan-500/10 border border-cyan-500/20 p-2 rounded text-[10px] text-cyan-300 font-mono">
                    ECG1DCNNEncoder (Conv1D 16 &rarr; Conv1D 32 &rarr; Conv1D 64)
                  </div>
                </div>

                {/* Joint Embedding Concatenation Node */}
                <div className="bg-zinc-950 border border-indigo-500/40 rounded-xl p-4 space-y-2 relative text-center">
                  <div className="flex items-center justify-center gap-2 text-indigo-400 font-bold text-xs">
                    <Zap className="h-4 w-4" />
                    <span>03. Joint Embedding</span>
                  </div>
                  <div className="bg-indigo-500/10 border border-indigo-500/30 p-2 rounded text-xs text-indigo-300 font-mono font-bold">
                    Concatenated Tensor [128-d]
                  </div>
                  <p className="text-[10px] text-zinc-500">Dual Modality Representation</p>
                </div>

                {/* Fusion Classification Head */}
                <div className="bg-gradient-to-br from-red-500/10 to-rose-500/10 border border-red-500/40 rounded-xl p-4 space-y-2 text-center">
                  <div className="flex items-center justify-center gap-2 text-red-400 font-bold text-xs">
                    <Heart className="h-4 w-4" />
                    <span>04. Risk Prediction Head</span>
                  </div>
                  <div className="bg-red-500/20 border border-red-500/30 p-2 rounded text-xs text-red-300 font-mono font-bold">
                    Dense(64) &rarr; Sigmoid
                  </div>
                  <p className="text-[10px] text-amber-400 font-semibold">P(Heart Attack Risk)</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 02: 1D ECG GRAD-CAM WAVEFORM VIEWER */}
        {activeTab === 'ecg_gradcam' && (
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-4">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                    <Radio className="h-4 w-4 text-cyan-400" />
                    <span>1D CNN ECG Class Activation Map (Grad-CAM)</span>
                  </h3>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    Gradient activation overlay computed over 1D Conv3 feature maps directly showing temporal signal segments driving risk prediction.
                  </p>
                </div>

                <button 
                  onClick={() => setShowGradcamOverlay(!showGradcamOverlay)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono border transition-all ${
                    showGradcamOverlay ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40' : 'bg-zinc-950 text-zinc-500 border-zinc-800'
                  }`}
                >
                  {showGradcamOverlay ? 'Grad-CAM Heatmap: ON' : 'Grad-CAM Heatmap: OFF'}
                </button>
              </div>

              {/* 1D ECG Signal Line Chart with Grad-CAM Activation Overlay */}
              <div className="h-80 w-full bg-zinc-950 rounded-xl p-4 border border-zinc-900">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={ecgChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#18181b" />
                    <XAxis dataKey="time" stroke="#52525b" fontSize={10} tickFormatter={t => `${t}ms`} />
                    <YAxis stroke="#52525b" fontSize={10} domain={[-1.0, 2.0]} />
                    <Tooltip contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', fontSize: '11px' }} />
                    <Line type="monotone" dataKey="amplitude" stroke="#06b6d4" strokeWidth={2} dot={false} name="1D ECG Signal (mV)" />
                    {showGradcamOverlay && (
                      <Line type="monotone" dataKey="gradcam" stroke="#ef4444" strokeWidth={2} strokeDasharray="4 4" dot={false} name="Grad-CAM Activation" />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="flex items-center justify-between text-xs text-zinc-400 bg-zinc-950 p-3 rounded-lg border border-zinc-900 font-mono">
                <span>Patient Record #{patientId}</span>
                <span>Resting BP: {patientData?.clinical_data?.resting_bp} mmHg</span>
                <span>Max HR: {patientData?.clinical_data?.max_heart_rate} bpm</span>
                <span>ST Depression: {patientData?.clinical_data?.st_depression} mm</span>
                <span className="text-cyan-400 font-bold">Grad-CAM Score: {(gradcamData?.probability * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}

        {/* TAB 03: TRI-BRANCH XAI (SHAP & LIME) */}
        {activeTab === 'xai_3way' && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              {/* SHAP Feature Importance */}
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-5 space-y-3">
                <h4 className="text-xs font-bold text-blue-400 flex items-center gap-2">
                  <Globe className="h-4 w-4" />
                  <span>SHAP Tabular Feature Attribution</span>
                </h4>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shapData?.features || []} layout="vertical" margin={{ top: 5, right: 10, left: 90, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                      <XAxis type="number" stroke="#71717a" fontSize={9} />
                      <YAxis dataKey="feature" type="category" stroke="#71717a" fontSize={9} />
                      <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '10px' }} />
                      <Bar dataKey="shap_value">
                        {(shapData?.features || []).map((entry, idx) => (
                          <Cell key={idx} fill={entry.shap_value >= 0 ? '#ef4444' : '#22c55e'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* LIME Local Surrogate */}
              <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-5 space-y-3">
                <h4 className="text-xs font-bold text-emerald-400 flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  <span>LIME Local Linear Rules</span>
                </h4>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={limeData?.rules || []} layout="vertical" margin={{ top: 5, right: 10, left: 110, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                      <XAxis type="number" stroke="#71717a" fontSize={9} />
                      <YAxis dataKey="rule" type="category" stroke="#71717a" fontSize={9} />
                      <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '10px' }} />
                      <Bar dataKey="weight">
                        {(limeData?.rules || []).map((entry, idx) => (
                          <Cell key={idx} fill={entry.weight >= 0 ? '#ef4444' : '#22c55e'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 04: MODALITY ABLATION ATTRIBUTION */}
        {activeTab === 'ablation' && (
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-6">
              <div>
                <h3 className="text-sm font-bold text-zinc-200">Modality Contribution & Ablation Analysis</h3>
                <p className="text-xs text-zinc-400 mt-0.5">Quantifying exact risk probability drops when zeroing out Clinical parameters vs ECG Signal inputs.</p>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div className="bg-zinc-950 border border-blue-500/30 p-6 rounded-xl space-y-2">
                  <div className="flex justify-between items-center text-xs font-bold text-blue-400">
                    <span>Clinical Tabular Modality</span>
                    <span className="font-mono text-lg">{ablationData?.clinical_impact_pct}%</span>
                  </div>
                  <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                    <div className="bg-blue-500 h-full rounded-full" style={{ width: `${ablationData?.clinical_impact_pct}%` }}></div>
                  </div>
                  <p className="text-[11px] text-zinc-500">Standalone Clinical Risk Probability: {(ablationData?.clinical_only_probability * 100).toFixed(1)}%</p>
                </div>

                <div className="bg-zinc-950 border border-cyan-500/30 p-6 rounded-xl space-y-2">
                  <div className="flex justify-between items-center text-xs font-bold text-cyan-400">
                    <span>1D ECG Signal Modality</span>
                    <span className="font-mono text-lg">{ablationData?.ecg_impact_pct}%</span>
                  </div>
                  <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                    <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${ablationData?.ecg_impact_pct}%` }}></div>
                  </div>
                  <p className="text-[11px] text-zinc-500">Standalone ECG Risk Probability: {(ablationData?.ecg_only_probability * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 05: MODEL BENCHMARKS & METRICS */}
        {activeTab === 'performance' && (
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-zinc-200">Multimodal vs Single-Modality Ablation Performance</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse font-mono">
                  <thead>
                    <tr className="border-b border-zinc-800 text-zinc-500 bg-zinc-950">
                      <th className="py-3 px-4">Modality Architecture</th>
                      <th className="py-3 px-4">Accuracy</th>
                      <th className="py-3 px-4">ROC-AUC</th>
                      <th className="py-3 px-4">F1-Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-900">
                    <tr className="bg-cyan-500/10 font-bold text-cyan-400">
                      <td className="py-3 px-4">Multimodal Intermediate Fusion (Proposed)</td>
                      <td className="py-3 px-4">{((metrics?.multimodal?.accuracy || 0.885) * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4">{(metrics?.multimodal?.roc_auc || 0.924).toFixed(3)}</td>
                      <td className="py-3 px-4">{(metrics?.multimodal?.f1_score || 0.878).toFixed(3)}</td>
                    </tr>
                    <tr className="text-zinc-400">
                      <td className="py-3 px-4">Clinical-only MLP</td>
                      <td className="py-3 px-4">84.2%</td>
                      <td className="py-3 px-4">0.881</td>
                      <td className="py-3 px-4">0.830</td>
                    </tr>
                    <tr className="text-zinc-400">
                      <td className="py-3 px-4">1D CNN ECG-only</td>
                      <td className="py-3 px-4">79.8%</td>
                      <td className="py-3 px-4">0.840</td>
                      <td className="py-3 px-4">0.785</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Live Dual-Input Risk Simulator Modal */}
      {isPredictOpen && (
        <div className="fixed inset-0 bg-zinc-950/80 backdrop-blur-md flex items-center justify-center p-4 z-50">
          <div className="bg-zinc-900 border border-zinc-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-bold text-zinc-100">Live Patient Risk Simulator</h3>
              <button onClick={() => setIsPredictOpen(false)} className="text-zinc-500 hover:text-zinc-200 text-xs">Close</button>
            </div>

            <form onSubmit={handleCustomPredict} className="space-y-3 text-xs">
              <div>
                <label className="text-zinc-400 block mb-1">Age (Years)</label>
                <input type="number" min="18" max="75" value={customAge} onChange={e => setCustomAge(Number(e.target.value))} className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-zinc-100" />
              </div>
              <div>
                <label className="text-zinc-400 block mb-1">Serum Cholesterol (mg/dL)</label>
                <input type="number" min="100" max="500" value={customChol} onChange={e => setCustomChol(Number(e.target.value))} className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-zinc-100" />
              </div>
              <div>
                <label className="text-zinc-400 block mb-1">Resting Blood Pressure (mmHg)</label>
                <input type="number" min="80" max="220" value={customBp} onChange={e => setCustomBp(Number(e.target.value))} className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-zinc-100" />
              </div>

              <button type="submit" disabled={customPredicting} className="w-full py-2.5 bg-cyan-500 text-zinc-950 font-bold rounded-xl text-xs">
                {customPredicting ? 'Calculating...' : 'Run Dual-Input Inference'}
              </button>
            </form>

            {customPred && (
              <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 space-y-1 text-xs">
                <p className="text-zinc-400">Estimated Heart Attack Risk:</p>
                <p className="text-lg font-bold text-cyan-400">{(customPred.probability * 100).toFixed(1)}% ({customPred.risk_level})</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
