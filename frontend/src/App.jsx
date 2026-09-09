import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Cell, ReferenceLine, CartesianGrid, PieChart, Pie 
} from 'recharts';
import { 
  Heart, Activity, ShieldAlert, Sliders, Database, Globe, User, 
  RefreshCw, CheckCircle, AlertTriangle, ArrowRight, ChevronLeft, 
  ChevronRight, Play, Stethoscope, AlertCircle, Info, BarChart2, 
  FileText, Award, Layers, Zap
} from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [testSamples, setTestSamples] = useState([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [localExplanation, setLocalExplanation] = useState(null);
  const [globalShap, setGlobalShap] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [localLoading, setLocalLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [predicting, setPredicting] = useState(false);

  // Form states for retraining
  const [nEstimators, setNEstimators] = useState(100);
  const [maxDepth, setMaxDepth] = useState('None');

  // Form state for live patient risk prediction
  const [predInput, setPredInput] = useState({
    age: 34,
    sex: 1,
    chest_pain_type: 1,
    resting_bp: 135,
    cholesterol: 245,
    fasting_bs: 0,
    resting_ecg: 0,
    max_heart_rate: 168,
    exercise_angina: 0,
    st_depression: 0.8,
    st_slope: 1,
    num_major_vessels: 0,
    thalassemia: 3
  });
  const [predResult, setPredResult] = useState(null);

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    setLoading(true);
    try {
      const [infoRes, dbRes, globalRes] = await Promise.all([
        fetch(`${API_BASE}/model-info`),
        fetch(`${API_BASE}/dataset`),
        fetch(`${API_BASE}/explain/shap-global`)
      ]);
      const info = await infoRes.json();
      const db = await dbRes.json();
      const glob = await globalRes.json();

      setModelInfo(info);
      setTestSamples(db);
      setGlobalShap(glob);
      setNEstimators(info.n_estimators);
      setMaxDepth(info.max_depth);

      await fetchLocalExplanation(0);
    } catch (e) {
      console.error("Error fetching initial data", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchLocalExplanation = async (idx) => {
    setLocalLoading(true);
    try {
      const res = await fetch(`${API_BASE}/explain/local/${idx}`);
      const data = await res.json();
      setLocalExplanation(data);
      setSelectedIdx(idx);
    } catch (e) {
      console.error("Error fetching local explanation", e);
    } finally {
      setLocalLoading(false);
    }
  };

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      const res = await fetch(`${API_BASE}/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_estimators: nEstimators,
          max_depth: maxDepth === 'None' ? null : maxDepth
        })
      });
      const info = await res.json();
      setModelInfo(info);

      const [dbRes, globalRes] = await Promise.all([
        fetch(`${API_BASE}/dataset`),
        fetch(`${API_BASE}/explain/shap-global`)
      ]);
      const db = await dbRes.json();
      const glob = await globalRes.json();

      setTestSamples(db);
      setGlobalShap(glob);
      await fetchLocalExplanation(selectedIdx);
    } catch (e) {
      console.error("Error retraining model", e);
    } finally {
      setRetraining(false);
    }
  };

  const handleCustomPredict = async (e) => {
    e.preventDefault();
    setPredicting(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(predInput)
      });
      const data = await res.json();
      setPredResult(data);
    } catch (e) {
      console.error("Error running risk prediction", e);
    } finally {
      setPredicting(false);
    }
  };

  const loadBorderlineCase = async () => {
    setLocalLoading(true);
    try {
      const res = await fetch(`${API_BASE}/explain/uncertain`);
      const data = await res.json();
      setLocalExplanation(data);
      setSelectedIdx(data.index);
      setActiveTab('local');
    } catch (e) {
      console.error(e);
    } finally {
      setLocalLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-zinc-950 text-zinc-400">
        <div className="flex flex-col items-center gap-4">
          <Heart className="h-10 w-10 animate-pulse text-red-500" />
          <p className="text-sm font-medium">Initializing Cardiovascular XAI Pipeline & UCI Dataset...</p>
        </div>
      </div>
    );
  }

  const localShapData = localExplanation?.shap?.features.map(f => ({
    name: f.feature,
    val: f.value,
    shap: f.shap_value,
    display: `${f.feature} (${f.value.toFixed(1)})`
  })) || [];

  const localLimeData = localExplanation?.lime.map(l => ({
    rule: l.feature_rule,
    weight: l.weight
  })) || [];

  const localGradcamData = localExplanation?.gradcam.map(g => ({
    feature: g.feature,
    importance: g.importance,
    display: `${g.feature} (${g.value.toFixed(1)})`
  })) || [];

  return (
    <div className="flex min-h-screen bg-zinc-950 text-zinc-100">
      {/* Sidebar Navigation & Controls */}
      <aside className="w-80 border-r border-zinc-900 bg-zinc-950 p-6 flex flex-col justify-between">
        <div>
          {/* Brand Header */}
          <div className="mb-6">
            <div className="flex items-center gap-2">
              <Heart className="h-6 w-6 text-red-500 fill-red-500/20" />
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-red-500 via-rose-400 to-amber-400 bg-clip-text text-transparent">
                CardioXAI Console
              </h1>
            </div>
            <p className="text-[11px] text-zinc-500 mt-1">Early Heart Attack Risk Analysis (18–40 Focus)</p>
          </div>

          {/* 18-40 Young Adult Badge */}
          <div className="bg-gradient-to-r from-amber-500/10 to-rose-500/10 border border-amber-500/30 rounded-xl p-3 mb-6">
            <div className="flex items-center gap-2 text-amber-400 font-bold text-xs">
              <Award className="h-4 w-4" />
              <span>18–40 YOUNG ADULT MODE</span>
            </div>
            <p className="text-[10px] text-zinc-400 mt-1">
              Active Focus: {modelInfo?.young_adult_total} Real Young Adult Patient Records (Age 28–40) evaluated.
            </p>
          </div>

          {/* Model Hyperparameter Panel */}
          <div className="glass-panel p-4 space-y-4 mb-6">
            <div className="flex items-center gap-2 text-zinc-300 font-semibold text-xs">
              <Sliders className="h-3.5 w-3.5 text-blue-500" />
              <span>Random Forest Classifier Config</span>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-[11px]">
                <span className="text-zinc-400">Estimators</span>
                <span className="text-blue-400 font-semibold">{nEstimators}</span>
              </div>
              <input 
                type="range" 
                min="10" 
                max="300" 
                step="10"
                value={nEstimators} 
                onChange={(e) => setNEstimators(Number(e.target.value))}
                className="w-full accent-blue-500 h-1 bg-zinc-800 rounded cursor-pointer"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[11px] text-zinc-400 block">Max Depth</label>
              <select 
                value={maxDepth} 
                onChange={(e) => setMaxDepth(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded p-1.5 text-xs text-zinc-300"
              >
                <option value="None">None (Unlimited)</option>
                <option value="3">3</option>
                <option value="5">5</option>
                <option value="8">8</option>
                <option value="10">10</option>
              </select>
            </div>

            <button 
              onClick={handleRetrain}
              disabled={retraining}
              className="w-full glow-btn flex items-center justify-center gap-2 text-xs py-2"
            >
              {retraining ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              {retraining ? 'Retraining...' : 'Retrain Pipeline'}
            </button>
          </div>

          {/* Quick Actions */}
          <button 
            onClick={loadBorderlineCase}
            className="w-full bg-zinc-900/60 hover:bg-zinc-900 border border-zinc-800 hover:border-amber-500/50 text-zinc-300 hover:text-amber-400 rounded-lg p-3 text-xs flex items-center justify-between transition-all"
          >
            <span className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-amber-500" />
              <span>Explain Borderline Case</span>
            </span>
            <ArrowRight className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Academic Footer */}
        <div className="text-[10px] text-zinc-600 space-y-1 pt-4 border-t border-zinc-900">
          <p>© 2026 Academic Research XAI Suite</p>
          <p>Explainable Artificial Intelligence (23AM501)</p>
        </div>
      </aside>

      {/* Main Content View */}
      <main className="flex-1 p-8 overflow-y-auto space-y-6">
        {/* Header & Academic Medical Safety Disclaimer Banner */}
        <div className="space-y-4">
          <header className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-zinc-100">
                Early Heart Attack Risk Prediction & XAI Console
              </h2>
              <p className="text-xs text-zinc-400 mt-1">
                Interpretable Machine Learning for Cardiovascular Risk Assessment in Young Adults (18–40).
              </p>
            </div>
            <div className="flex items-center gap-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1.5 rounded-full text-xs font-semibold">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
              <span>API Server Online</span>
            </div>
          </header>

          {/* Mandatory Academic Disclaimer Banner */}
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-3 flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="text-xs text-amber-200/90 leading-relaxed">
              <strong className="font-semibold text-amber-300">Academic & Research Disclaimer:</strong> This system is an academic Explainable AI prototype developed for research and decision-support demonstration. Predictions are statistical model estimates and should <strong>not</strong> be used for clinical medical diagnosis.
            </div>
          </div>
        </div>

        {/* Navigation Tabs Bar */}
        <nav className="flex border-b border-zinc-900 gap-2 overflow-x-auto pb-1">
          {[
            { id: 'overview', label: '01 — Overview' },
            { id: 'predict', label: '02 — Risk Prediction' },
            { id: 'dataset', label: '03 — Dataset Explorer' },
            { id: 'global', label: '04 — Global SHAP' },
            { id: 'local', label: '05 — Local Explanation' },
            { id: 'compare', label: '06 — SHAP vs LIME' },
            { id: 'young', label: '07 — 18–40 Analysis' },
            { id: 'metrics', label: '08 — Model Performance' },
            { id: 'about', label: '09 — About XAI' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-2 text-xs font-medium border-b-2 whitespace-nowrap transition-all ${
                activeTab === tab.id 
                  ? 'border-red-500 text-red-400 font-semibold bg-red-500/5 rounded-t-lg' 
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Tab 01: OVERVIEW */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-6 gap-4">
              <div className="glass-panel p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Total Patients</p>
                <h3 className="text-xl font-bold text-zinc-100 mt-1">{modelInfo?.total_dataset_size}</h3>
              </div>
              <div className="glass-panel p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">18–40 Young Adults</p>
                <h3 className="text-xl font-bold text-amber-400 mt-1">{modelInfo?.young_adult_total}</h3>
              </div>
              <div className="glass-panel p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Accuracy</p>
                <h3 className="text-xl font-bold text-emerald-400 mt-1">{(modelInfo?.accuracy * 100).toFixed(1)}%</h3>
              </div>
              <div className="glass-panel p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">F1-Score</p>
                <h3 className="text-xl font-bold text-blue-400 mt-1">{modelInfo?.f1_score.toFixed(3)}</h3>
              </div>
              <div className="glass-panel p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">ROC-AUC Score</p>
                <h3 className="text-xl font-bold text-purple-400 mt-1">{modelInfo?.roc_auc.toFixed(3)}</h3>
              </div>
              <div className="glass-panel p-4">
                <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">PyTorch Engine</p>
                <h3 className="text-xl font-bold text-rose-400 mt-1">{modelInfo?.has_pytorch ? 'Active' : 'Fallback'}</h3>
              </div>
            </div>

            {/* Three XAI Method Perspectives Banner */}
            <div className="glass-panel p-6">
              <h3 className="text-sm font-bold text-zinc-200 mb-2">Three-Perspective Explainable AI Architecture</h3>
              <p className="text-xs text-zinc-400 mb-6">
                This console interprets model decision-making through cooperative game theory, local surrogate linear modeling, and gradient-based neural network representations.
              </p>
              
              <div className="grid grid-cols-3 gap-6">
                <div className="border border-blue-500/20 bg-blue-500/5 rounded-xl p-4">
                  <div className="flex items-center gap-2 text-blue-400 font-bold text-sm mb-2">
                    <Globe className="h-4 w-4" />
                    <span>01. SHAP</span>
                  </div>
                  <p className="text-xs text-zinc-400">
                    Calculates Shapley values based on cooperative game theory to measure exact feature contribution towards predicted risk.
                  </p>
                </div>

                <div className="border border-emerald-500/20 bg-emerald-500/5 rounded-xl p-4">
                  <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm mb-2">
                    <Activity className="h-4 w-4" />
                    <span>02. LIME</span>
                  </div>
                  <p className="text-xs text-zinc-400">
                    Constructs an interpretable local surrogate linear model by sampling perturbed instances around the patient's data point.
                  </p>
                </div>

                <div className="border border-purple-500/20 bg-purple-500/5 rounded-xl p-4">
                  <div className="flex items-center gap-2 text-purple-400 font-bold text-sm mb-2">
                    <Zap className="h-4 w-4" />
                    <span>03. Tabular Grad-CAM</span>
                  </div>
                  <p className="text-xs text-zinc-400">
                    Uses backpropagated gradients of target output logits relative to intermediate neural network representations.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 02: RISK PREDICTION FORM */}
        {activeTab === 'predict' && (
          <div className="space-y-6">
            <div className="glass-panel p-6">
              <h3 className="text-sm font-bold text-zinc-200 mb-1">Interactive Patient Risk Prediction Form</h3>
              <p className="text-xs text-zinc-400 mb-6">Enter clinical feature parameters to calculate model-estimated cardiovascular risk and view live 3-way XAI explanations.</p>

              <form onSubmit={handleCustomPredict} className="grid grid-cols-4 gap-4 text-xs">
                <div>
                  <label className="text-zinc-400 block mb-1">Age (Years)</label>
                  <input type="number" min="18" max="75" value={predInput.age} onChange={e => setPredInput({...predInput, age: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100" />
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">Sex</label>
                  <select value={predInput.sex} onChange={e => setPredInput({...predInput, sex: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100">
                    <option value={0}>Female</option>
                    <option value={1}>Male</option>
                  </select>
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">Chest Pain Subtype</label>
                  <select value={predInput.chest_pain_type} onChange={e => setPredInput({...predInput, chest_pain_type: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100">
                    <option value={0}>Typical Angina</option>
                    <option value={1}>Atypical Angina</option>
                    <option value={2}>Non-anginal Pain</option>
                    <option value={3}>Asymptomatic</option>
                  </select>
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">Resting Blood Pressure (mmHg)</label>
                  <input type="number" min="80" max="220" value={predInput.resting_bp} onChange={e => setPredInput({...predInput, resting_bp: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100" />
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">Serum Cholesterol (mg/dL)</label>
                  <input type="number" min="100" max="500" value={predInput.cholesterol} onChange={e => setPredInput({...predInput, cholesterol: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100" />
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">Fasting Blood Sugar &gt; 120 mg/dL</label>
                  <select value={predInput.fasting_bs} onChange={e => setPredInput({...predInput, fasting_bs: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100">
                    <option value={0}>No (&le;120 mg/dL)</option>
                    <option value={1}>Yes (&gt;120 mg/dL)</option>
                  </select>
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">Resting ECG Results</label>
                  <select value={predInput.resting_ecg} onChange={e => setPredInput({...predInput, resting_ecg: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100">
                    <option value={0}>Normal</option>
                    <option value={1}>ST-T Wave Abnormality</option>
                    <option value={2}>LV Hypertrophy</option>
                  </select>
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">Max Heart Rate (bpm)</label>
                  <input type="number" min="60" max="220" value={predInput.max_heart_rate} onChange={e => setPredInput({...predInput, max_heart_rate: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100" />
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">Exercise Angina</label>
                  <select value={predInput.exercise_angina} onChange={e => setPredInput({...predInput, exercise_angina: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100">
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>
                <div>
                  <label className="text-zinc-400 block mb-1">ST Depression (mm)</label>
                  <input type="number" step="0.1" min="0" max="6" value={predInput.st_depression} onChange={e => setPredInput({...predInput, st_depression: Number(e.target.value)})} className="w-full bg-zinc-900 border border-zinc-800 rounded p-2 text-zinc-100" />
                </div>

                <div className="col-span-4 mt-2">
                  <button type="submit" disabled={predicting} className="glow-btn px-6 py-2 text-xs flex items-center gap-2">
                    {predicting ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Stethoscope className="h-4 w-4" />}
                    <span>Calculate Risk Probability & Generate XAI</span>
                  </button>
                </div>
              </form>
            </div>

            {/* Live Prediction Output */}
            {predResult && (
              <div className="space-y-6">
                <div className={`glass-panel p-6 border-l-4 ${predResult.pred_label === 1 ? 'border-l-red-500' : 'border-l-emerald-500'}`}>
                  <div className="flex justify-between items-center">
                    <div>
                      <span className="text-[10px] uppercase font-bold text-zinc-500">Model Risk Estimation Output</span>
                      <h3 className="text-xl font-bold mt-1 text-zinc-100">
                        {predResult.pred_label === 1 ? (
                          <span className="text-red-400">ELEVATED MODEL-ESTIMATED RISK</span>
                        ) : (
                          <span className="text-emerald-400">LOW MODEL-ESTIMATED RISK</span>
                        )}
                      </h3>
                      <p className="text-xs text-zinc-400 mt-1">
                        Estimated Risk Probability: <strong>{(predResult.risk_prob * 100).toFixed(1)}%</strong>
                        {predResult.is_young_adult && <span className="ml-2 text-amber-400 font-semibold">[Young Adult 18–40 Patient]</span>}
                      </p>
                    </div>
                  </div>
                </div>

                {/* 3-Way Live XAI Output */}
                <div className="grid grid-cols-3 gap-6">
                  {/* SHAP */}
                  <div className="glass-panel p-4">
                    <h4 className="text-xs font-bold text-blue-400 mb-3">SHAP Feature Attribution</h4>
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={predResult.shap.features.slice(0, 8)} layout="vertical" margin={{ top: 5, right: 10, left: 70, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                          <XAxis type="number" stroke="#71717a" fontSize={9} />
                          <YAxis dataKey="feature" type="category" stroke="#71717a" fontSize={9} />
                          <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '10px' }} />
                          <ReferenceLine x={0} stroke="#444" />
                          <Bar dataKey="shap_value">
                            {predResult.shap.features.slice(0, 8).map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.shap_value >= 0 ? '#ef4444' : '#22c55e'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* LIME */}
                  <div className="glass-panel p-4">
                    <h4 className="text-xs font-bold text-emerald-400 mb-3">LIME Local Surrogate Rules</h4>
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={predResult.lime.slice(0, 8)} layout="vertical" margin={{ top: 5, right: 10, left: 90, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                          <XAxis type="number" stroke="#71717a" fontSize={9} />
                          <YAxis dataKey="feature_rule" type="category" stroke="#71717a" fontSize={9} />
                          <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '10px' }} />
                          <ReferenceLine x={0} stroke="#444" />
                          <Bar dataKey="weight">
                            {predResult.lime.slice(0, 8).map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.weight >= 0 ? '#ef4444' : '#22c55e'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Grad-CAM */}
                  <div className="glass-panel p-4">
                    <h4 className="text-xs font-bold text-purple-400 mb-3">Tabular Grad-CAM Heatmap</h4>
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={predResult.gradcam.slice(0, 8)} layout="vertical" margin={{ top: 5, right: 10, left: 80, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                          <XAxis type="number" stroke="#71717a" fontSize={9} />
                          <YAxis dataKey="feature" type="category" stroke="#71717a" fontSize={9} />
                          <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '10px' }} />
                          <Bar dataKey="importance" fill="#a855f7" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 03: DATASET EXPLORER */}
        {activeTab === 'dataset' && (
          <div className="glass-panel p-6 space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-sm font-bold text-zinc-200">UCI Heart Disease Test Samples</h3>
                <p className="text-xs text-zinc-500">Searchable clinical dataset view. Click any patient to view explanations.</p>
              </div>
              <span className="text-xs text-zinc-400 font-mono">{testSamples.length} Test Instances</span>
            </div>

            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500 bg-zinc-900/50">
                    <th className="py-2.5 px-3">ID</th>
                    <th className="py-2.5 px-3">Age</th>
                    <th className="py-2.5 px-3">Cohort</th>
                    <th className="py-2.5 px-3">Actual Class</th>
                    <th className="py-2.5 px-3">Predicted Class</th>
                    <th className="py-2.5 px-3">Risk Prob</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-900 font-mono">
                  {testSamples.map(s => (
                    <tr key={s.id} className={`hover:bg-zinc-900/40 ${selectedIdx === s.id ? 'bg-blue-500/10' : ''}`}>
                      <td className="py-2.5 px-3 font-semibold">{s.id}</td>
                      <td className="py-2.5 px-3">{s.age} yrs</td>
                      <td className="py-2.5 px-3">
                        {s.is_young_adult ? (
                          <span className="px-1.5 py-0.5 rounded text-[9px] bg-amber-500/10 text-amber-400 border border-amber-500/20">18–40 Young</span>
                        ) : (
                          <span className="text-zinc-500 text-[10px]">&gt;40 Adult</span>
                        )}
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] ${s.true_label === 1 ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                          {modelInfo?.class_names[s.true_label]}
                        </span>
                      </td>
                      <td className="py-2.5 px-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] ${s.pred_label === 1 ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                          {modelInfo?.class_names[s.pred_label]}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-bold">{(s.risk_prob * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 text-right">
                        <button 
                          onClick={() => { fetchLocalExplanation(s.id); setActiveTab('local'); }} 
                          className="text-blue-400 hover:underline font-sans font-semibold"
                        >
                          Explain sample
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 04: GLOBAL SHAP */}
        {activeTab === 'global' && (
          <div className="glass-panel p-6 space-y-4">
            <div>
              <h3 className="text-sm font-bold text-zinc-200">Global SHAP Feature Importance</h3>
              <p className="text-xs text-zinc-400 mt-1">Features sorted by mean absolute Shapley value across the dataset.</p>
            </div>
            <div className="h-80 w-full mt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={globalShap} layout="vertical" margin={{ top: 5, right: 30, left: 120, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                  <XAxis type="number" stroke="#71717a" fontSize={10} />
                  <YAxis dataKey="feature" type="category" stroke="#71717a" fontSize={10} />
                  <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a' }} />
                  <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Tab 05: THREE-WAY LOCAL EXPLANATION */}
        {activeTab === 'local' && (
          <div className="space-y-6">
            {/* Instance Selector */}
            <div className="flex justify-between items-center bg-zinc-900/40 border border-zinc-800 p-3 rounded-xl">
              <div className="flex items-center gap-3">
                <button disabled={selectedIdx <= 0} onClick={() => fetchLocalExplanation(selectedIdx - 1)} className="p-1 rounded border border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-100 disabled:opacity-30">
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <span className="text-xs font-semibold">Patient Index: <span className="font-mono text-red-400">{selectedIdx}</span> / {testSamples.length - 1}</span>
                <button disabled={selectedIdx >= testSamples.length - 1} onClick={() => fetchLocalExplanation(selectedIdx + 1)} className="p-1 rounded border border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-100 disabled:opacity-30">
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>

              <div className="flex gap-4 text-xs font-semibold">
                <span>Age: <span className="font-mono text-amber-400">{localExplanation?.shap.features.find(f => f.feature === 'age')?.value} yrs</span></span>
                <span>True Class: <span className="font-mono text-zinc-300">{modelInfo?.class_names[localExplanation?.true_label]}</span></span>
                <span>Predicted Risk: <span className="font-mono text-red-400">{(localExplanation?.risk_prob * 100).toFixed(1)}%</span></span>
              </div>
            </div>

            {/* 3-Way Panels */}
            {localLoading ? (
              <div className="flex h-64 items-center justify-center text-xs text-zinc-500 gap-2">
                <RefreshCw className="h-4 w-4 animate-spin text-red-500" />
                <span>Computing SHAP, LIME & Grad-CAM explanations...</span>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-6">
                {/* SHAP Waterfall */}
                <div className="glass-panel p-4">
                  <h4 className="text-xs font-bold text-blue-400 mb-2">01. SHAP Waterfall Attributions</h4>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={localShapData} layout="vertical" margin={{ top: 5, right: 10, left: 90, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                        <XAxis type="number" stroke="#71717a" fontSize={9} />
                        <YAxis dataKey="display" type="category" stroke="#71717a" fontSize={9} />
                        <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '10px' }} />
                        <ReferenceLine x={0} stroke="#444" />
                        <Bar dataKey="shap">
                          {localShapData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.shap >= 0 ? '#ef4444' : '#22c55e'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* LIME */}
                <div className="glass-panel p-4">
                  <h4 className="text-xs font-bold text-emerald-400 mb-2">02. LIME Local Linear Surrogate</h4>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={localLimeData} layout="vertical" margin={{ top: 5, right: 10, left: 100, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                        <XAxis type="number" stroke="#71717a" fontSize={9} />
                        <YAxis dataKey="rule" type="category" stroke="#71717a" fontSize={9} />
                        <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '10px' }} />
                        <ReferenceLine x={0} stroke="#444" />
                        <Bar dataKey="weight">
                          {localLimeData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.weight >= 0 ? '#ef4444' : '#22c55e'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Grad-CAM */}
                <div className="glass-panel p-4">
                  <h4 className="text-xs font-bold text-purple-400 mb-2">03. Tabular Grad-CAM Heatmap</h4>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={localGradcamData} layout="vertical" margin={{ top: 5, right: 10, left: 90, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                        <XAxis type="number" stroke="#71717a" fontSize={9} />
                        <YAxis dataKey="display" type="category" stroke="#71717a" fontSize={9} />
                        <Tooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', fontSize: '10px' }} />
                        <Bar dataKey="importance" fill="#a855f7" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 06: SHAP VS LIME COMPARISON */}
        {activeTab === 'compare' && (
          <div className="glass-panel p-6 space-y-6">
            <div>
              <h3 className="text-sm font-bold text-zinc-200">SHAP vs. LIME Explanation Comparison</h3>
              <p className="text-xs text-zinc-400 mt-1">Directly comparing feature attributions for Patient Index {selectedIdx}.</p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500 bg-zinc-900/50">
                    <th className="py-2.5 px-3">Feature Name</th>
                    <th className="py-2.5 px-3 text-right">SHAP Value</th>
                    <th className="py-2.5 px-3 text-right">LIME Weight</th>
                    <th className="py-2.5 px-3">Direction Consistency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-900 font-mono">
                  {localShapData.map(shapItem => {
                    const limeMatch = localLimeData.find(l => l.rule.includes(shapItem.name));
                    const limeVal = limeMatch ? limeMatch.weight : 0;
                    const sameDir = (shapItem.shap >= 0 && limeVal >= 0) || (shapItem.shap < 0 && limeVal < 0);
                    return (
                      <tr key={shapItem.name}>
                        <td className="py-2 px-3 font-sans font-semibold text-zinc-300">{shapItem.name}</td>
                        <td className={`py-2 px-3 text-right font-bold ${shapItem.shap >= 0 ? 'text-red-400' : 'text-emerald-400'}`}>{shapItem.shap.toFixed(4)}</td>
                        <td className={`py-2 px-3 text-right font-bold ${limeVal >= 0 ? 'text-red-400' : 'text-emerald-400'}`}>{limeVal.toFixed(4)}</td>
                        <td className="py-2 px-3">
                          {sameDir ? (
                            <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">Consistent Direction</span>
                          ) : (
                            <span className="text-[10px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">Magnitude Disagreement</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 07: 18-40 YOUNG ADULT ANALYSIS */}
        {activeTab === 'young' && (
          <div className="space-y-6">
            <div className="glass-panel p-6">
              <h3 className="text-sm font-bold text-zinc-200 mb-1">18–40 Young Adult Risk Cohort Analysis</h3>
              <p className="text-xs text-zinc-400 mb-6">Specific evaluation metrics and risk distribution for young adult patients (aged 18–40).</p>

              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="border border-amber-500/30 bg-amber-500/5 p-4 rounded-xl">
                  <span className="text-[10px] text-amber-400 uppercase font-bold">Cohort Size</span>
                  <h4 className="text-xl font-bold text-zinc-100 mt-1">{modelInfo?.young_adult_total} patients</h4>
                </div>
                <div className="border border-zinc-800 p-4 rounded-xl">
                  <span className="text-[10px] text-zinc-500 uppercase font-bold">Accuracy on 18–40</span>
                  <h4 className="text-xl font-bold text-emerald-400 mt-1">{(modelInfo?.young_adult_accuracy * 100).toFixed(1)}%</h4>
                </div>
                <div className="border border-zinc-800 p-4 rounded-xl">
                  <span className="text-[10px] text-zinc-500 uppercase font-bold">F1-Score on 18–40</span>
                  <h4 className="text-xl font-bold text-blue-400 mt-1">{modelInfo?.young_adult_f1.toFixed(3)}</h4>
                </div>
                <div className="border border-zinc-800 p-4 rounded-xl">
                  <span className="text-[10px] text-zinc-500 uppercase font-bold">Elevated Risk Rate</span>
                  <h4 className="text-xl font-bold text-red-400 mt-1">{(modelInfo?.young_adult_pos_rate * 100).toFixed(1)}%</h4>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 08: MODEL PERFORMANCE */}
        {activeTab === 'metrics' && (
          <div className="space-y-6">
            <div className="glass-panel p-6">
              <h3 className="text-sm font-bold text-zinc-200 mb-4">Model Performance & Error Analysis</h3>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h4 className="text-xs font-bold text-zinc-400 mb-3">Confusion Matrix</h4>
                  {modelInfo && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="border border-zinc-800 p-4 rounded bg-zinc-950 flex flex-col items-center">
                        <span className="text-[10px] text-zinc-500 uppercase">True Negative (TN)</span>
                        <span className="text-2xl font-bold text-emerald-400">{modelInfo.confusion_matrix[0][0]}</span>
                      </div>
                      <div className="border border-zinc-800 p-4 rounded bg-zinc-950 flex flex-col items-center">
                        <span className="text-[10px] text-zinc-500 uppercase">False Positive (FP)</span>
                        <span className="text-2xl font-bold text-amber-400">{modelInfo.confusion_matrix[0][1]}</span>
                      </div>
                      <div className="border border-zinc-800 p-4 rounded bg-zinc-950 flex flex-col items-center">
                        <span className="text-[10px] text-zinc-500 uppercase">False Negative (FN)</span>
                        <span className="text-2xl font-bold text-red-500">{modelInfo.confusion_matrix[1][0]}</span>
                      </div>
                      <div className="border border-zinc-800 p-4 rounded bg-zinc-950 flex flex-col items-center">
                        <span className="text-[10px] text-zinc-500 uppercase">True Positive (TP)</span>
                        <span className="text-2xl font-bold text-blue-500">{modelInfo.confusion_matrix[1][1]}</span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="space-y-3 text-xs text-zinc-400">
                  <h4 className="text-xs font-bold text-zinc-200">Clinical Error Significance</h4>
                  <div className="bg-red-500/10 border border-red-500/20 p-3 rounded-lg">
                    <strong className="text-red-400 block mb-1">False Negatives (FN): {modelInfo?.confusion_matrix[1][0]}</strong>
                    Patients with elevated risk misclassified as low risk. In medical screening, minimizing FN is prioritized to prevent missed early intervention opportunities.
                  </div>
                  <div className="bg-amber-500/10 border border-amber-500/20 p-3 rounded-lg">
                    <strong className="text-amber-400 block mb-1">False Positives (FP): {modelInfo?.confusion_matrix[0][1]}</strong>
                    Low risk patients misclassified as elevated risk, resulting in unnecessary follow-up diagnostic testing.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 09: ABOUT XAI */}
        {activeTab === 'about' && (
          <div className="glass-panel p-6 space-y-6 text-xs text-zinc-300 leading-relaxed">
            <h3 className="text-base font-bold text-zinc-100">Academic Overview of XAI Methodologies</h3>
            
            <div className="grid grid-cols-3 gap-6">
              <div className="bg-zinc-900/50 border border-zinc-800 p-4 rounded-xl">
                <h4 className="font-bold text-blue-400 text-sm mb-2">SHAP (Shapley Additive exPlanations)</h4>
                <p>Grounded in cooperative game theory. Evaluates exact additive marginal contributions of each feature across all possible feature combinations.</p>
              </div>

              <div className="bg-zinc-900/50 border border-zinc-800 p-4 rounded-xl">
                <h4 className="font-bold text-emerald-400 text-sm mb-2">LIME (Local Interpretable Surrogates)</h4>
                <p>Constructs an interpretable linear model locally around a specific instance by drawing perturbed samples and weighting them by similarity.</p>
              </div>

              <div className="bg-zinc-900/50 border border-zinc-800 p-4 rounded-xl">
                <h4 className="font-bold text-purple-400 text-sm mb-2">Tabular Grad-CAM</h4>
                <p>Extends gradient-based activation mapping to tabular neural networks by measuring gradients of target logits with respect to hidden layer activations.</p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
