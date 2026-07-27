import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Cell, ReferenceLine, CartesianGrid 
} from 'recharts';
import { 
  Sliders, Database, Globe, User, ShieldAlert, 
  RefreshCw, CheckCircle, AlertTriangle, ArrowRight, 
  ChevronLeft, ChevronRight, Play 
} from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [testSamples, setTestSamples] = useState([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [localExplanation, setLocalExplanation] = useState(null);
  const [globalShap, setGlobalShap] = useState([]);
  const [activeTab, setActiveTab] = useState('dataset');
  const [loading, setLoading] = useState(true);
  const [localLoading, setLocalLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);

  // Form states
  const [nEstimators, setNEstimators] = useState(100);
  const [maxDepth, setMaxDepth] = useState('None');

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

      // Load first explanation
      await fetchLocalExplanation(0);
    } catch (e) {
      console.error("Error fetching data", e);
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
      console.error("Error fetching local expl", e);
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

      // Reload dataset and global explanations
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
          <RefreshCw className="h-10 w-10 animate-spin text-blue-500" />
          <p className="text-sm font-medium">Initializing XAI Environment & Wine Dataset...</p>
        </div>
      </div>
    );
  }

  // Pre-process local SHAP data for waterfall/bar representation
  const localShapData = localExplanation?.shap?.features.map(f => ({
    name: f.feature,
    val: f.value,
    shap: f.shap_value,
    display: `${f.feature} (${f.value.toFixed(2)})`
  })) || [];

  const localLimeData = localExplanation?.lime.map(l => ({
    rule: l.feature_rule,
    weight: l.weight
  })) || [];

  return (
    <div className="flex min-h-screen bg-zinc-950 text-zinc-100">
      {/* Sidebar Panel */}
      <aside className="w-80 border-r border-zinc-900 bg-zinc-950/80 p-6 flex flex-col justify-between">
        <div>
          {/* Logo & Header */}
          <div className="mb-8">
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 bg-clip-text text-transparent">
              🔬 XAI Lab Console
            </h1>
            <p className="text-xs text-zinc-500 mt-1">SHAP & LIME Interpretability Suite</p>
          </div>

          {/* Model Config Panel */}
          <div className="glass-panel p-5 space-y-5">
            <div className="flex items-center gap-2 text-zinc-300 font-semibold text-sm">
              <Sliders className="h-4 w-4 text-blue-500" />
              <span>Random Forest Config</span>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-xs">
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
                className="w-full accent-blue-500 h-1 bg-zinc-800 rounded-lg cursor-pointer"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs text-zinc-400 block">Max Depth</label>
              <select 
                value={maxDepth} 
                onChange={(e) => setMaxDepth(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-md p-1.5 text-xs focus:ring-1 focus:ring-blue-500"
              >
                <option value="None">None (Unlimited)</option>
                <option value="3">3</option>
                <option value="5">5</option>
                <option value="8">8</option>
                <option value="10">10</option>
                <option value="15">15</option>
              </select>
            </div>

            <button 
              onClick={handleRetrain}
              disabled={retraining}
              className="w-full glow-btn flex items-center justify-center gap-2 text-xs py-2"
            >
              {retraining ? (
                <RefreshCw className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              {retraining ? 'Retraining...' : 'Retrain & Re-explain'}
            </button>
          </div>

          {/* Quick Actions */}
          <div className="mt-6 space-y-2">
            <button 
              onClick={loadBorderlineCase}
              className="w-full bg-zinc-900/50 hover:bg-zinc-900 border border-zinc-800/80 hover:border-amber-500/50 text-zinc-300 hover:text-amber-400 rounded-lg p-3 text-xs flex items-center justify-between transition-all"
            >
              <span className="flex items-center gap-2">
                <ShieldAlert className="h-4 w-4 text-amber-500" />
                <span>Explain Borderline Case</span>
              </span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Footer Academic Info */}
        <div className="text-[10px] text-zinc-600 space-y-1">
          <p>© 2026 Sri Krishna College of Technology</p>
          <p>Explainable AI Lab (23AM501)</p>
        </div>
      </aside>

      {/* Main Panel Content */}
      <main className="flex-1 p-8 overflow-y-auto space-y-6">
        {/* Main Header & Status */}
        <header className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">XAI Wine Classification Suite</h2>
            <p className="text-xs text-zinc-500 mt-1">Comparing cooperative game theory (SHAP) with local linear surrogates (LIME).</p>
          </div>
          <div className="flex items-center gap-2 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1.5 rounded-full text-xs font-semibold">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping"></span>
            <span>API Engine Online</span>
          </div>
        </header>

        {/* KPIs Cards */}
        <section className="grid grid-cols-4 gap-4">
          <div className="glass-panel p-5">
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Model Accuracy</p>
            <h3 className="text-2xl font-bold text-zinc-100 mt-1">{(modelInfo?.accuracy * 100).toFixed(1)}%</h3>
          </div>
          <div className="glass-panel p-5">
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">F1-Score (Weighted)</p>
            <h3 className="text-2xl font-bold text-zinc-100 mt-1">{modelInfo?.f1_score.toFixed(3)}</h3>
          </div>
          <div className="glass-panel p-5">
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Dataset split</p>
            <h3 className="text-2xl font-bold text-zinc-100 mt-1">{modelInfo?.train_size} / {modelInfo?.test_size} <span className="text-xs text-zinc-600 font-normal">samples</span></h3>
          </div>
          <div className="glass-panel p-5">
            <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Target Problem</p>
            <h3 className="text-2xl font-bold text-zinc-100 mt-1 text-blue-500">Binary <span className="text-xs text-zinc-500 font-normal">Classification</span></h3>
          </div>
        </section>

        {/* Tabs Bar */}
        <section>
          <div className="flex border-b border-zinc-900 gap-6">
            {[
              { id: 'dataset', label: 'Dataset Explorer', icon: Database },
              { id: 'global', label: 'Global Explanations (SHAP)', icon: Globe },
              { id: 'local', label: 'Local Explanations (SHAP vs LIME)', icon: User }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 pb-3 text-sm font-medium border-b-2 transition-all ${
                  activeTab === tab.id 
                    ? 'border-blue-500 text-blue-500 font-semibold' 
                    : 'border-transparent text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <tab.icon className="h-4 w-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </section>

        {/* Tab Contents */}
        <section className="min-h-[400px]">
          {/* TAB 1: DATASET EXPLORER */}
          {activeTab === 'dataset' && (
            <div className="space-y-6">
              <div className="glass-panel p-6">
                <h4 className="text-sm font-bold text-zinc-200 mb-4">Wine Dataset Test Samples</h4>
                <div className="overflow-x-auto max-h-96">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-zinc-800 text-zinc-500">
                        <th className="py-3 px-4 font-semibold">Index</th>
                        <th className="py-3 px-4 font-semibold">True Class</th>
                        <th className="py-3 px-4 font-semibold">Prediction</th>
                        <th className="py-3 px-4 font-semibold">Confidence</th>
                        <th className="py-3 px-4 font-semibold">Key Indicators</th>
                        <th className="py-3 px-4 font-semibold text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-900">
                      {testSamples.map(sample => (
                        <tr 
                          key={sample.id} 
                          className={`hover:bg-zinc-900/30 transition-all ${selectedIdx === sample.id ? 'bg-blue-500/5' : ''}`}
                        >
                          <td className="py-3 px-4 font-mono font-medium">{sample.id}</td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                              sample.true_label === 0 ? 'bg-blue-500/10 text-blue-400' : 'bg-purple-500/10 text-purple-400'
                            }`}>
                              {modelInfo?.class_names[sample.true_label]}
                            </span>
                          </td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                              sample.true_label === sample.pred_label 
                                ? 'bg-emerald-500/10 text-emerald-400' 
                                : 'bg-red-500/10 text-red-400'
                            }`}>
                              {modelInfo?.class_names[sample.pred_label]}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-medium">{(sample.confidence * 100).toFixed(1)}%</td>
                          <td className="py-3 px-4 text-zinc-500 font-mono truncate max-w-xs">
                            Alc: {sample.alcohol?.toFixed(1)}% | Color: {sample.color_intensity?.toFixed(1)}
                          </td>
                          <td className="py-3 px-4 text-right">
                            <button 
                              onClick={() => {
                                fetchLocalExplanation(sample.id);
                                setActiveTab('local');
                              }}
                              className="text-blue-500 hover:text-blue-400 font-semibold hover:underline"
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

              {/* Confusion Matrix Block */}
              <div className="grid grid-cols-2 gap-6">
                <div className="glass-panel p-6 flex flex-col justify-between">
                  <div>
                    <h4 className="text-sm font-bold text-zinc-200">Confusion Matrix</h4>
                    <p className="text-xs text-zinc-500 mt-1">Visualizing correct predictions vs misclassifications.</p>
                  </div>
                  {modelInfo && (
                    <div className="grid grid-cols-2 gap-4 mt-6 max-w-xs mx-auto">
                      <div className="border border-zinc-800 p-4 rounded bg-zinc-950 flex flex-col items-center">
                        <span className="text-[10px] text-zinc-500 uppercase">True Type 0 (TN)</span>
                        <span className="text-2xl font-bold text-blue-500">{modelInfo.confusion_matrix[0][0]}</span>
                      </div>
                      <div className="border border-zinc-800 p-4 rounded bg-zinc-950 flex flex-col items-center">
                        <span className="text-[10px] text-zinc-500 uppercase">False Others (FP)</span>
                        <span className="text-2xl font-bold text-red-500">{modelInfo.confusion_matrix[0][1]}</span>
                      </div>
                      <div className="border border-zinc-800 p-4 rounded bg-zinc-950 flex flex-col items-center">
                        <span className="text-[10px] text-zinc-500 uppercase">False Type 0 (FN)</span>
                        <span className="text-2xl font-bold text-red-500">{modelInfo.confusion_matrix[1][0]}</span>
                      </div>
                      <div className="border border-zinc-800 p-4 rounded bg-zinc-950 flex flex-col items-center">
                        <span className="text-[10px] text-zinc-500 uppercase">True Others (TP)</span>
                        <span className="text-2xl font-bold text-purple-500">{modelInfo.confusion_matrix[1][1]}</span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="glass-panel p-6 flex flex-col justify-between">
                  <div>
                    <h4 className="text-sm font-bold text-zinc-200">Wine Feature Characteristics</h4>
                    <p className="text-xs text-zinc-500 mt-1">Chemical compositions analyzed for explainable model inputs.</p>
                  </div>
                  <div className="text-xs text-zinc-400 space-y-3 mt-4">
                    <p>🧪 <strong>Alcohol</strong>: Primary indicator for wine class separation.</p>
                    <p>🎨 <strong>Color Intensity</strong>: Visual attribute correlating highly with target groupings.</p>
                    <p>⛰️ <strong>Proline</strong>: Nitrogen compound crucial for model classification decision paths.</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: GLOBAL EXPLANATIONS */}
          {activeTab === 'global' && (
            <div className="space-y-6">
              <div className="glass-panel p-6">
                <div className="mb-4">
                  <h4 className="text-sm font-bold text-zinc-200">Global Feature Importance (SHAP)</h4>
                  <p className="text-xs text-zinc-500 mt-1">Average contribution magnitude (mean absolute SHAP values) across the entire test subset.</p>
                </div>
                <div className="h-80 w-full mt-6">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={globalShap}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                      <XAxis type="number" stroke="#71717a" fontSize={10} />
                      <YAxis dataKey="feature" type="category" stroke="#71717a" fontSize={10} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a' }}
                        labelStyle={{ color: '#f4f4f5', fontSize: '12px' }}
                      />
                      <Bar dataKey="importance" fill="#2563eb" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: LOCAL EXPLANATIONS */}
          {activeTab === 'local' && (
            <div className="space-y-6">
              {/* Instance Navigation Header */}
              <div className="flex justify-between items-center bg-zinc-900/30 border border-zinc-800/80 p-4 rounded-xl">
                <div className="flex items-center gap-3">
                  <button 
                    disabled={selectedIdx <= 0}
                    onClick={() => fetchLocalExplanation(selectedIdx - 1)}
                    className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-100 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </button>
                  <span className="text-xs font-semibold text-zinc-300">
                    Sample Index: <span className="font-mono text-blue-500 text-sm">{selectedIdx}</span> / {testSamples.length - 1}
                  </span>
                  <button 
                    disabled={selectedIdx >= testSamples.length - 1}
                    onClick={() => fetchLocalExplanation(selectedIdx + 1)}
                    className="p-1.5 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-400 hover:text-zinc-100 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                  >
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>

                <div className="flex gap-4 text-xs font-semibold">
                  <div>
                    <span className="text-zinc-500">True Class: </span>
                    <span className={`px-2 py-0.5 rounded ${
                      localExplanation?.true_label === 0 ? 'bg-blue-500/10 text-blue-400' : 'bg-purple-500/10 text-purple-400'
                    }`}>
                      {modelInfo?.class_names[localExplanation?.true_label]}
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500">Prediction: </span>
                    <span className={`px-2 py-0.5 rounded ${
                      localExplanation?.true_label === localExplanation?.pred_label 
                        ? 'bg-emerald-500/10 text-emerald-400' 
                        : 'bg-red-500/10 text-red-400'
                    }`}>
                      {modelInfo?.class_names[localExplanation?.pred_label]}
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500">Probs: </span>
                    <span className="font-mono text-zinc-300">
                      [{localExplanation?.probs[0].toFixed(2)}, {localExplanation?.probs[1].toFixed(2)}]
                    </span>
                  </div>
                </div>
              </div>

              {localLoading ? (
                <div className="flex h-64 w-full items-center justify-center text-zinc-500 text-xs gap-2">
                  <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />
                  <span>Computing Shapley and Surrogate metrics...</span>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-6">
                  {/* SHAP waterfall representation */}
                  <div className="glass-panel p-6">
                    <div>
                      <h4 className="text-sm font-bold text-zinc-200">SHAP Waterfall Explanation</h4>
                      <p className="text-xs text-zinc-500 mt-1">Shows feature value attributions pulling predictions towards or away from the base value ({localExplanation?.shap.base_value.toFixed(2)}).</p>
                    </div>
                    
                    <div className="h-72 w-full mt-6">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={localShapData}
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                          <XAxis type="number" stroke="#71717a" fontSize={10} />
                          <YAxis dataKey="display" type="category" stroke="#71717a" fontSize={10} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a' }}
                            labelStyle={{ color: '#f4f4f5', fontSize: '12px' }}
                          />
                          <ReferenceLine x={0} stroke="#444" />
                          <Bar dataKey="shap">
                            {localShapData.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={entry.shap >= 0 ? '#22c55e' : '#ef4444'} 
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* LIME weights representation */}
                  <div className="glass-panel p-6">
                    <div>
                      <h4 className="text-sm font-bold text-zinc-200">LIME Local Surrogates</h4>
                      <p className="text-xs text-zinc-500 mt-1">Weights derived from a local linear model representing decision rules around this sample's neighborhood.</p>
                    </div>

                    <div className="h-72 w-full mt-6">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={localLimeData}
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
                          <XAxis type="number" stroke="#71717a" fontSize={10} />
                          <YAxis dataKey="rule" type="category" stroke="#71717a" fontSize={10} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a' }}
                            labelStyle={{ color: '#f4f4f5', fontSize: '12px' }}
                          />
                          <ReferenceLine x={0} stroke="#444" />
                          <Bar dataKey="weight">
                            {localLimeData.map((entry, index) => (
                              <Cell 
                                key={`cell-${index}`} 
                                fill={entry.weight >= 0 ? '#22c55e' : '#ef4444'} 
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
