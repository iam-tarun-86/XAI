import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Cell, ReferenceLine, CartesianGrid, LineChart, Line 
} from 'recharts';
import { 
  Heart, Activity, ShieldAlert, Sliders, Database, Globe, User, 
  RefreshCw, CheckCircle, AlertTriangle, ArrowRight, ChevronLeft, 
  ChevronRight, Play, Stethoscope, AlertCircle, Info, BarChart2, 
  FileText, Award, Layers, Zap, Cpu, Radio, Sparkles, BookOpen, Layers3
} from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [cohort, setCohort] = useState('all');
  const [summary, setSummary] = useState(null);
  const [provenance, setProvenance] = useState(null);
  const [patientIdx, setPatientIdx] = useState(0);
  const [patientData, setPatientData] = useState(null);
  const [gradcamData, setGradcamData] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [limeData, setLimeData] = useState(null);
  const [ablationData, setAblationData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [multiMetrics, setMultiMetrics] = useState(null);
  const [activeTab, setActiveTab] = useState('multi_dataset');
  const [loading, setLoading] = useState(true);
  const [showGradcamOverlay, setShowGradcamOverlay] = useState(true);

  useEffect(() => {
    fetchInitialData();
  }, [cohort]);

  useEffect(() => {
    if (patientIdx !== null) {
      fetchPatientXAI(patientIdx);
    }
  }, [patientIdx]);

  const fetchInitialData = async () => {
    setLoading(true);
    try {
      const [sumRes, provRes, metRes, multiMetRes] = await Promise.all([
        fetch(`${API_BASE}/dataset/summary?cohort=${cohort}`),
        fetch(`${API_BASE}/provenance`),
        fetch(`${API_BASE}/metrics`),
        fetch(`${API_BASE}/multi-dataset-metrics`)
      ]);
      setSummary(await sumRes.json());
      setProvenance(await provRes.json());
      setMetrics(await metRes.json());
      setMultiMetrics(await multiMetRes.json());
      await fetchPatientXAI(0);
    } catch (e) {
      console.error("Error fetching multi-dataset API endpoints", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchPatientXAI = async (idx) => {
    try {
      const [pRes, gcRes, spRes, lmRes, abRes] = await Promise.all([
        fetch(`${API_BASE}/patient/${idx}`),
        fetch(`${API_BASE}/explain/gradcam-1d`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_index: idx })
        }),
        fetch(`${API_BASE}/explain/shap`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_index: idx })
        }),
        fetch(`${API_BASE}/explain/lime`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_index: idx })
        }),
        fetch(`${API_BASE}/explain/ablation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_index: idx })
        })
      ]);

      setPatientData(await pRes.json());
      setGradcamData(await gcRes.json());
      setShapData(await spRes.json());
      setLimeData(await lmRes.json());
      setAblationData(await abRes.json());
    } catch (e) {
      console.error("Error fetching patient XAI", e);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-zinc-950 text-cyan-400 font-mono">
        <div className="flex flex-col items-center gap-4">
          <Activity className="h-10 w-10 animate-spin text-cyan-400" />
          <p className="text-sm font-semibold">INITIALIZING MULTI-DATASET RESEARCH SYSTEM...</p>
        </div>
      </div>
    );
  }

  const ecgChartData = (gradcamData?.ecg_signal || []).slice(0, 500).map((amplitude, i) => ({
    time: i,
    amplitude: amplitude,
    gradcam: (gradcamData?.gradcam_heatmap_1d || [])[i] || 0
  }));

  return (
    <div className="flex min-h-screen bg-zinc-950 text-zinc-100 font-sans">
      {/* Sidebar */}
      <aside className="w-80 border-r border-zinc-900 bg-zinc-950 p-6 flex flex-col justify-between">
        <div>
          <div className="mb-6">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-cyan-500 flex items-center justify-center">
                <Activity className="h-5 w-5 text-zinc-950" />
              </div>
              <div>
                <h1 className="text-base font-bold text-cyan-400 font-mono">MULTI-DATASET XAI</h1>
                <p className="text-[10px] text-zinc-500 font-mono">PTB-XL FUSION + UCI VALIDATION</p>
              </div>
            </div>
          </div>

          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-3.5 mb-6 space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-zinc-400">Cohort Filter</span>
              <span className="font-mono text-cyan-400 font-bold">{cohort === 'young' ? '18–40 Young Adults' : 'All Patients'}</span>
            </div>
            <div className="grid grid-cols-2 gap-1.5 p-1 bg-zinc-950 rounded-lg text-[11px] font-semibold">
              <button onClick={() => setCohort('all')} className={`py-1 rounded ${cohort === 'all' ? 'bg-cyan-500/20 text-cyan-400' : 'text-zinc-500'}`}>
                All ({summary?.total_records})
              </button>
              <button onClick={() => setCohort('young')} className={`py-1 rounded ${cohort === 'young' ? 'bg-amber-500/20 text-amber-400' : 'text-zinc-500'}`}>
                18–40 ({summary?.young_adult_count})
              </button>
            </div>
          </div>

          <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 mb-6 space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400">Selected Patient</span>
              <span className="text-cyan-400 font-bold">#{patientData?.patient_id}</span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <button disabled={patientIdx <= 0} onClick={() => setPatientIdx(patientIdx - 1)} className="flex-1 py-1.5 bg-zinc-950 border border-zinc-800 rounded text-xs font-semibold disabled:opacity-40">
                Previous
              </button>
              <button disabled={patientIdx >= (summary?.total_records || 8128) - 1} onClick={() => setPatientIdx(patientIdx + 1)} className="flex-1 py-1.5 bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 rounded text-xs font-semibold disabled:opacity-40">
                Next Record
              </button>
            </div>
          </div>
        </div>

        <div className="text-[10px] text-zinc-600 border-t border-zinc-900 pt-3">
          <p>© 2026 Academic Multi-Dataset Research</p>
          <p>PTB-XL Multimodal + UCI External Benchmark</p>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 p-8 overflow-y-auto space-y-6">
        <header className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-zinc-100 flex items-center gap-3">
              <span>Multi-Dataset Early Heart Attack Risk & XAI Console</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                PTB-XL Primary + UCI External Validation
              </span>
            </h2>
            <p className="text-xs text-zinc-400 mt-1">
              Primary PTB-XL Multimodal Intermediate Neural Fusion with Independent UCI Heart Disease External Feature Validation.
            </p>
          </div>
        </header>

        {/* Navigation Bar */}
        <nav className="flex border-b border-zinc-900 gap-2 font-mono text-xs pb-1">
          {[
            { id: 'multi_dataset', label: '01 — Multi-Dataset Validation' },
            { id: 'faculty_qa', label: '02 — Faculty Methodology Q&A' },
            { id: 'architecture', label: '03 — Fusion Flow' },
            { id: 'ecg_gradcam', label: '04 — 12-Lead ECG Grad-CAM' },
            { id: 'xai_3way', label: '05 — SHAP & LIME' },
            { id: 'ablation', label: '06 — Modality Ablation' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-xs font-medium border-b-2 transition-all ${
                activeTab === tab.id 
                  ? 'border-cyan-400 text-cyan-400 font-semibold bg-cyan-500/5 rounded-t-lg' 
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* TAB 01: MULTI-DATASET VALIDATION & COMPARISON */}
        {activeTab === 'multi_dataset' && (
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-6">
              <div>
                <h3 className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                  <Layers3 className="h-4 w-4 text-cyan-400" />
                  <span>Multi-Dataset Research Matrix & External Validation</span>
                </h3>
                <p className="text-xs text-zinc-400 mt-1">
                  Comparing primary multimodal dataset (PTB-XL) against complementary independent external dataset (UCI Heart Disease).
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse font-mono">
                  <thead>
                    <tr className="border-b border-zinc-800 text-zinc-500 bg-zinc-950">
                      <th className="py-3 px-4">Dataset Name</th>
                      <th className="py-3 px-4">Official Source</th>
                      <th className="py-3 px-4">Record Count</th>
                      <th className="py-3 px-4">18–40 Young Adults</th>
                      <th className="py-3 px-4">ECG Availability</th>
                      <th className="py-3 px-4">Target Label</th>
                      <th className="py-3 px-4">Defensible Scientific Role</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-900">
                    <tr className="bg-cyan-500/10 font-bold text-cyan-400">
                      <td className="py-3 px-4">PhysioNet PTB-XL v1.0.3</td>
                      <td className="py-3 px-4">PhysioNet (PTB Germany)</td>
                      <td className="py-3 px-4">8,128 Records</td>
                      <td className="py-3 px-4">1,282 Records (1,206 Patients)</td>
                      <td className="py-3 px-4">12-Lead ($1000 \times 12$)</td>
                      <td className="py-3 px-4">Myocardial Infarction (MI)</td>
                      <td className="py-3 px-4 text-emerald-400 font-sans font-semibold">Primary Multimodal Fusion</td>
                    </tr>
                    <tr className="text-zinc-400">
                      <td className="py-3 px-4">UCI Heart Disease Dataset</td>
                      <td className="py-3 px-4">UCI ML Repository</td>
                      <td className="py-3 px-4">920 Records</td>
                      <td className="py-3 px-4">93 Records</td>
                      <td className="py-3 px-4">None (Tabular only)</td>
                      <td className="py-3 px-4">CAD Narrowing (&gt;50%)</td>
                      <td className="py-3 px-4 text-amber-400 font-sans font-semibold">Independent External Validation</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 02: FACULTY METHODOLOGY Q&A */}
        {activeTab === 'faculty_qa' && (
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-6">
              <h3 className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-cyan-400" />
                <span>Faculty Review: Scientific & Methodological Explanations</span>
              </h3>

              <div className="space-y-4 text-xs">
                <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 space-y-2">
                  <h4 className="font-bold text-cyan-400 text-sm">Q1: Why use multiple datasets for this research?</h4>
                  <p className="text-zinc-400 leading-relaxed">
                    Different biomedical datasets contain complementary clinical information and represent diverse populations. We retain the <strong>UCI Heart Disease dataset</strong> for independent clinical feature comparison and external validation, while using <strong>PTB-XL</strong> for multimodal learning.
                  </p>
                </div>

                <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 space-y-2">
                  <h4 className="font-bold text-emerald-400 text-sm">Q2: Why is PTB-XL the primary dataset for Multimodal Fusion?</h4>
                  <p className="text-zinc-400 leading-relaxed">
                    PTB-XL natively provides 12-lead raw ECG signal waveforms ($1000 \times 12$) and structured demographic metadata linked through <strong>genuine patient identifiers (`patient_id`)</strong>. This allows legitimate, un-fabricated multimodal learning without artificial row-to-row pairings.
                  </p>
                </div>

                <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 space-y-2">
                  <h4 className="font-bold text-purple-400 text-sm">Q3: How is data leakage prevented during multimodal training?</h4>
                  <p className="text-zinc-400 leading-relaxed">
                    Splitting is strictly conducted using <strong>Patient-Level Grouping</strong>. All ECG recordings belonging to any individual patient are guaranteed to remain within the same split fold, ensuring zero test set leakage.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 03: FUSION ARCHITECTURE */}
        {activeTab === 'architecture' && (
          <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 space-y-6">
            <h3 className="text-sm font-bold text-zinc-200">Multimodal Feature Fusion Architecture</h3>
            <div className="grid grid-cols-4 gap-6 text-xs">
              <div className="bg-zinc-950 border border-blue-500/30 p-4 rounded-xl space-y-2">
                <span className="text-blue-400 font-bold">PTB-XL Structured Metadata</span>
                <p className="text-zinc-400">Age, Sex, Height, Weight</p>
                <div className="bg-blue-500/10 p-2 rounded font-mono text-blue-300 text-[10px]">ClinicalEncoder &rarr; 64-d</div>
              </div>

              <div className="bg-zinc-950 border border-cyan-500/30 p-4 rounded-xl space-y-2">
                <span className="text-cyan-400 font-bold">PTB-XL 12-Lead ECG</span>
                <p className="text-zinc-400">1000 pts x 12 Leads</p>
                <div className="bg-cyan-500/10 p-2 rounded font-mono text-cyan-300 text-[10px]">ECG1DCNNEncoder &rarr; 64-d</div>
              </div>

              <div className="bg-zinc-950 border border-indigo-500/30 p-4 rounded-xl space-y-2 text-center">
                <span className="text-indigo-400 font-bold">Joint Embedding</span>
                <div className="bg-indigo-500/10 p-2 rounded font-mono text-indigo-300 text-[11px] font-bold">Concatenated [128-d]</div>
              </div>

              <div className="bg-zinc-950 border border-red-500/30 p-4 rounded-xl space-y-2 text-center">
                <span className="text-red-400 font-bold">Prediction Head</span>
                <p className="text-zinc-400">Sigmoid &rarr; P(MI Risk)</p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 04: 12-LEAD ECG GRAD-CAM */}
        {activeTab === 'ecg_gradcam' && (
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-4">
            <h3 className="text-sm font-bold text-zinc-200">12-Lead ECG 1D CNN Grad-CAM Overlay</h3>
            <div className="h-80 w-full bg-zinc-950 rounded-xl p-4 border border-zinc-900">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ecgChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#18181b" />
                  <XAxis dataKey="time" stroke="#52525b" fontSize={10} tickFormatter={t => `${t}ms`} />
                  <YAxis stroke="#52525b" fontSize={10} domain={[-1.0, 2.0]} />
                  <Tooltip contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="amplitude" stroke="#06b6d4" strokeWidth={2} dot={false} name="Lead II Signal (mV)" />
                  <Line type="monotone" dataKey="gradcam" stroke="#ef4444" strokeWidth={2} strokeDasharray="4 4" dot={false} name="1D Grad-CAM Activation" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* TAB 05: SHAP & LIME */}
        {activeTab === 'xai_3way' && (
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-5 space-y-3">
              <h4 className="text-xs font-bold text-blue-400">Structured Clinical SHAP Values</h4>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData?.features || []} layout="vertical" margin={{ top: 5, right: 10, left: 70, bottom: 5 }}>
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

            <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-5 space-y-3">
              <h4 className="text-xs font-bold text-emerald-400">Structured Clinical LIME Rules</h4>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={limeData?.rules || []} layout="vertical" margin={{ top: 5, right: 10, left: 100, bottom: 5 }}>
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
        )}

        {/* TAB 06: MODALITY ABLATION */}
        {activeTab === 'ablation' && (
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-6">
            <h3 className="text-sm font-bold text-zinc-200">Leave-One-Modality-Out Ablation Contribution</h3>
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-zinc-950 border border-blue-500/30 p-6 rounded-xl space-y-2">
                <div className="flex justify-between items-center text-xs font-bold text-blue-400">
                  <span>Structured Clinical Metadata</span>
                  <span className="font-mono text-lg">{ablationData?.clinical_impact_pct}%</span>
                </div>
                <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                  <div className="bg-blue-500 h-full rounded-full" style={{ width: `${ablationData?.clinical_impact_pct}%` }}></div>
                </div>
              </div>

              <div className="bg-zinc-950 border border-cyan-500/30 p-6 rounded-xl space-y-2">
                <div className="flex justify-between items-center text-xs font-bold text-cyan-400">
                  <span>12-Lead ECG Signal Modality</span>
                  <span className="font-mono text-lg">{ablationData?.ecg_impact_pct}%</span>
                </div>
                <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                  <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${ablationData?.ecg_impact_pct}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
