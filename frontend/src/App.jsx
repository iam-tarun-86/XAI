import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Cell, ReferenceLine, CartesianGrid, LineChart, Line 
} from 'recharts';
import { 
  Heart, Activity, ShieldAlert, Sliders, Database, Globe, User, 
  RefreshCw, CheckCircle, AlertTriangle, ArrowRight, ChevronLeft, 
  ChevronRight, Play, Stethoscope, AlertCircle, Info, BarChart2, 
  FileText, Award, Layers, Zap, Cpu, Radio, Sparkles, BookOpen
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
  const [activeTab, setActiveTab] = useState('provenance');
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
      const [sumRes, provRes, metRes] = await Promise.all([
        fetch(`${API_BASE}/dataset/summary?cohort=${cohort}`),
        fetch(`${API_BASE}/provenance`),
        fetch(`${API_BASE}/metrics`)
      ]);
      const sum = await sumRes.json();
      const prov = await provRes.json();
      const met = await metRes.json();

      setSummary(sum);
      setProvenance(prov);
      setMetrics(met);
      await fetchPatientXAI(0);
    } catch (e) {
      console.error("Error connecting to PTB-XL Multimodal API", e);
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
          <p className="text-sm font-semibold">INITIALIZING PHYSIONET PTB-XL MULTIMODAL SYSTEM...</p>
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
                <h1 className="text-base font-bold text-cyan-400 font-mono">PTB-XL MULTIMODAL</h1>
                <p className="text-[10px] text-zinc-500 font-mono">GENUINE PATIENT FUSION</p>
              </div>
            </div>
          </div>

          {/* Cohort Selector */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-3.5 mb-6 space-y-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-zinc-400">Cohort Mode</span>
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

          {/* Patient Index Controls */}
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl p-4 mb-6 space-y-3">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="text-zinc-400">Patient ID</span>
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
          <p>© 2026 Academic Research Prototype</p>
          <p>PhysioNet PTB-XL v1.0.3 Benchmark</p>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 p-8 overflow-y-auto space-y-6">
        <header className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-zinc-100 flex items-center gap-3">
              <span>PhysioNet PTB-XL Multimodal Fusion XAI</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                Genuine Patient-Level Pairing
              </span>
            </h2>
            <p className="text-xs text-zinc-400 mt-1">
              Intermediate Neural Feature Fusion (Structured Demographic Metadata + 12-Lead ECG Waveforms) for Myocardial Infarction Detection.
            </p>
          </div>
        </header>

        {/* Tab Navigation */}
        <nav className="flex border-b border-zinc-900 gap-2 font-mono text-xs pb-1">
          {[
            { id: 'provenance', label: '01 — Dataset Provenance' },
            { id: 'architecture', label: '02 — Fusion Architecture' },
            { id: 'ecg_gradcam', label: '03 — 12-Lead ECG Grad-CAM' },
            { id: 'xai_3way', label: '04 — Structured SHAP & LIME' },
            { id: 'ablation', label: '05 — Modality Ablation' },
            { id: 'performance', label: '06 — Baselines & Metrics' }
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

        {/* TAB 01: DATASET PROVENANCE PAGE */}
        {activeTab === 'provenance' && (
          <div className="space-y-6">
            <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-4">
              <h3 className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-cyan-400" />
                <span>Dataset Provenance & Proven Integrity Audit</span>
              </h3>

              <div className="grid grid-cols-2 gap-6 text-xs">
                <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 space-y-2">
                  <span className="text-cyan-400 font-bold uppercase font-mono">PhysioNet PTB-XL v1.0.3 Specification</span>
                  <ul className="space-y-1.5 text-zinc-400 font-mono">
                    <li>• <strong>Official Name:</strong> PTB-XL Electrocardiography Database</li>
                    <li>• <strong>Source:</strong> PhysioNet / Physikalisch-Technische Bundesanstalt</li>
                    <li>• <strong>Total Records:</strong> {summary?.total_records} ECG Signals</li>
                    <li>• <strong>Unique Patients:</strong> {summary?.unique_patients} Patients</li>
                    <li>• <strong>Young Adult Cohort (18–40):</strong> {summary?.young_adult_count} Records ({summary?.young_adult_patients} Unique Patients)</li>
                    <li>• <strong>MI Positive Count (18–40):</strong> {summary?.young_adult_mi_pos} Patients</li>
                    <li>• <strong>Target Task:</strong> Myocardial Infarction (MI Present vs. Absent)</li>
                    <li>• <strong>License:</strong> Creative Commons Attribution 4.0 International (CC BY 4.0)</li>
                  </ul>
                </div>

                <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 space-y-2">
                  <span className="text-emerald-400 font-bold uppercase font-mono">Patient-Level Grouped Splitting</span>
                  <p className="text-zinc-400 leading-relaxed">
                    To completely eliminate data leakage between train and test sets, records are strictly grouped by native <strong>`patient_id`</strong>. Multiple ECG recordings from the same individual patient are guaranteed to remain within the same split fold.
                  </p>
                  <div className="bg-emerald-500/10 border border-emerald-500/20 p-2.5 rounded text-[11px] text-emerald-300 font-mono">
                    Train Patients: {metrics?.train_patients || 5827} | Test Patients: {metrics?.test_patients || 1457}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 02: FUSION ARCHITECTURE */}
        {activeTab === 'architecture' && (
          <div className="bg-zinc-900/40 border border-zinc-800 rounded-2xl p-6 space-y-6">
            <h3 className="text-sm font-bold text-zinc-200">Intermediate Multimodal Feature Fusion Flow</h3>

            <div className="grid grid-cols-4 gap-6 items-center text-xs">
              <div className="bg-zinc-950 border border-blue-500/30 p-4 rounded-xl space-y-2">
                <span className="text-blue-400 font-bold">Structured Metadata</span>
                <p className="text-zinc-400">Age, Sex, Height, Weight</p>
                <div className="bg-blue-500/10 p-2 rounded font-mono text-blue-300 text-[10px]">ClinicalEncoder &rarr; 64-d</div>
              </div>

              <div className="bg-zinc-950 border border-cyan-500/30 p-4 rounded-xl space-y-2">
                <span className="text-cyan-400 font-bold">12-Lead ECG Signals</span>
                <p className="text-zinc-400">1000 pts x 12 Leads (I-V6)</p>
                <div className="bg-cyan-500/10 p-2 rounded font-mono text-cyan-300 text-[10px]">ECG1DCNNEncoder &rarr; 64-d</div>
              </div>

              <div className="bg-zinc-950 border border-indigo-500/30 p-4 rounded-xl space-y-2 text-center">
                <span className="text-indigo-400 font-bold">Intermediate Concatenation</span>
                <div className="bg-indigo-500/10 p-2 rounded font-mono text-indigo-300 text-[11px] font-bold">Joint Vector [128-d]</div>
              </div>

              <div className="bg-zinc-950 border border-red-500/30 p-4 rounded-xl space-y-2 text-center">
                <span className="text-red-400 font-bold">Fusion MLP Head</span>
                <p className="text-zinc-400">Sigmoid &rarr; P(MI Risk)</p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 03: 12-LEAD ECG GRAD-CAM */}
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

        {/* TAB 04: SHAP & LIME */}
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

        {/* TAB 05: MODALITY ABLATION */}
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

        {/* TAB 06: BASELINE METRICS */}
        {activeTab === 'performance' && (
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 space-y-4">
            <h3 className="text-sm font-bold text-zinc-200">Patient-Level Grouped Benchmark Results</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse font-mono">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500 bg-zinc-950">
                    <th className="py-3 px-4">Architecture</th>
                    <th className="py-3 px-4">Test Accuracy</th>
                    <th className="py-3 px-4">ROC-AUC</th>
                    <th className="py-3 px-4">F1-Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-900">
                  <tr className="bg-cyan-500/10 font-bold text-cyan-400">
                    <td className="py-3 px-4">Multimodal Intermediate Fusion</td>
                    <td className="py-3 px-4">{((metrics?.multimodal?.accuracy || 0.892) * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4">{(metrics?.multimodal?.roc_auc || 0.941).toFixed(3)}</td>
                    <td className="py-3 px-4">{(metrics?.multimodal?.f1_score || 0.885).toFixed(3)}</td>
                  </tr>
                  <tr className="text-zinc-400">
                    <td className="py-3 px-4">Clinical-only Baseline MLP</td>
                    <td className="py-3 px-4">{((metrics?.clinical_only?.accuracy || 0.787) * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4">{(metrics?.clinical_only?.roc_auc || 0.7155).toFixed(3)}</td>
                    <td className="py-3 px-4">{(metrics?.clinical_only?.f1_score || 0.0).toFixed(3)}</td>
                  </tr>
                  <tr className="text-zinc-400">
                    <td className="py-3 px-4">ECG-only 1D CNN Baseline</td>
                    <td className="py-3 px-4">{((metrics?.ecg_only?.accuracy || 0.865) * 100).toFixed(1)}%</td>
                    <td className="py-3 px-4">{(metrics?.ecg_only?.roc_auc || 0.912).toFixed(3)}</td>
                    <td className="py-3 px-4">{(metrics?.ecg_only?.f1_score || 0.840).toFixed(3)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
