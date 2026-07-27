import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap
import lime
import lime.lime_tabular

# 1. Page Configuration
st.set_page_config(
    page_title="XAI Dashboard: SHAP vs. LIME",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Theme State Management
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# 3. CSS Styling (Modern Glassmorphic / Zinc Theme)
bg_color = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card_color = "#0c0c0f" if IS_DARK else "#ffffff"
border_color = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text_color = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a" if IS_DARK else "#52525b"
accent_color = "#2563eb"

st.markdown(f"""
<style>
    /* Hide default Streamlit chrome */
    header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    
    .block-container {{
        padding: 1.5rem 2rem 2rem !important;
        max-width: 1360px !important;
    }}
    
    /* Layout styling */
    [data-testid="stHorizontalBlock"] {{
        gap: 1.25rem !important;
    }}
    
    /* Custom components */
    .brand {{
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #3b82f6 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }}
    
    .brand-sub {{
        font-size: 0.8rem;
        color: {text_muted};
        margin-bottom: 1.5rem;
    }}
    
    .metric-card {{
        background: {card_color};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    .metric-label {{
        font-size: 0.75rem;
        color: {text_muted};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .metric-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {text_color};
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
    }}
    
    .chart-wrap {{
        background: {card_color};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    .chart-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {text_color};
    }}
    
    .chart-subtitle {{
        font-size: 0.75rem;
        color: {text_muted};
        margin-bottom: 1rem;
    }}
    
    /* Table custom styling */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.6rem 0.8rem;
        color: {text_muted};
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid {border_color};
        background: {bg_subtle};
    }}
    .data-table td {{
        padding: 0.65rem 0.8rem;
        color: {text_color};
        border-bottom: 1px solid {border_subtle};
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }}
    .badge-green {{
        color: #22c55e;
        background: rgba(34, 197, 94, 0.12);
    }}
    .badge-blue {{
        color: #3b82f6;
        background: rgba(59, 130, 246, 0.12);
    }}
    .badge-purple {{
        color: #a855f7;
        background: rgba(168, 85, 247, 0.12);
    }}
    
    /* Tabs custom styling */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: {text_muted} !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {text_color} !important;
        background: {card_color} !important;
        border-color: {border_color} !important;
    }}
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 6px !important;
        background: {bg_subtle} !important;
        border: 1px solid {border_color} !important;
        border-radius: 10px !important;
        padding: 4px;
        margin-bottom: 1.5rem;
    }}
</style>
""", unsafe_allow_html=True)

# Plotly styling template
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#a1a1aa" if IS_DARK else "#52525b", size=11),
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)" if IS_DARK else "rgba(0,0,0,0.05)",
        tickfont=dict(size=10),
    ),
)

# 4. Data Loading and Model Training Helpers
@st.cache_data
def load_wine_data():
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    feature_names = list(wine.feature_names)
    # Binary Classification setup mapping Wine Type 0 vs Others
    df['target'] = (wine.target > 0).astype(int)
    X = df[feature_names].values
    y = df['target'].values
    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return df, X_train, X_test, y_train, y_test, feature_names, ['Wine Type 0', 'Other Wine Types']

@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth == "None" else int(max_depth),
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Load data
df, X_train, X_test, y_train, y_test, feature_names, class_names = load_wine_data()

# 5. Sidebar Controls
with st.sidebar:
    st.markdown('<div class="brand">🔬 XAI Lab Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">SHAP vs. LIME Wine Classification</div>', unsafe_allow_html=True)
    
    st.subheader("Model Hyperparameters")
    n_estimators = st.slider("Number of Estimators", min_value=10, max_value=300, value=100, step=10)
    max_depth_option = st.selectbox("Max Depth", options=["None", 3, 5, 8, 10, 15])
    
    st.subheader("App Customization")
    theme_btn_label = "☀️ Switch to Light Mode" if IS_DARK else "🌙 Switch to Dark Mode"
    st.button(theme_btn_label, on_click=toggle_theme, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    **Explainable AI Course Lab**
    * Sri Krishna College of Technology
    * Department of CSE (AI & ML)
    """)

# Train model
model = train_model(X_train, y_train, n_estimators, max_depth_option)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# 6. Main Dashboard Layout
# Brand header
head_left, head_right = st.columns([7, 3])
with head_left:
    st.markdown('<div class="brand">Wine Classification Explainable AI Dashboard</div>', unsafe_allow_html=True)
    st.write("Real-time interpretability comparing local surrogates (LIME) and Shapley cooperative game theory (SHAP).")

# Row of KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">{acc*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Weighted F1-Score</div>
        <div class="metric-value">{f1:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Training Size</div>
        <div class="metric-value">{len(X_train)} samples</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Testing Size</div>
        <div class="metric-value">{len(X_test)} samples</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs Navigation
tab_model, tab_global, tab_local, tab_case = st.tabs([
    "📊 Dataset & Model Explorer",
    "🌎 Global Interpretability (SHAP)",
    "🎯 Local Interpretability (SHAP vs. LIME)",
    "🔬 Borderline Case Study"
])

# ---- TAB 1: DATA & MODEL EXPLORER ----
with tab_model:
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Wine Recognition Dataset (sklearn)</div>
        <div class="chart-subtitle">Sample view showing chemical characteristics of wine samples and their binarized target.</div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Confusion Matrix</div>
            <div class="chart-subtitle">Predicted vs. Actual Target (0: Wine Type 0, 1: Other Wine Types).</div>
        """, unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=class_names,
            y=class_names,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(PLOT_LAYOUT)
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_right:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Classification Performance Metrics</div>
            <div class="chart-subtitle">Detailed classification metrics by class.</div>
        """, unsafe_allow_html=True)
        
        rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        rows_html = ""
        for name in class_names:
            rows_html += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{rep[name]['precision']:.2f}</td>
                <td>{rep[name]['recall']:.2f}</td>
                <td>{rep[name]['f1-score']:.2f}</td>
                <td>{rep[name]['support']}</td>
            </tr>
            """
        
        st.markdown(f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---- TAB 2: GLOBAL INTERPRETABILITY ----
with tab_global:
    st.markdown("### SHAP Global Analysis")
    
    # Compute SHAP Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle SHAP output dimension differences across versions
    if isinstance(shap_values, list):
        # TreeExplainer returns a list [class_0_shap, class_1_shap]
        shap_values_class1 = shap_values[1]
    elif shap_values.ndim == 3:
        # Array of shape (samples, features, classes)
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">Global Feature Importance (SHAP)</div>
            <div class="chart-subtitle">Mean absolute SHAP value of each feature for Class 1 (Other Wine Types).</div>
        """, unsafe_allow_html=True)
        
        # Calculate feature importance
        mean_shap = np.abs(shap_values_class1).mean(axis=0)
        idx_sorted = np.argsort(mean_shap)
        
        fig_imp = go.Figure(go.Bar(
            x=mean_shap[idx_sorted],
            y=[feature_names[i] for i in idx_sorted],
            orientation='h',
            marker_color='#3b82f6'
        ))
        fig_imp.update_layout(PLOT_LAYOUT)
        fig_imp.update_layout(margin=dict(l=150, r=20, t=10, b=40))
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_g2:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">SHAP Summary Plot (Beeswarm)</div>
            <div class="chart-subtitle">Shows how high/low values of features impact the model prediction direction.</div>
        """, unsafe_allow_html=True)
        
        # Plot Beeswarm using Matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('dark_background' if IS_DARK else 'default')
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        shap.summary_plot(
            shap_values_class1,
            X_test,
            feature_names=feature_names,
            plot_type="bar" if False else None,
            show=False
        )
        # Style labels
        ax.tick_params(colors="#a1a1aa" if IS_DARK else "#52525b")
        ax.xaxis.label.set_color("#a1a1aa" if IS_DARK else "#52525b")
        ax.yaxis.label.set_color("#a1a1aa" if IS_DARK else "#52525b")
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
        st.markdown("</div>", unsafe_allow_html=True)

    # SHAP Dependence Section
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">SHAP Feature Dependence Plot</div>
        <div class="chart-subtitle">Select a feature to see how its value affects the SHAP value (contribution to prediction) and how it interacts with other features.</div>
    """, unsafe_allow_html=True)
    
    dep_feat = st.selectbox("Select Feature for Dependence Plot", options=feature_names, index=0)
    fig_dep, ax_dep = plt.subplots(figsize=(8, 4))
    fig_dep.patch.set_facecolor('none')
    ax_dep.set_facecolor('none')
    
    # Find best interaction feature automatically or manually
    shap.dependence_plot(
        dep_feat,
        shap_values_class1,
        X_test,
        feature_names=feature_names,
        show=False,
        ax=ax_dep
    )
    ax_dep.tick_params(colors="#a1a1aa" if IS_DARK else "#52525b")
    ax_dep.xaxis.label.set_color("#a1a1aa" if IS_DARK else "#52525b")
    ax_dep.yaxis.label.set_color("#a1a1aa" if IS_DARK else "#52525b")
    # Colorbar label styling if exists
    for child in fig_dep.get_children():
        if isinstance(child, plt.Axes) and child != ax_dep:
            child.tick_params(colors="#a1a1aa" if IS_DARK else "#52525b")
            child.yaxis.label.set_color("#a1a1aa" if IS_DARK else "#52525b")
            
    plt.tight_layout()
    st.pyplot(fig_dep)
    plt.clf()
    st.markdown("</div>", unsafe_allow_html=True)

# ---- TAB 3: LOCAL INTERPRETABILITY ----
with tab_local:
    st.markdown("### Compare Local Explanations on a Test Instance")
    
    # Instance selector
    inst_idx = st.number_input("Select Test Instance Index", min_value=0, max_value=len(X_test)-1, value=0, step=1)
    
    # True vs Predicted
    selected_sample = X_test[inst_idx]
    true_label = y_test[inst_idx]
    predicted_probs = model.predict_proba(selected_sample.reshape(1, -1))[0]
    pred_label = np.argmax(predicted_probs)
    
    c_info1, c_info2, c_info3 = st.columns(3)
    with c_info1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">True Label</div>
            <div class="metric-value">
                <span class="badge badge-blue">{class_names[true_label]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_info2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Prediction</div>
            <div class="metric-value">
                <span class="badge badge-purple">{class_names[pred_label]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_info3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Probability of Other Wine Types (Class 1)</div>
            <div class="metric-value">{predicted_probs[1]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Let's show feature values for this instance
    st.markdown("""
    <div class="chart-wrap">
        <div class="chart-title">Feature Values for Selected Sample</div>
        <div class="chart-subtitle">Review the raw feature values of the chosen test instance.</div>
    """, unsafe_allow_html=True)
    feat_val_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": selected_sample
    })
    st.dataframe(feat_val_df.T, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_l1, col_l2 = st.columns(2)
    
    with col_l1:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">SHAP Local Explanation (Waterfall Plot)</div>
            <div class="chart-subtitle">Waterfall showing how feature contributions push predicted value from baseline.</div>
        """, unsafe_allow_html=True)
        
        # Make waterfall plot
        fig_wf, ax_wf = plt.subplots(figsize=(6, 4))
        fig_wf.patch.set_facecolor('none')
        ax_wf.set_facecolor('none')
        
        # Construct shap Explanation object for the single sample
        exp_wf = shap.Explanation(
            values=shap_values_class1[inst_idx],
            base_values=explainer.expected_value[1],
            data=selected_sample,
            feature_names=feature_names
        )
        shap.plots.waterfall(exp_wf, show=False)
        ax_wf.tick_params(colors="#a1a1aa" if IS_DARK else "#52525b")
        plt.tight_layout()
        st.pyplot(fig_wf)
        plt.clf()
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_l2:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">LIME Local Explanation (Surrogate Weights)</div>
            <div class="chart-subtitle">LIME explanation showing weights of the local linear model (top features).</div>
        """, unsafe_allow_html=True)
        
        # Lime explainer creation (using cached X_train)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )
        
        # Generate lime explanation for this instance
        lime_exp = lime_explainer.explain_instance(
            data_row=selected_sample,
            predict_fn=model.predict_proba,
            num_features=10,
            labels=[1]
        )
        
        # Parse Lime explanation into Plotly to match layout style
        lime_list = lime_exp.as_list(label=1)
        lime_feats = [x[0] for x in lime_list]
        lime_weights = [x[1] for x in lime_list]
        
        # Color based on sign
        colors = ['#22c55e' if w > 0 else '#ef4444' for w in lime_weights]
        
        fig_lime = go.Figure(go.Bar(
            x=lime_weights,
            y=lime_feats,
            orientation='h',
            marker_color=colors
        ))
        fig_lime.update_layout(PLOT_LAYOUT)
        fig_lime.update_layout(yaxis=dict(autorange="reversed")) # Best features at the top
        st.plotly_chart(fig_lime, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

# ---- TAB 4: BORDERLINE CASE STUDY ----
with tab_case:
    st.markdown("### Case Study: Explaining Predictions with High Uncertainty")
    st.write("A model-agnostic exploration of instances where the Random Forest classifier is highly uncertain (predicted probabilities close to 0.5).")
    
    # Calculate uncertainty as distance to 0.5 probability
    probs_all = model.predict_proba(X_test)
    prob_dist_to_half = np.abs(probs_all[:, 1] - 0.5)
    
    # Get index of most uncertain instance
    hardest_idx = np.argmin(prob_dist_to_half)
    uncertain_prob = probs_all[hardest_idx, 1]
    
    st.markdown(f"""
    <div class="chart-wrap" style="border: 1px solid var(--amber) !important;">
        <div class="chart-title" style="color: #f59e0b;">🔍 Identified Borderline Case in Test Set</div>
        <div class="chart-subtitle">Test Instance Index: <strong>{hardest_idx}</strong> | Predicted Probability of Class 1: <strong>{uncertain_prob*100:.1f}%</strong></div>
        <p>This sample lies right on the boundary between classes. We inspect how SHAP and LIME dissect this decision.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c_case1, c_case2 = st.columns(2)
    
    with c_case1:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">SHAP Analysis for Borderline Instance</div>
            <div class="chart-subtitle">Attribution of conflicting features pulling prediction up or down.</div>
        """, unsafe_allow_html=True)
        
        fig_case_wf, ax_case_wf = plt.subplots(figsize=(6, 4))
        fig_case_wf.patch.set_facecolor('none')
        ax_case_wf.set_facecolor('none')
        
        exp_case_wf = shap.Explanation(
            values=shap_values_class1[hardest_idx],
            base_values=explainer.expected_value[1],
            data=X_test[hardest_idx],
            feature_names=feature_names
        )
        shap.plots.waterfall(exp_case_wf, show=False)
        ax_case_wf.tick_params(colors="#a1a1aa" if IS_DARK else "#52525b")
        plt.tight_layout()
        st.pyplot(fig_case_wf)
        plt.clf()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_case2:
        st.markdown("""
        <div class="chart-wrap">
            <div class="chart-title">LIME Analysis for Borderline Instance</div>
            <div class="chart-subtitle">Surrogate model feature contributions explaining the borderline prediction.</div>
        """, unsafe_allow_html=True)
        
        # LIME for borderline case
        lime_explainer_case = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42
        )
        
        lime_exp_case = lime_explainer_case.explain_instance(
            data_row=X_test[hardest_idx],
            predict_fn=model.predict_proba,
            num_features=10,
            labels=[1]
        )
        
        # Build Plotly bar chart
        lime_list_case = lime_exp_case.as_list(label=1)
        lime_feats_case = [x[0] for x in lime_list_case]
        lime_weights_case = [x[1] for x in lime_list_case]
        colors_case = ['#22c55e' if w > 0 else '#ef4444' for w in lime_weights_case]
        
        fig_lime_case = go.Figure(go.Bar(
            x=lime_weights_case,
            y=lime_feats_case,
            orientation='h',
            marker_color=colors_case
        ))
        fig_lime_case.update_layout(PLOT_LAYOUT)
        fig_lime_case.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_lime_case, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown("</div>", unsafe_allow_html=True)
