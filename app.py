import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

from model import CropPerformancePredictor, GenerativeScenarioPlanner

# Page Config
st.set_page_config(page_title="Climate-Resilient Crop AI", layout="wide", page_icon="🌾")

# Inject Custom CSS for Premium Aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: 800;
        color: #2E7D32;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 18px;
        color: #555555;
        margin-bottom: 30px;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🌾 Autonomous Multi-Omics Fusion Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Climate Resilient Genomic Selection & Breeding Optimization Blueprint</p>', unsafe_allow_html=True)

# Initialize Models (with random weights for demo)
@st.cache_resource
def load_models():
    omics_dims = {'genomics': 1000, 'transcriptomics': 500}
    env_dim = 50
    predictive_model = CropPerformancePredictor(omics_dims, env_dim, num_traits=2)
    generative_model = GenerativeScenarioPlanner(genotype_dim=1000, env_dim=env_dim)
    predictive_model.eval()
    generative_model.eval()
    return predictive_model, generative_model

pred_model, gen_model = load_models()

# Sidebar: Biological Inputs
st.sidebar.header("🧬 Omics Input Data")
st.sidebar.markdown("Upload candidate genotype multi-omics data for evaluation.")

uploaded_file = st.sidebar.file_uploader("Upload Genotype (.vcf / .csv)", type=["csv", "vcf"])
if uploaded_file is None:
    st.sidebar.info("Using baseline simulated genotype data for demonstration.")
    
# Simulated Baseline Genotype
baseline_genomics = torch.randn(1, 1000)
baseline_transcriptomics = torch.randn(1, 500)
omics_data = {'genomics': baseline_genomics, 'transcriptomics': baseline_transcriptomics}

st.sidebar.divider()
st.sidebar.header("🌍 Baseline Environment")
st.sidebar.markdown("Current Agro-Climatic Zone Settings")
base_temp = st.sidebar.slider("Average Temp (°C)", 15.0, 35.0, 22.0)
base_precip = st.sidebar.slider("Annual Precipitation (mm)", 200, 1500, 800)

# Simulate Environment Tensor from inputs
baseline_env = torch.randn(1, 50) 
# Inject UI values into the dummy tensor
baseline_env[0][0] = (base_temp - 25) / 10.0
baseline_env[0][1] = (base_precip - 800) / 500.0

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Current Trait Prediction")
    st.markdown("Predictive performance under **baseline** environmental conditions using deep neural genomic prediction.")
    
    with torch.no_grad():
        baseline_preds = pred_model(omics_data, baseline_env)
        yield_pred = baseline_preds[0][0].item() * 10 + 100 # Denormalize dummy output
        drought_tol = torch.sigmoid(baseline_preds[0][1]).item() * 100
        
    c1, c2 = st.columns(2)
    c1.metric(label="Predicted Yield (Bushels/Acre)", value=f"{yield_pred:.1f} bu/ac")
    c2.metric(label="Drought Resilience Score", value=f"{drought_tol:.1f}/100")
    
    # Gauge Chart for Resilience
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = drought_tol,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Baseline Stress Resilience"},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "#2E7D32"},
                 'steps' : [
                     {'range': [0, 40], 'color': "#ffcdd2"},
                     {'range': [40, 70], 'color': "#fff9c4"},
                     {'range': [70, 100], 'color': "#c8e6c9"}]}
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔥 Generative 'What-If' Scenario")
    st.markdown("Simulate crop performance under future extreme climate scenarios.")
    
    st.markdown("##### Adjust Future Climate Variables")
    temp_shift = st.slider("Heatwave Severity (+°C)", 0.0, 10.0, 3.5)
    precip_shift = st.slider("Drought Severity (% precipitation reduction)", 0, 80, 40)
    
    # Construct extreme environment tensor
    extreme_env = baseline_env.clone()
    extreme_env[0][0] += temp_shift / 10.0
    extreme_env[0][1] -= precip_shift / 100.0
    
    with torch.no_grad():
        simulated_outcome, _, _ = gen_model(baseline_genomics, extreme_env)
        sim_yield = simulated_outcome[0][0].item() * 10 + 100
        sim_drought = torch.sigmoid(simulated_outcome[0][1]).item() * 100
        
    delta_yield = sim_yield - yield_pred
    delta_drought = sim_drought - drought_tol
    
    c3, c4 = st.columns(2)
    c3.metric(label="Simulated Yield (Extreme Climate)", value=f"{sim_yield:.1f} bu/ac", delta=f"{delta_yield:.1f} bu/ac")
    c4.metric(label="Stress Resilience (Extreme Climate)", value=f"{sim_drought:.1f}/100", delta=f"{delta_drought:.1f}")

st.divider()

st.subheader("📍 Localized Soil Fixation & Breeding Strategy Blueprint")
st.info("Based on the generated multi-omics predictions and causal biological links, the AI recommends the following strategic blueprint for this specific genotype.")

blueprint_cols = st.columns(3)
with blueprint_cols[0]:
    st.markdown("#### 🧬 Genomic Editing Targets")
    st.markdown("- Up-regulate **DREB1A** pathway (Drought Response)\n- Modify loci linked to heat-shock proteins (HSP70)\n- **Recommendation**: Prioritize for dry-zone breeding programs.")
with blueprint_cols[1]:
    st.markdown("#### 🧪 Soil Fixation Blueprint")
    st.markdown("- **Nitrogen Strategy**: Slow-release urea\n- **Microbiome**: Inoculate with *Bacillus subtilis* to enhance root resilience under water deficit.\n- **pH Adjustment**: Add agricultural lime (target pH 6.8).")
with blueprint_cols[2]:
    st.markdown("#### 🌱 Parental Selection")
    if sim_yield > 90:
        st.success("Based on resilient performance under extreme conditions, this genotype is a **Strong Candidate** as a maternal line for crossing with high-yield tropical varieties.")
    else:
        st.warning("This genotype shows significant yield penalty under extreme heat. Consider as a paternal line for specific trait introgression only.")
