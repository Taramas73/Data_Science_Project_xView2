import streamlit as st

# -------------------- TITLE --------------------
st.title("🌍 Project Overview")
st.markdown("### Automated Building Damage Assessment using Satellite Imagery")
st.markdown("---")

# -------------------- 1. CONTEXT --------------------
st.header("1. The xView2 Challenge overview")

st.markdown("""
The project is inspired by the **xView2 Challenge**, a large-scale computer vision competition focused on disaster response.

### 🎯 Goal
- Automatically **detect buildings**
- **Assess damage severity** from satellite image pairs (before & after disaster)

link : https://xview2.org/
            
### 🚑 Why It Matters
Rapid damage assessment is critical after natural disasters such as:
- Earthquakes  
- Floods  
- Hurricanes  
- Wildfires  

Accurate and automated analysis helps emergency teams:
- Prioritize interventions  
- Allocate resources efficiently  
- Reduce response time  
""")

# -------------------- 2. DATASET --------------------
st.header("2. The Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Total Images", "18,336")
col2.metric("Annotated Buildings", "850,000+")
col3.metric("Countries Covered", "15")

st.markdown("""
The dataset contains **bi-temporal satellite imagery**:
- 🟢 **Pre-disaster images** (reference state)
- 🔴 **Post-disaster images** (after event)
- 🏢 **Building masks with damage labels**

Each building is classified into one of four categories:
- No Damage  
- Minor Damage  
- Major Damage  
- Destroyed  
""")

col1.image("app/imgs/pre-intro.jpg", caption="Pre-disaster image")
col2.image("app/imgs/post-intro.jpg", caption="Post-disaster image")
col3.image("app/imgs/mask-intro.jpg", caption="Damage segmentation mask")

# -------------------- 3. APPROACH --------------------
st.header("3. Modeling Approach")

st.markdown("""
We frame the problem as a **semantic segmentation task**.

### 🧠 Why Segmentation?
Instead of predicting a single label per image, the model:
- Produces a **pixel-wise classification map**
- Identifies building footprints
- Assigns a damage severity class to each building region

### 🔍 Pipeline Overview
1. Input: Pre-disaster image  
2. Input: Post-disaster image  
3. Deep learning model (U-Net architecture)  
4. Output: Damage segmentation mask  

This approach allows fine-grained spatial understanding of disaster impact and enables more precise decision-making.
""")

st.markdown("---")
st.info("👉 Navigate to the Model Development page to explore architecture, training strategy, and performance metrics.")