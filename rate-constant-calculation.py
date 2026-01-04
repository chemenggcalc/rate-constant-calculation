import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import linregress

# --- 1. PAGE CONFIGURATION ---
# 'initial_sidebar_state="expanded"' is CRITICAL for Mobile usability
st.set_page_config(
    page_title="Rate Constant Calculation",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- 2. GHOST MODE CSS (Hides branding, Fixes Mobile) ---
st.markdown("""
    <style>
    /* HIDE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;} /* Hide 3-dot menu */
    footer {visibility: hidden;}    /* Hide 'Made with Streamlit' */
    [data-testid="stToolbar"] {visibility: hidden;} /* Hide Top Toolbar */
    [data-testid="stDecoration"] {visibility: hidden;} /* Hide Top Decoration Bar */
    
    /* MOBILE OPTIMIZATION */
    /* Reduce padding on mobile so it doesn't look squashed */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Force Sidebar to be viewable on mobile (Streamlit collapses it by default) */
    /* This CSS ensures the arrow to open it is always high-contrast and visible */
    [data-testid="stSidebarCollapsedControl"] {
        display: block;
        color: #0068c9;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚗️ Rate Constant Calculation")
st.subheader("For Zeroth, First, and Second Order")
st.markdown("---")

# --- 3. SIDEBAR: DATA INPUT ---
with st.sidebar:
    st.header("📝 Input Data")
    st.info("Paste **Time (t)** vs **Conc [A]**")
    
    default_data = """0, 1.000
10, 0.819
20, 0.670
30, 0.549
40, 0.449
50, 0.368
60, 0.301"""
    
    raw_text = st.text_area("Data (CSV)", value=default_data, height=200)
    delimiter = st.selectbox("Separator", [", (Comma)", "\\t (Tab)", " (Space)"])
    
    # Parsing Logic
    valid_data = False
    try:
        sep_char = "," if "Comma" in delimiter else "\t" if "Tab" in delimiter else " "
        data_rows = [row.strip().split(sep_char) for row in raw_text.strip().split('\n') if row.strip()]
        df = pd.DataFrame(data_rows, columns=["Time", "Conc"])
        df["Time"] = pd.to_numeric(df["Time"])
        df["Conc"] = pd.to_numeric(df["Conc"])
        df = df.dropna().sort_values(by="Time")
        valid_data = True
    except:
        st.error("Format Error. Use 'Time, Conc'")

# --- 4. CALCULATION ENGINE ---
if valid_data:
    # Calculate Regressions
    x0, y0 = df["Time"], df["Conc"]
    reg0 = linregress(x0, y0)
    
    df_log = df[df["Conc"] > 0].copy()
    x1, y1 = df_log["Time"], np.log(df_log["Conc"])
    reg1 = linregress(x1, y1)
    
    df_inv = df[df["Conc"] != 0].copy()
    x2, y2 = df_inv["Time"], 1.0 / df_inv["Conc"]
    reg2 = linregress(x2, y2)

    results = {
        "Zeroth": {"R2": reg0.rvalue**2, "k": -reg0.slope, "eqn": f"[A] = {reg0.intercept:.3f} - {abs(reg0.slope):.4f}t"},
        "First":  {"R2": reg1.rvalue**2, "k": -reg1.slope, "eqn": f"ln[A] = {reg1.intercept:.3f} - {abs(reg1.slope):.4f}t"},
        "Second": {"R2": reg2.rvalue**2, "k": reg2.slope,  "eqn": f"1/[A] = {reg2.intercept:.3f} + {abs(reg2.slope):.4f}t"}
    }

    best_order = max(results, key=lambda x: results[x]["R2"])
    best_data = results[best_order]

    # --- 5. DISPLAY RESULTS ---
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.success(f"### Best Fit: {best_order} Order")
        st.metric("Rate Constant (k)", f"{best_data['k']:.5f}")
        st.metric("R-Squared", f"{best_data['R2']:.5f}")
        st.info(f"**Eq:** {best_data['eqn']}")

    with col2:
        st.subheader("Linearization Plots")
        tab1, tab2, tab3 = st.tabs(["Zeroth", "First", "Second"])
        
        # Function to create clean graphs with NO TOOLBAR
        def create_plot(x, y, title, y_label, color, reg):
            fig = px.scatter(x=x, y=y, labels={'x': 'Time', 'y': y_label})
            fig.update_traces(marker=dict(size=8, color=color))
            
            # Add Line
            x_range = np.linspace(min(x), max(x), 100)
            y_pred = reg.slope * x_range + reg.intercept
            fig.add_scatter(x=x_range, y=y_pred, mode='lines', line=dict(color='red', dash='dash'), showlegend=False)
            
            # Clean Layout
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300)
            return fig

        # 'config' parameter hides the graph toolbar (Zoom/Pan buttons)
        config = {'displayModeBar': False}

        with tab1:
            st.plotly_chart(create_plot(x0, y0, "Zeroth", "[A]", "blue", reg0), width="stretch", config=config)
        with tab2:
            st.plotly_chart(create_plot(x1, y1, "First", "ln[A]", "green", reg1), width="stretch", config=config)
        with tab3:
            st.plotly_chart(create_plot(x2, y2, "Second", "1/[A]", "orange", reg2), width="stretch", config=config)

else:
    st.warning("👈 Please enter data in the sidebar.")
