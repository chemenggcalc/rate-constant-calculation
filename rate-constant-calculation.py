import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import linregress

# --- 1. PAGE CONFIGURATION ---
# Fixed: Added 'initial_sidebar_state="expanded"' so input is always visible
st.set_page_config(
    page_title="Rate Constant Calculation",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
# Fixed: Removed "header {visibility: hidden;}" so you can always reopen the sidebar
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} /* Hides the 3-dot menu */
    footer {visibility: hidden;}    /* Hides the 'Made with Streamlit' footer */
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("⚗️ Rate Constant Calculation")
st.subheader("For Zeroth, First, and Second Order using Integrated Rate Equations")
st.markdown("---")

# --- 2. SIDEBAR: DATA INPUT ---
with st.sidebar:
    st.header("📝 Input Experimental Data")
    st.info("Paste your **Time (t)** vs **Concentration [A]** data below.")
    
    # Default Dummy Data (First Order Decay)
    default_data = """0, 1.000
10, 0.819
20, 0.670
30, 0.549
40, 0.449
50, 0.368
60, 0.301"""
    
    raw_text = st.text_area("Data (CSV Format)", value=default_data, height=250, help="Format: Time, Concentration")
    delimiter = st.selectbox("Column Separator", [", (Comma)", "\\t (Tab)", " (Space)"])
    
    st.markdown("---")
    st.caption("Built for ChemEnggCalc.com")

    # Parsing Logic
    valid_data = False
    try:
        sep_char = "," if "Comma" in delimiter else "\t" if "Tab" in delimiter else " "
        
        # Split text into rows and columns
        data_rows = [row.strip().split(sep_char) for row in raw_text.strip().split('\n') if row.strip()]
        df = pd.DataFrame(data_rows, columns=["Time", "Conc"])
        
        # Convert to float
        df["Time"] = pd.to_numeric(df["Time"])
        df["Conc"] = pd.to_numeric(df["Conc"])
        
        # Clean Data
        df = df.dropna().sort_values(by="Time")
        valid_data = True
        
    except Exception as e:
        st.error(f"⚠️ Data Error: {e}")
        st.warning("Please ensure data is formatted as 'Time, Concentration'.")

# --- 3. CALCULATION ENGINE ---
if valid_data:
    # A. Zero Order Model: Plot [A] vs t
    x0 = df["Time"]
    y0 = df["Conc"]
    reg0 = linregress(x0, y0)
    
    # B. First Order Model: Plot ln[A] vs t
    # Filter out concentration <= 0 to avoid log errors
    df_log = df[df["Conc"] > 0].copy()
    x1 = df_log["Time"]
    y1 = np.log(df_log["Conc"])
    reg1 = linregress(x1, y1)
    
    # C. Second Order Model: Plot 1/[A] vs t
    # Filter out concentration == 0 to avoid division errors
    df_inv = df[df["Conc"] != 0].copy()
    x2 = df_inv["Time"]
    y2 = 1.0 / df_inv["Conc"]
    reg2 = linregress(x2, y2)

    # D. Compile Results
    results = {
        "Zeroth Order": {
            "R2": reg0.rvalue**2,
            "k": -reg0.slope, # Slope is negative
            "eqn": f"[A] = {reg0.intercept:.4f} - {abs(reg0.slope):.4f} t",
            "linear_y": "Concentration [A]"
        },
        "First Order": {
            "R2": reg1.rvalue**2,
            "k": -reg1.slope, # Slope is negative
            "eqn": f"ln[A] = {reg1.intercept:.4f} - {abs(reg1.slope):.4f} t",
            "linear_y": "ln([A])"
        },
        "Second Order": {
            "R2": reg2.rvalue**2,
            "k": reg2.slope, # Slope is positive
            "eqn": f"1/[A] = {reg2.intercept:.4f} + {abs(reg2.slope):.4f} t",
            "linear_y": "1 / [A]"
        }
    }

    # Identify Best Fit (R^2 closest to 1.0)
    best_order = max(results, key=lambda x: results[x]["R2"])
    best_data = results[best_order]

    # --- 4. DISPLAY RESULTS ---
    
    # Top Section: The Verdict
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.success(f"### ✅ Best Fit: {best_order}")
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px;">
            <p style="font-size:18px; margin:0;"><b>Rate Constant (k):</b></p>
            <p style="font-size:32px; color:#0068c9; font-weight:bold; margin:0;">{best_data['k']:.5f}</p>
            <hr style="margin:10px 0;">
            <p style="margin:0;"><b>R² Value:</b> {best_data['R2']:.5f}</p>
            <p style="margin:0;"><b>Integrated Equation:</b><br><code>{best_data['eqn']}</code></p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("See Full Comparison Table"):
            summary_df = pd.DataFrame(results).T[["R2", "k", "eqn"]]
            st.table(summary_df.style.format({"R2": "{:.4f}", "k": "{:.5f}"}))

    with col2:
        # Visualisation Tabs
        st.subheader("📈 Linearization Plots")
        tab1, tab2, tab3 = st.tabs(["Zeroth Order", "First Order", "Second Order"])
        
        def create_plot(x, y, title, y_label, color, fit_reg):
            # Scatter Plot of Data
            fig = px.scatter(x=x, y=y, labels={'x': 'Time (t)', 'y': y_label})
            fig.update_traces(marker=dict(size=10, color=color, line=dict(width=1, color='DarkSlateGrey')))
            
            # Add Regression Line
            x_range = np.linspace(min(x), max(x), 100)
            y_pred = fit_reg.slope * x_range + fit_reg.intercept
            
            fig.add_scatter(x=x_range, y=y_pred, mode='lines', name='Linear Fit', line=dict(color='red', dash='dash'))
            
            fig.update_layout(
                title=title,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            return fig

        with tab1:
            fig0 = create_plot(x0, y0, f"Zeroth Order Fit (R²={results['Zeroth Order']['R2']:.4f})", "[A]", "#636EFA", reg0)
            st.plotly_chart(fig0, use_container_width=True)
            
        with tab2:
            fig1 = create_plot(x1, y1, f"First Order Fit (R²={results['First Order']['R2']:.4f})", "ln[A]", "#EF553B", reg1)
            st.plotly_chart(fig1, use_container_width=True)
            
        with tab3:
            fig2 = create_plot(x2, y2, f"Second Order Fit (R²={results['Second Order']['R2']:.4f})", "1/[A]", "#00CC96", reg2)
            st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Please enter your experimental data in the sidebar to begin.")
