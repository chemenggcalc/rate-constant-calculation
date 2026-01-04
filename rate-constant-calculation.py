import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 1. Page Configuration
st.set_page_config(page_title="Kinetic Order Solver", layout="wide")

# 2. Modern CSS Injection
modern_css = """
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    button[title="View fullscreen"] {visibility: hidden; display: none;}

    /* Modern Title Styling */
    .main-title {
        font-size: 28px !important;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .sub-text {
        font-size: 14px;
        color: #64748B;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Card Styling for Inputs and Results */
    div.stTextInput, div.stMetric, .stPlotlyChart {
        background-color: #ffffff;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Column Spacing */
    [data-testid="column"] {
        padding: 10px;
    }

    /* Success box styling */
    .stSuccess {
        background-color: #F0FDF4 !important;
        color: #166534 !important;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
    }

    /* Adjust block container padding for embedding */
    .block-container {
        padding-top: 1.3rem;
        max-width: 900px;
    }
    </style>
    """
st.markdown(modern_css, unsafe_allow_html=True)

def calculate_r2(x, y):
    if len(x) < 2: return 0.0
    try:
        coefficients = np.polyfit(x, y, 1)
        p = np.poly1d(coefficients)
        y_pred = p(x)
        return r2_score(y, y_pred)
    except:
        return 0.0

def main():
    # Modern Header
    st.markdown('<h1 class="main-title">Kinetic Order Solver</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Input experimental data to identify the best-fit reaction model.</p>', unsafe_allow_html=True)

    # --- Step 1: Data Input ---
    default_time = "0, 10, 20, 30, 40, 50"
    default_conc = "1.0, 0.707, 0.5, 0.354, 0.25, 0.177"

    col_a, col_b = st.columns(2)
    with col_a:
        time_str = st.text_input("Time points (t)", default_time)
    with col_b:
        conc_str = st.text_input("Concentration [A]", default_conc)

    try:
        t = np.array([float(i.strip()) for i in time_str.split(",") if i.strip()])
        a = np.array([float(i.strip()) for i in conc_str.split(",") if i.strip()])

        if len(t) != len(a):
            st.error("Point mismatch: Check counts for t and [A]")
            return

        # Kinetic Analysis
        r2_zero = calculate_r2(t, a)
        r2_first = calculate_r2(t, np.log(a))
        r2_second = calculate_r2(t, 1/a)

        # --- Step 2: Metrics ---
        st.markdown("#### Regression Analysis ($R^2$)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Zeroth Order", f"{r2_zero:.4f}")
        m2.metric("First Order", f"{r2_first:.4f}")
        m3.metric("Second Order", f"{r2_second:.4f}")

        # Best Fit Alert
        scores = {"Zeroth Order": r2_zero, "First Order": r2_first, "Second Order": r2_second}
        best_order_name = max(scores, key=scores.get)
        st.success(f"**Optimal Model Identified:** {best_order_name}")

        # --- Step 3: Visualization ---
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Color palette
        accent_color = "#3B82F6" # Modern Blue
        
        if best_order_name == "Zeroth Order":
            y_plot, ylabel = a, "Concentration [A]"
        elif best_order_name == "First Order":
            y_plot, ylabel = np.log(a), "ln([A])"
        else:
            y_plot, ylabel = 1/a, "1/[A]"

        m, b = np.polyfit(t, y_plot, 1)
        
        # Style the plot for a modern look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        ax.scatter(t, y_plot, color=accent_color, s=80, edgecolors='white', linewidth=1.5, label='Experimental', zorder=3)
        ax.plot(t, m*t + b, color='#94A3B8', linestyle='--', linewidth=1.5, label=f'Linear Fit (k={abs(m):.4f})', zorder=2)
        
        ax.set_xlabel("Time (s)", fontsize=10, fontweight='600')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='600')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis='y', linestyle='-', alpha=0.1)
        
        st.pyplot(fig)

    except Exception:
        st.info("Please enter valid comma-separated numeric data.")

if __name__ == "__main__":
    main()
