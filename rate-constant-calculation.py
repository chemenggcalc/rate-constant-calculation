import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 1. Page Config & CSS to hide Streamlit branding/footer
st.set_page_config(page_title="Kinetic Order Solver", layout="wide")

# Corrected CSS injection
hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """
st.markdown(hide_style, unsafe_allow_html=True)

def calculate_r2(x, y):
    """Calculates R-squared value for a linear fit."""
    if len(x) < 2: return 0.0
    # Linear regression: y = mx + c
    try:
        coefficients = np.polyfit(x, y, 1)
        p = np.poly1d(coefficients)
        y_pred = p(x)
        return r2_score(y, y_pred)
    except:
        return 0.0

def main():
    st.title("🧪 Kinetic Best-Fit Calculator")
    st.write("Determine the reaction order and rate constant by comparing $R^2$ values across different models.")

    # --- Data Input Section (Responsive Columns) ---
    st.subheader("Step 1: Enter Experimental Data")
    
    # Prefilled data (Example: First Order decay data)
    default_time = "0, 10, 20, 30, 40, 50"
    default_conc = "1.0, 0.707, 0.5, 0.354, 0.25, 0.177"

    col_a, col_b = st.columns([1, 1])
    with col_a:
        time_str = st.text_input("Time points (t)", default_time, help="Separate values with commas")
    with col_b:
        conc_str = st.text_input("Concentration [A]", default_conc, help="Separate values with commas")

    try:
        # Data Cleanup
        t = np.array([float(i.strip()) for i in time_str.split(",") if i.strip()])
        a = np.array([float(i.strip()) for i in conc_str.split(",") if i.strip()])

        if len(t) != len(a):
            st.error("Error: The number of Time points and Concentration points must match.")
            return
        
        if len(t) < 2:
            st.warning("Please enter at least two data points.")
            return

        # --- Kinetic Transformations ---
        # Zeroth: [A]
        r2_zero = calculate_r2(t, a)
        
        # First: ln[A]
        # We use np.where to avoid log(0) errors
        ln_a = np.log(a)
        r2_first = calculate_r2(t, ln_a)
        
        # Second: 1/[A]
        inv_a = 1/a
        r2_second = calculate_r2(t, inv_a)

        # --- Statistics Table ---
        st.markdown("---")
        st.subheader("Step 2: Statistical Comparison ($R^2$)")
        
        # Determine the best fit
        scores = {
            "Zeroth Order": r2_zero,
            "First Order": r2_first,
            "Second Order": r2_second
        }
        best_order_name = max(scores, key=scores.get)

        c1, c2, c3 = st.columns(3)
        c1.metric("Zeroth Order $R^2$", f"{r2_zero:.4f}")
        c2.metric("First Order $R^2$", f"{r2_first:.4f}")
        c3.metric("Second Order $R^2$", f"{r2_second:.4f}")

        st.success(f"**Best Fit Result:** The data follows **{best_order_name}** kinetics.")

        # --- Visualization ---
        st.markdown("---")
        st.subheader(f"Step 3: Linearized Plot for {best_order_name}")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if best_order_name == "Zeroth Order":
            y_plot = a
            ylabel = "Concentration [A]"
        elif best_order_name == "First Order":
            y_plot = ln_a
            ylabel = "ln([A])"
        else:
            y_plot = inv_a
            ylabel = "1/[A]"

        # Regression line for plotting
        m, b = np.polyfit(t, y_plot, 1)
        line = m * t + b
        
        ax.scatter(t, y_plot, color='#e74c3c', label='Experimental Data', s=50)
        ax.plot(t, line, color='#2c3e50', linestyle='--', label=f'Best Fit Line (slope/k = {abs(m):.4f})')
        
        ax.set_xlabel("Time (t)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
        
        st.pyplot(fig)

    except ValueError:
        st.error("Invalid Input: Please ensure all values are numbers and separated by commas.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
