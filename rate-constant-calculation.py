import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 1. Page Config & CSS to hide Streamlit branding
st.set_page_config(page_title="Reaction Order Predictor", layout="wide")

hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_style, unsafe_base64=True)

def calculate_r2(x, y):
    """Calculates R-squared value."""
    if len(x) < 2: return 0
    # Linear regression: y = mx + c
    coefficients = np.polyfit(x, y, 1)
    p = np.poly1d(coefficients)
    y_pred = p(x)
    return r2_score(y, y_pred)

def main():
    st.title("🧪 Kinetic Best-Fit Calculator")
    st.write("Input your experimental data below. The app will automatically determine the reaction order.")

    # --- Data Input Section ---
    st.subheader("Step 1: Enter Experimental Data")
    
    # Prefilled data (Example: A typical First Order decay)
    default_time = "0, 10, 20, 30, 40, 50"
    default_conc = "1.0, 0.707, 0.5, 0.354, 0.25, 0.177"

    col1, col2 = st.columns(2)
    with col1:
        time_str = st.text_input("Time points (comma separated)", default_time)
    with col2:
        conc_str = st.text_input("Concentrations [A] (comma separated)", default_conc)

    try:
        # Convert strings to numpy arrays
        t = np.array([float(i.strip()) for i in time_str.split(",")])
        a = np.array([float(i.strip()) for i in conc_str.split(",")])

        if len(t) != len(a):
            st.error("Error: Time and Concentration arrays must be the same length.")
            return

        # --- Kinetic Transformations ---
        # Zeroth: [A] vs t
        r2_zero = calculate_r2(t, a)
        
        # First: ln[A] vs t
        r2_first = calculate_r2(t, np.log(a))
        
        # Second: 1/[A] vs t
        r2_second = calculate_r2(t, 1/a)

        # --- Determine Best Fit ---
        results = {
            "Zeroth Order ([A] vs t)": r2_zero,
            "First Order (ln[A] vs t)": r2_first,
            "Second Order (1/[A] vs t)": r2_second
        }
        best_order = max(results, key=results.get)

        # --- Results Display ---
        st.markdown("---")
        st.subheader("Step 2: Statistical Analysis ($R^2$)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Zeroth Order $R^2$", f"{r2_zero:.4f}")
        c2.metric("First Order $R^2$", f"{r2_first:.4f}")
        c3.metric("Second Order $R^2$", f"{r2_second:.4f}")

        st.success(f"**Conclusion:** The data best fits a **{best_order}** kinetics model.")

        # --- Visualizing the Best Fit ---
        st.markdown("---")
        st.subheader("Step 3: Best Fit Visualization")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        if best_order.startswith("Zeroth"):
            y_plot = a
            label_y = "Concentration [A]"
        elif best_order.startswith("First"):
            y_plot = np.log(a)
            label_y = "ln([A])"
        else:
            y_plot = 1/a
            label_y = "1/[A]"

        # Plot points and regression line
        m, b = np.polyfit(t, y_plot, 1)
        ax.scatter(t, y_plot, color='red', label='Experimental Data')
        ax.plot(t, m*t + b, color='blue', linestyle='--', label=f'Linear Fit (k={abs(m):.4f})')
        
        ax.set_xlabel("Time")
        ax.set_ylabel(label_y)
        ax.legend()
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)

    except Exception as e:
        st.warning("Please ensure your inputs are numbers separated by commas.")

if __name__ == "__main__":
    main()
