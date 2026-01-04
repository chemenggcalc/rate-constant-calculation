import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Page Configuration
    st.set_page_config(page_title="Chemical Kinetics Calculator", page_icon="🧪")

    st.title("🧪 Rate Constant ($k$) Calculator")
    st.markdown("""
    This application calculates the rate constant **($k$)** using **Integrated Rate Equations**.
    Select your reaction order from the sidebar to begin.
    """)

    # --- Sidebar Configuration ---
    st.sidebar.header("Reaction Parameters")
    order = st.sidebar.selectbox(
        "Select Reaction Order",
        ("Zeroth Order", "First Order", "Second Order")
    )

    # --- Input Section ---
    st.subheader(f"Input Data for {order}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        a0 = st.number_input("Initial Conc. $[A]_0$ (M)", min_value=0.0001, value=1.0, format="%.4f")
    
    with col2:
        at = st.number_input("Final Conc. $[A]_t$ (M)", min_value=0.0000, value=0.5, format="%.4f")
        
    with col3:
        t = st.number_input("Time elapsed $t$ (seconds)", min_value=0.0001, value=10.0, format="%.2f")

    # --- Logic & Calculation ---
    st.markdown("---")
    st.subheader("Calculation & Results")

    # Initialize k
    k = 0.0
    valid_calculation = True

    # 1. Zeroth Order Logic
    if order == "Zeroth Order":
        st.info("For a Zeroth Order reaction, the rate is independent of concentration.")
        st.latex(r"k = \frac{[A]_0 - [A]_t}{t}")
        
        if st.button("Calculate k"):
            k = (a0 - at) / t
            units = "M/s"

    # 2. First Order Logic
    elif order == "First Order":
        st.info("For a First Order reaction, the rate depends linearly on concentration.")
        st.latex(r"k = \frac{1}{t} \ln\left(\frac{[A]_0}{[A]_t}\right)")
        
        if st.button("Calculate k"):
            if at <= 0:
                st.error("Final concentration must be > 0 for Logarithmic calculation.")
                valid_calculation = False
            else:
                k = (1/t) * np.log(a0 / at)
                units = "1/s"

    # 3. Second Order Logic
    elif order == "Second Order":
        st.info("For a Second Order reaction, the rate depends on the square of the concentration.")
        st.latex(r"k = \frac{1}{t} \left( \frac{1}{[A]_t} - \frac{1}{[A]_0} \right)")
        
        if st.button("Calculate k"):
            if at <= 0:
                st.error("Final concentration must be > 0 for inverse calculation.")
                valid_calculation = False
            else:
                k = (1/t) * ((1/at) - (1/a0))
                units = "1/(M·s)"

    # --- Display Results ---
    if valid_calculation and k != 0:
        st.success(f"The calculated Rate Constant is:")
        st.metric(label="Rate Constant (k)", value=f"{k:.5f} {units}")
        
        # --- Visualization (Optional but helpful) ---
        st.subheader("Concentration Decay Plot")
        
        # Generate time points for the graph (from 0 to 1.5x the input time)
        t_plot = np.linspace(0, t * 1.5, 100)
        
        # Calculate concentration over time based on the derived k
        y_plot = []
        
        if order == "Zeroth Order":
            # [A]t = [A]0 - kt
            y_plot = a0 - (k * t_plot)
            # Filter out negative concentrations for the graph
            y_plot = [y if y > 0 else 0 for y in y_plot] 
            
        elif order == "First Order":
            # [A]t = [A]0 * e^(-kt)
            y_plot = a0 * np.exp(-k * t_plot)
            
        elif order == "Second Order":
            # 1/[A]t = kt + 1/[A]0  ->  [A]t = 1 / (kt + 1/[A]0)
            y_plot = 1 / ((k * t_plot) + (1/a0))

        # Plotting
        fig, ax = plt.subplots()
        ax.plot(t_plot, y_plot, label=f'{order} Decay', color='teal', linewidth=2)
        ax.scatter([t], [at], color='red', zorder=5, label='Measured Point') # Show the user's input point
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration [A] (M)")
        ax.set_title(f"Reaction Progress ($k={k:.4f}$)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()
