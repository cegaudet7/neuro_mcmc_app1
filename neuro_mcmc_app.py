# File: neuro_mcmc_app.py

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Neuropsych Low Score Simulator", layout="wide")
st.title("Neuropsych Test Low Score Simulator (MCMC)")

# Step 1: Upload correlation matrix
st.sidebar.header("Step 1: Upload Correlation Matrix")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV correlation matrix", type=["xlsx","csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        corr_df = pd.read_excel(uploaded_file, index_col=0)
    else:
        corr_df = pd.read_csv(uploaded_file, index_col=0)
    
    # Verify symmetry
    if not np.allclose(corr_df.values, corr_df.values.T):
        st.warning("Warning: Your correlation matrix is not symmetric!")

    variables = list(corr_df.columns)
    
    # Step 2: Select subtests/indices
    st.sidebar.header("Step 2: Select Subtests/Indices")
    selected_vars = st.sidebar.multiselect("Select tests to simulate", variables, default=variables)

    if len(selected_vars) < 1:
        st.warning("Select at least one subtest/index to simulate.")
    else:
        corr_sub = corr_df.loc[selected_vars, selected_vars].values

        # Step 3: Set cutoff
        st.sidebar.header("Step 3: Set cutoff")
        cutoff = st.sidebar.number_input("Enter cutoff in SD units (e.g., -1 for <1 SD below mean)", value=-1.0, step=0.1)

        # Step 4: Set simulation parameters
        st.sidebar.header("Step 4: Simulation settings")
        n_sim = st.sidebar.number_input("Number of simulated participants", value=10000, step=1000)
        random_seed = st.sidebar.number_input("Random seed", value=42, step=1)

        # Step 5: Run simulation
        if st.sidebar.button("Run Simulation"):
            np.random.seed(random_seed)
            # Generate multivariate normal scores
corr_sub_pd = nearest_pd(corr_sub)

simulated_scores = np.random.multivariate_normal(
    mean=np.zeros(len(selected_vars)),
    cov=corr_sub_pd,
    size=n_sim
)
            
# Count low scores per participant
low_scores_counts = (simulated_scores < cutoff).sum(axis=1)

            st.subheader("Simulation Results")
            st.write(f"Number of simulated participants: {n_sim}")
            st.write(f"Cutoff: {cutoff} SD")
            
            # Expected number of low scores per person
            expected_low = low_scores_counts.mean()
            st.write(f"Expected number of low scores per person: {expected_low:.2f}")
            
            # Distribution of low scores
            st.write("Distribution of low scores across simulated participants:")
            st.bar_chart(pd.Series(low_scores_counts).value_counts().sort_index())

            # Optional: display a table
            st.write("Example simulated scores (first 10 participants):")
            st.dataframe(pd.DataFrame(simulated_scores, columns=selected_vars).head(10))

def nearest_pd(A):
    """Find the nearest positive-definite matrix to A"""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_pd(B):
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False