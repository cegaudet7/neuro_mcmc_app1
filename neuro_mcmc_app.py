import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------
# Utilities for PD correction
# -------------------------------

def is_pd(B):
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_pd(A):
    """Nearest positive-definite matrix"""
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


# -------------------------------
# Streamlit App
# -------------------------------

st.set_page_config(page_title="Neuropsych MCMC Simulator", layout="wide")
st.title("Neuropsychological MCMC Low-Score Simulator")

st.markdown(
    """
    **Assumptions**
    - All variables are standardized (mean = 0, SD = 1)
    - Input file is a **square correlation matrix**
    - Rows and columns must match exactly
    """
)

# -------------------------------
# Upload correlation matrix
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload correlation matrix (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:

    # Read matrix
    corr_df = pd.read_excel(uploaded_file, index_col=0)

    # Force numeric
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")

    # Drop empty rows/cols
    corr_df = corr_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Basic checks
    if corr_df.shape[0] != corr_df.shape[1]:
        st.error("Correlation matrix must be square.")
        st.stop()

    if not (corr_df.index.equals(corr_df.columns)):
        st.error("Row and column labels must match exactly.")
        st.stop()

    if corr_df.isna().any().any():
        st.error("Matrix contains non-numeric or empty cells.")
        st.stop()

    st.success("Correlation matrix loaded successfully.")

    # -------------------------------
    # Variable selection
    # -------------------------------

    variables = corr_df.index.tolist()

    selected_vars = st.multiselect(
        "Select indices/subtests to include",
        options=variables,
        default=variables
    )

    if len(selected_vars) < 2:
        st.warning("Select at least two variables.")
        st.stop()

    # -------------------------------
    # Simulation parameters
    # -------------------------------

    col1, col2 = st.columns(2)

    with col1:
        cutoff = st.number_input(
            "Low-score cutoff (SD units)",
            value=-1.0,
            step=0.1
        )

    with col2:
        n_sim = st.number_input(
            "Number of simulated profiles",
            min_value=1000,
            max_value=500000,
            value=100000,
            step=1000
        )

    # -------------------------------
    # Run simulation
    # -------------------------------

    if st.button("Run Simulation"):

        # Subset correlation matrix
        corr_sub = corr_df.loc[selected_vars, selected_vars].values

        # Enforce positive definiteness
        corr_sub_pd = nearest_pd(corr_sub)

        # Simulate multivariate normal
        simulated_scores = np.random.multivariate_normal(
            mean=np.zeros(len(selected_vars)),
            cov=corr_sub_pd,
            size=n_sim
        )

        # Count low scores per profile
        low_scores_counts = (simulated_scores < cutoff).sum(axis=1)

        # Results
        st.subheader("Results")

        mean_low = low_scores_counts.mean()
        st.write(f"**Expected number of low scores:** {mean_low:.2f}")

        # Distribution table
        dist = (
            pd.Series(low_scores_counts)
            .value_counts()
            .sort_index()
            .rename("Count")
            .to_frame()
        )

        dist["Proportion"] = dist["Count"] / n_sim

        st.dataframe(dist)

        st.success("Simulation completed successfully.")