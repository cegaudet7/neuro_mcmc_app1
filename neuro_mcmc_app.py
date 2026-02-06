import streamlit as st
import numpy as np
import pandas as pd

# ======================================================
# Positive-definite utilities
# ======================================================

def is_pd(B):
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_pd(A):
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


# ======================================================
# Score definitions
# ======================================================

WAIS_INDEX_SCORES = ["VCI", "VSI", "FRI", "WMI", "PSI"]
WMS_INDEX_SCORES = ["AMI", "VMI", "IMI", "DMI"]
INDEX_SCORES = WAIS_INDEX_SCORES + WMS_INDEX_SCORES


INDEX_MEAN = 100
INDEX_SD = 15

SUBTEST_MEAN = 10
SUBTEST_SD = 3


# ======================================================
# Results table
# ======================================================

def display_results(low_scores_counts, n_sim):

    counts = (
        pd.Series(low_scores_counts)
        .value_counts()
        .sort_index()
    )

    results = pd.DataFrame({
        "Low Scores (X)": counts.index,
        "Point Probability (%)": (counts.values / n_sim) * 100
    })

    results["Cumulative Probability (% ≥ X)"] = (
        results["Point Probability (%)"][::-1]
        .cumsum()[::-1]
    )

    results["Point Probability (%)"] = results["Point Probability (%)"].round(2)
    results["Cumulative Probability (% ≥ X)"] = results["Cumulative Probability (% ≥ X)"].round(2)

    st.dataframe(results, use_container_width=True)


# ======================================================
# Streamlit app
# ======================================================

st.set_page_config(page_title="Neuropsych Base Rate Simulator", layout="wide")
st.title("Neuropsychological Low-Score Base Rate Simulator")

st.markdown(
    """
    **Score Metrics**
    • Index scores: Mean = 100, SD = 15  
    • Subtest scores: Mean = 10, SD = 3  
    • Correlation matrix must be square and numeric  
    """
)

uploaded_file = st.file_uploader(
    "Upload correlation matrix (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:

    corr_df = pd.read_excel(uploaded_file, index_col=0)
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")
    corr_df = corr_df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if corr_df.shape[0] != corr_df.shape[1]:
        st.error("Correlation matrix must be square.")
        st.stop()

    if not corr_df.index.equals(corr_df.columns):
        st.error("Row and column labels must match exactly.")
        st.stop()

    if corr_df.isna().any().any():
        st.error("Matrix contains non-numeric or empty cells.")
        st.stop()

    st.success("Correlation matrix loaded.")

    variables = corr_df.index.tolist()

    selected_vars = st.multiselect(
        "Select variables",
        options=variables,
        default=variables
    )

    if len(selected_vars) < 2:
        st.warning("Select at least two variables.")
        st.stop()

    selected_indices = [v for v in selected_vars if v in INDEX_SCORES]
    selected_subtests = [v for v in selected_vars if v not in INDEX_SCORES]

    col1, col2, col3 = st.columns(3)

    with col1:
        index_cutoff = st.number_input(
            "Index low-score cutoff",
            value=85,
            step=1
        )

    with col2:
        subtest_cutoff = st.number_input(
            "Subtest low-score cutoff",
            value=7,
            step=1
        )

    with col3:
        n_sim = st.number_input(
            "Number of simulations",
            min_value=1000,
            max_value=500000,
            value=100000,
            step=1000
        )

    if st.button("Run Simulation"):

        st.subheader("Index Score Base Rates")

        if len(selected_indices) >= 2:
            corr_idx = corr_df.loc[selected_indices, selected_indices].values
            corr_idx_pd = nearest_pd(corr_idx)

            z_idx = np.random.multivariate_normal(
                mean=np.zeros(len(selected_indices)),
                cov=corr_idx_pd,
                size=n_sim
            )

            idx_scores = INDEX_MEAN + INDEX_SD * z_idx
            low_idx = (idx_scores < index_cutoff).sum(axis=1)

            display_results(low_idx, n_sim)

        else:
            st.info("At least two index scores required.")

        st.subheader("Subtest Score Base Rates")

        if len(selected_subtests) >= 2:
            corr_sub = corr_df.loc[selected_subtests, selected_subtests].values
            corr_sub_pd = nearest_pd(corr_sub)

            z_sub = np.random.multivariate_normal(
                mean=np.zeros(len(selected_subtests)),
                cov=corr_sub_pd,
                size=n_sim
            )

            sub_scores = SUBTEST_MEAN + SUBTEST_SD * z_sub
            low_sub = (sub_scores < subtest_cutoff).sum(axis=1)

            display_results(low_sub, n_sim)

        else:
            st.info("At least two subtests required.")