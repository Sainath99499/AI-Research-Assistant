import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("🧠 AI Research Assistant")
st.write("Upload your dataset and train a machine learning model instantly.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Show a preview and columns to help the user pick the correct target
if uploaded_file is not None:
    try:
        # Seek to start in case Streamlit has already read the file
        uploaded_file.seek(0)
        df_preview = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded CSV")
        st.dataframe(df_preview.head())
        st.write("**Columns:**", list(df_preview.columns))
    except Exception as e:
        st.warning(f"Could not preview CSV: {e}")

# Target Column Input
if uploaded_file is None:
    target_column = st.text_input("Enter Target Column Name")
else:
    try:
        uploaded_file.seek(0)
        df_preview = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded CSV")
        st.dataframe(df_preview.head())
        st.write("**Columns:**")
        st.json(list(df_preview.columns))
        target_column = st.selectbox("Select Target Column", options=list(df_preview.columns))
    except Exception as e:
        st.warning(f"Could not preview CSV: {e}")
        target_column = st.text_input("Enter Target Column Name")

if st.button("Train Model"):

    if uploaded_file is None or (target_column == "" or target_column is None):
        st.warning("Please upload file and enter/select target column.")
    else:
        # Streamlit's uploaded_file provides getvalue() (bytes) and name
        uploaded_file.seek(0)
        file_bytes = uploaded_file.getvalue()
        files = {
            "file": (uploaded_file.name, file_bytes, "text/csv")
        }
        data = {"target_column": target_column}

        response = requests.post(
            f"{API_URL}/train",
            files=files,
            data=data
        )

        if response.status_code == 200:
            result = response.json()

            st.success("Model Comparison Completed ✅")

            st.write("### Model Accuracies")
            st.json(result["accuracies"])

            st.metric("Best Model", result["best_model"])
            st.metric("Best Accuracy", result["best_accuracy"])

        else:
            st.error(response.json()["error"])

st.markdown("---")

# Download PDF Report
if st.button("Download PDF Report"):
    response = requests.get(f"{API_URL}/download-report")

    if response.status_code == 200:
        with open("report.pdf", "wb") as f:
            f.write(response.content)

        with open("report.pdf", "rb") as f:
            st.download_button(
                label="Download Report",
                data=f,
                file_name="AI_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.error("Generate report first from backend.")