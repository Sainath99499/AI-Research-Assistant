import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.title("🧠 AI Research Assistant")
st.write("Upload your dataset and train a machine learning model instantly.")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# Target Column Input
target_column = st.text_input("Enter Target Column Name")

if st.button("Train Model"):

    if uploaded_file is None or target_column == "":
        st.warning("Please upload file and enter target column.")
    else:
        response = requests.post(
            f"{API_URL}/train",
            files={"file": uploaded_file},
            data={"target_column": target_column}
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