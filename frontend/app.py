"""
Streamlit dashboard for the Multi-Agent AI Data Analyst.
Upload a dataset, ask a question, and inspect the full agent trace alongside
the final evidence-backed report.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Multi-Agent Data Analyst", layout="wide")
st.title("🧑‍💼 Multi-Agent AI Data Analyst")

tab_upload, tab_analyze = st.tabs(["1. Upload data", "2. Ask a question"])

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

with tab_upload:
    st.subheader("Upload a CSV or Excel file")
    uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if uploaded and st.button("Upload"):
        with st.spinner("Loading dataset..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.ok:
            data = resp.json()
            st.session_state.dataset_id = data["dataset_id"]
            st.success(f"Loaded '{data['dataset_id']}' — {data['num_rows']} rows, {data['num_columns']} columns")
            st.json(data["schema"])
        else:
            st.error(resp.text)

with tab_analyze:
    st.subheader("Ask an analytical question")
    if st.session_state.dataset_id:
        st.info(f"Active dataset: **{st.session_state.dataset_id}**")
    else:
        st.warning("Upload a dataset first (tab 1).")

    question = st.text_area("Question", placeholder="Why did sales decrease last month?")
    if st.button("Analyze", type="primary") and question and st.session_state.dataset_id:
        with st.spinner("Manager → SQL/Python/Statistics agents → Critic → Report..."):
            resp = requests.post(
                f"{API_URL}/analyze",
                json={"dataset_id": st.session_state.dataset_id, "question": question},
            )
        if resp.ok:
            data = resp.json()
            st.markdown("### 📄 Final Report")
            st.markdown(data["report"])

            if data["chart_paths"]:
                st.markdown("### 📊 Charts")
                cols = st.columns(min(3, len(data["chart_paths"])))
                for i, path in enumerate(data["chart_paths"]):
                    try:
                        cols[i % len(cols)].image(path)
                    except Exception:
                        cols[i % len(cols)].write(path)

            st.markdown("### 🔍 Critic revision rounds")
            for r in data["critic_rounds"]:
                st.write(f"Round {r['round']}: **{r['verdict']}**" + (f" — {r['feedback']}" if r["feedback"] else ""))

            with st.expander(f"🛠️ Full agent trace ({len(data['audit_trail'])} tool calls)"):
                for call in data["audit_trail"]:
                    icon = "✅" if call["success"] else "❌"
                    st.markdown(f"**{icon} {call['agent']} → `{call['tool']}`**")
                    st.code(str(call["arguments"]))
                    st.text(call["result_summary"])

            st.caption(
                f"Session {data['session_id']} · {data['revision_rounds_used']} revision round(s) · "
                f"{data['elapsed_seconds']:.1f}s"
            )
        else:
            st.error(resp.text)
