import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Cardiac Risk Assessment", page_icon="❤️", layout="wide")


@st.cache_resource
def load_model_and_metadata():
    model = joblib.load("heart_model.joblib")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return model, metadata


model, metadata = load_model_and_metadata()

st.title("❤️ Cardiac Risk Assessment")
st.caption("Educational demo — not a medical device and not a substitute for professional diagnosis.")

st.sidebar.header("Patient Vitals")
inputs = {}

for feat in metadata["numeric_features"]:
    lo, hi = metadata["numeric_ranges"][feat]
    default = metadata["numeric_defaults"][feat]
    inputs[feat] = st.sidebar.slider(feat, float(lo), float(hi), float(default))

for feat in metadata["categorical_features"]:
    options = metadata["categorical_options"][feat]
    inputs[feat] = st.sidebar.selectbox(feat, options)

if st.sidebar.button("Assess Risk", type="primary"):
    input_df = pd.DataFrame([inputs])
    probability = model.predict_proba(input_df)[0][1]
    is_high_risk = probability >= 0.5

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Result")
        st.metric("Estimated risk score", f"{probability * 100:.1f}%")
        if is_high_risk:
            st.error("Higher risk indicated")
        else:
            st.success("Lower risk indicated")

    with col2:
        st.subheader("Model Comparison")
        st.dataframe(pd.DataFrame(metadata["model_comparison"]), hide_index=True)
        st.caption(f"Model used: **{metadata['model_name']}**")

    st.subheader("What Drives This Model, Overall")
    imp_df = pd.DataFrame(metadata["global_importance"])
    st.bar_chart(imp_df.set_index("feature"))
else:
    st.info("Fill in the vitals in the sidebar and click **Assess Risk**.")
