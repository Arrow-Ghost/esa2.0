import streamlit as st
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="Smart Belt Dashboard", layout="centered")

st.title("Smart Belt AI Dashboard")
st.caption("Prototype context-aware fall risk assessment")

event = st.selectbox("ML event", ["fall", "no_fall"])
confidence = st.slider("ML confidence", 0.0, 1.0, 0.94, 0.01)

heart_rate = st.number_input("Heart rate", min_value=0, value=118)
temperature = st.number_input("Temperature °C", value=38.1)
inactivity = st.number_input("Inactivity seconds", min_value=0, value=45)

user_response = st.selectbox(
    "User response",
    ["unknown", "okay", "needs_help", "no_response"]
)

latitude = st.number_input("Latitude", value=12.9716, format="%.6f")
longitude = st.number_input("Longitude", value=77.5946, format="%.6f")

if st.button("Assess risk"):
    payload = {
        "ml_prediction": {
            "event": event,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "context": {
            "heart_rate": heart_rate,
            "temperature_c": temperature,
            "latitude": latitude,
            "longitude": longitude,
            "inactivity_seconds": inactivity,
            "user_response": user_response
        }
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/assess",
            json=payload,
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        result = data["assessment"]

        st.metric("Risk Score", result["risk_score"])
        st.write("Risk level:", result["risk_level"])
        st.write("Alert:", result["alert"])
        st.write("Recommended action:", result["recommended_action"])

        st.subheader("Reasons")
        for reason in result["reasons"]:
            st.write("-", reason)

        st.subheader("Raw API response")
        st.json(data)

    except requests.RequestException as exc:
        st.error(
            "Could not reach API. Start it with: "
            "`uvicorn app.main:app --reload`"
        )
        st.exception(exc)
