import json
import os
import streamlit as st

# ── Friendly display names for each vital ────────────────────────────────────
VITAL_FRIENDLY = {
    "HR":   "Heart Rate",
    "SpO2": "Oxygen Level (SpO2)",
    "PR":   "Pulse Rate",
    "Resp": "Breathing Rate",
    "NIBP": "Blood Pressure (NIBP)",
    "Temp": "Temperature",
}

# ── Built-in defaults (Mindray Beneview T8) ───────────────────────────────────
DEFAULT_ALIASES = {
    "HR":   ["ecg", "hr", "heart", "eco", "ecd", "ecq"],
    "SpO2": ["spo2", "spoz", "spo", "sp02", "sp0", "sp0z", "oxygen"],
    "PR":   ["pr"],
    "Resp": ["resp", "rosp", "rsp", "resp.", "rr"],
    "NIBP": ["nibp", "n1bp", "nibp:"],
    "Temp": ["temp", "tmp", "t1", "temp.", "tc"],
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "monitor_config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {k: list(v) for k, v in DEFAULT_ALIASES.items()}


def save_config(config: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def render():
    st.markdown("## Monitor Settings")
    st.markdown(
        "Type what labels appear on your monitor screen next to each vital sign. "
        "Separate multiple alternatives with a comma."
    )
    st.markdown("---")

    config = load_config()
    updated = {}

    for key, friendly in VITAL_FRIENDLY.items():
        current_aliases = config.get(key, DEFAULT_ALIASES[key])
        current_text = ", ".join(current_aliases)

        st.markdown(f"**{friendly}**")
        col1, col2 = st.columns([5, 1])
        with col1:
            val = st.text_input(
                friendly,
                value=current_text,
                key=f"alias_{key}",
                label_visibility="collapsed",
                placeholder=f"e.g. {current_text}",
            )
        with col2:
            if st.button("Reset", key=f"reset_{key}"):
                val = ", ".join(DEFAULT_ALIASES[key])
                st.session_state[f"alias_{key}"] = val

        aliases = [a.strip().lower() for a in val.split(",") if a.strip()]
        updated[key] = aliases if aliases else list(DEFAULT_ALIASES[key])

    st.markdown("---")
    if st.button("Save Settings", use_container_width=True):
        save_config(updated)
        st.success("Settings saved. The pipeline will use these labels on the next scan.")
