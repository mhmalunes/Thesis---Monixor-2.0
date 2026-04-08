import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monixor 2.0",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* Base */
        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
            background-color: #F7F3EE;
            color: #1A1A1A;
        }
        .stApp {
            background-color: #F7F3EE;
        }

        /* Hide Streamlit chrome */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 2rem; padding-bottom: 5rem; }

        /* Bottom nav bar — fixed to viewport bottom */
        .nav-bar-wrapper {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            background: #FFFFFF;
            border-top: 1.5px solid #E0D8D0;
            z-index: 9999;
            padding: 0.4rem 0 0.5rem;
        }
        .nav-bar-inner {
            display: flex;
            justify-content: space-around;
            align-items: center;
        }
        .nav-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-size: 0.7rem;
            color: #999;
            gap: 2px;
            min-width: 70px;
            text-align: center;
            text-decoration: none;
            cursor: pointer;
        }
        .nav-item.active { color: #7B1C1C; font-weight: 700; }
        .nav-icon { font-size: 1.3rem; line-height: 1; }

        /* Primary buttons — always solid maroon, white text */
        .stButton > button {
            background-color: #7B1C1C !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.55rem 1.4rem !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #5e1515 !important;
        }
        .stButton > button:active {
            background-color: #4a1010 !important;
        }

        /* Text inputs */
        .stTextInput > div > div > input {
            border: 1.5px solid #d4c8bc !important;
            border-radius: 8px !important;
            background-color: #FFFFFF !important;
            font-family: 'DM Sans', sans-serif !important;
            padding: 0.5rem 0.75rem !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #7B1C1C !important;
            box-shadow: 0 0 0 2px rgba(123,28,28,0.15) !important;
        }

        /* Cards */
        .card {
            background: #FFFFFF;
            border: 1px solid #E0D8D0;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        }

        /* Gold accent text */
        .accent { color: #C9922A; font-weight: 600; }

        /* Section headings */
        h1, h2, h3 {
            font-family: 'DM Sans', sans-serif;
            color: #7B1C1C;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "screen": "login",        # current screen key
        "authenticated": False,
        "user": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Navigation helper ─────────────────────────────────────────────────────────
def go_to(screen: str):
    st.session_state.screen = screen
    st.rerun()

# ── Screen imports (add as screens are built) ─────────────────────────────────
# from screens.dashboard     import render as render_dashboard
# from screens.patient_list  import render as render_patient_list
# from screens.patient_detail import render as render_patient_detail
# from screens.transcription import render as render_transcription
# from screens.review        import render as render_review
# from screens.history       import render as render_history
from screens.settings      import render as render_settings

# ── Login screen (inline — no separate file yet) ──────────────────────────────
def render_login():
    # Centered narrow card
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            """
            <div style="text-align:center; margin-bottom:2rem;">
                <div style="font-size:3rem;">🏥</div>
                <h1 style="margin:0.25rem 0 0.1rem; font-size:2rem;">Monixor 2.0</h1>
                <p style="color:#C9922A; font-weight:600; font-size:0.95rem; margin:0;">
                    Vital Signs Transcription System
                </p>
                <p style="color:#888; font-size:0.82rem; margin-top:0.25rem;">
                    PGH Medical Intensive Care Unit
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            "<p style='font-weight:600; margin-bottom:0.25rem;'>Employee ID</p>",
            unsafe_allow_html=True,
        )
        employee_id = st.text_input(
            "Employee ID", placeholder="e.g. PGH-12345", label_visibility="collapsed"
        )

        st.markdown(
            "<p style='font-weight:600; margin-bottom:0.25rem; margin-top:1rem;'>Password</p>",
            unsafe_allow_html=True,
        )
        password = st.text_input(
            "Password", type="password", placeholder="Enter your password",
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
        if st.button("Log In", key="btn_login"):
            if employee_id.strip() and password.strip():
                # Placeholder auth — replace with real logic later
                st.session_state.authenticated = True
                st.session_state.user = {"id": employee_id.strip(), "name": "Nurse"}
                go_to("dashboard")
            else:
                st.error("Please enter your Employee ID and password.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<p style='text-align:center; color:#aaa; font-size:0.78rem; margin-top:1.5rem;'>"
            "Authorized personnel only · Monixor 2.0 © 2026"
            "</p>",
            unsafe_allow_html=True,
        )

# ── Bottom navigation bar ─────────────────────────────────────────────────────
NAV_ITEMS = [
    ("dashboard", "🏠", "Home"),
    ("patient_list", "📋", "Records"),
    ("history", "🕐", "History"),
    ("settings", "⚙️", "Settings"),
]

def render_bottom_nav(current_screen: str):
    items_html = ""
    for screen_key, icon, label in NAV_ITEMS:
        active_class = "active" if current_screen == screen_key else ""
        # Use query param link so clicking navigates via Streamlit's URL mechanism
        items_html += (
            f"<div class='nav-item {active_class}' "
            f"onclick=\"window.location.href='?nav={screen_key}'\">"
            f"<span class='nav-icon'>{icon}</span>{label}</div>"
        )

    st.markdown(
        f"<div class='nav-bar-wrapper'><div class='nav-bar-inner'>{items_html}</div></div>",
        unsafe_allow_html=True,
    )

    # Handle nav click via query params
    params = st.query_params
    if "nav" in params and params["nav"] != current_screen:
        target = params["nav"]
        st.query_params.clear()
        go_to(target)


# ── Router ────────────────────────────────────────────────────────────────────
def main():
    screen = st.session_state.screen

    if screen == "login":
        render_login()
        return

    # All screens below require authentication
    if not st.session_state.authenticated:
        go_to("login")
        return

    if screen == "dashboard":
        st.info("Dashboard screen — coming soon.")
    elif screen == "patient_list":
        st.info("Patient List screen — coming soon.")
    elif screen == "patient_detail":
        st.info("Patient Detail screen — coming soon.")
    elif screen == "transcription":
        st.info("Transcription screen — coming soon.")
    elif screen == "review":
        st.info("Review screen — coming soon.")
    elif screen == "history":
        st.info("History screen — coming soon.")
    elif screen == "settings":
        render_settings()
    else:
        go_to("login")

    render_bottom_nav(screen)

main()
