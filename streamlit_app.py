"""
inBalance – Streamlit UI
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
from datetime import date, timedelta

API_BASE = "http://localhost:8000"

# ── Phase metadata ────────────────────────────────────────────────────────────
PHASE_META = {
    "MEN": {"label": "Menstrual",    "color": "#c0392b", "emoji": "🔴"},
    "FOL": {"label": "Follicular",   "color": "#27ae60", "emoji": "🌱"},
    "OVU": {"label": "Ovulatory",    "color": "#f39c12", "emoji": "✨"},
    "LUT": {"label": "Luteal",       "color": "#8e44ad", "emoji": "🌙"},
    "LL":  {"label": "Late Luteal",  "color": "#6c3483", "emoji": "🌒"},
}

DOMAIN_EMOJI = {
    "exercise":  "🏃",
    "nutrition": "🥗",
    "sleep":     "😴",
    "mental":    "🧠",
}

INTENSITY_COLOR = {
    "Low":      "#27ae60",
    "Moderate": "#f39c12",
    "High":     "#c0392b",
}

# ── Enum options ──────────────────────────────────────────────────────────────
ENERGY_OPTS        = ["Very Low", "Low", "Medium", "High", "Very High"]
STRESS_OPTS        = ["Low", "Moderate", "High", "Severe"]
MOOD_OPTS          = ["Very Low", "Low", "Medium", "High", "Very High"]
CRAMP_OPTS         = ["None", "Mild", "Moderate", "Severe", "Very Severe"]
BLOATING_OPTS      = ["None", "Mild", "Moderate", "Severe"]
SLEEP_OPTS         = ["Poor", "Fair", "Good", "Very Good"]
IRRITABILITY_OPTS  = ["None", "Mild", "Moderate", "Severe"]
MOOD_SWING_OPTS    = ["None", "Mild", "Moderate", "Severe"]
CRAVINGS_OPTS      = ["None", "Mild", "Moderate", "Strong"]
FLOW_OPTS          = ["None", "Light", "Moderate", "Heavy", "Very Heavy"]
TENDERNESS_OPTS    = ["None", "Mild", "Moderate", "Severe"]
INDIGESTION_OPTS   = ["None", "Mild", "Moderate", "Severe"]
BRAIN_FOG_OPTS     = ["None", "Mild", "Moderate", "Severe"]
MUCUS_OPTS         = ["unknown", "dry", "sticky", "creamy", "watery", "eggwhite"]
PHASE_OPTS         = {
    "MEN": "Menstrual",
    "FOL": "Follicular",
    "OVU": "Ovulatory",
    "LUT": "Luteal",
    "LL":  "Late Luteal",
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="inBalance",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .phase-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        margin-bottom: 8px;
    }
    .activity-card {
        background: #f8f9fa;
        border-left: 4px solid #8e44ad;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .activity-card.blocked {
        background: #fff5f5;
        border-left-color: #c0392b;
        opacity: 0.7;
    }
    .score-pill {
        display: inline-block;
        background: #8e44ad;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .score-pill.blocked {
        background: #c0392b;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .fertility-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌸 inBalance")
    st.caption("Cycle-aware wellness engine")
    st.divider()

    mode = st.radio(
        "Mode",
        ["Full Prediction", "Manual Phase"],
        help="Full Prediction uses your period history to estimate your cycle phase. Manual lets you pick the phase yourself.",
    )

    st.divider()
    user_id = st.text_input("User ID (optional)", placeholder="e.g. user_001", help="Enables personalization across sessions.")
    top_n   = st.slider("Recommendations per category", 1, 6, 3)

    st.divider()
    st.caption(f"API: {API_BASE}")
    if st.button("🔍 Health check"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=3)
            if r.ok:
                st.success(f"API online — v{r.json().get('version','?')}")
            else:
                st.error(f"API error {r.status_code}")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")


# ── Main layout ───────────────────────────────────────────────────────────────
st.header("Daily Check-In")

# ── Period / Phase section ────────────────────────────────────────────────────
if mode == "Full Prediction":
    with st.expander("📅 Period History", expanded=True):
        st.caption("Enter your recent period start dates. More history = better prediction.")
        num_periods = st.number_input("How many past periods to enter?", 1, 12, 3, step=1)

        period_starts = []
        cols = st.columns(min(int(num_periods), 4))
        for i in range(int(num_periods)):
            default = date.today() - timedelta(days=28 * (int(num_periods) - i - 1))
            d = cols[i % 4].date_input(f"Period {i + 1}", value=default, key=f"pd_{i}")
            period_starts.append(str(d))

        period_start_logged = st.checkbox("I started my period today")
else:
    phase_key = st.selectbox(
        "Current Phase",
        options=list(PHASE_OPTS.keys()),
        format_func=lambda k: f"{PHASE_META[k]['emoji']} {PHASE_OPTS[k]}",
    )

# ── Symptom signals ───────────────────────────────────────────────────────────
st.subheader("How are you feeling today?")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Energy & Vitality**")
    energy       = st.select_slider("Energy",        ENERGY_OPTS,       value="Medium")
    mood         = st.select_slider("Mood",          MOOD_OPTS,         value="Medium")
    sleep_q      = st.select_slider("Sleep Quality", SLEEP_OPTS,        value="Good")
    brain_fog    = st.select_slider("Brain Fog",     BRAIN_FOG_OPTS,    value="None")

with c2:
    st.markdown("**Pain & Discomfort**")
    cramp        = st.select_slider("Cramps",             CRAMP_OPTS,       value="None")
    headache     = st.slider("Headache",            0, 10, 0)
    bloating     = st.select_slider("Bloating",          BLOATING_OPTS,    value="None")
    tenderness   = st.select_slider("Breast Tenderness",  TENDERNESS_OPTS,  value="None")
    indigestion  = st.select_slider("Indigestion",        INDIGESTION_OPTS, value="None")

with c3:
    st.markdown("**Mood & Behaviour**")
    stress       = st.select_slider("Stress",        STRESS_OPTS,       value="Low")
    irritability = st.select_slider("Irritability",  IRRITABILITY_OPTS, value="None")
    mood_swing   = st.select_slider("Mood Swings",   MOOD_SWING_OPTS,   value="None")
    cravings     = st.select_slider("Cravings",      CRAVINGS_OPTS,     value="None")
    flow         = st.select_slider("Flow",          FLOW_OPTS,         value="None")

# ── Lifestyle signals ─────────────────────────────────────────────────────────
with st.expander("🔬 Additional Signals (optional)"):
    lc1, lc2, lc3 = st.columns(3)
    mucus    = lc1.selectbox("Cervical Mucus", MUCUS_OPTS, index=0)
    appetite = lc2.slider("Appetite (0–10)", 0, 10, 5)
    exercise = lc3.slider("Exercise Level (0–10)", 0, 10, 5)

st.divider()

# ── Submit ────────────────────────────────────────────────────────────────────
submitted = st.button("✨ Get My Recommendations", type="primary", use_container_width=True)

if submitted:
    common = {
        "ENERGY":            energy,
        "STRESS":            stress,
        "MOOD":              mood,
        "CRAMP":             cramp,
        "HEADACHE":          headache,
        "BLOATING":          bloating,
        "SLEEP_QUALITY":     sleep_q,
        "IRRITABILITY":      irritability,
        "MOOD_SWING":        mood_swing,
        "CRAVINGS":          cravings,
        "FLOW":              flow,
        "BREAST_TENDERNESS": tenderness,
        "INDIGESTION":       indigestion,
        "BRAIN_FOG":         brain_fog,
        "cervical_mucus":    mucus,
        "appetite":          appetite,
        "exercise_level":    exercise,
        "top_n":             top_n,
    }
    if user_id.strip():
        common["user_id"] = user_id.strip()

    with st.spinner("Analysing your signals…"):
        try:
            if mode == "Full Prediction":
                payload = {
                    **common,
                    "period_starts":       period_starts,
                    "period_start_logged": period_start_logged,
                }
                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=15)
                is_predict = True
            else:
                payload = {**common, "PHASE": phase_key}
                resp = requests.post(f"{API_BASE}/recommend", json=payload, timeout=15)
                is_predict = False

            if not resp.ok:
                st.error(f"API error {resp.status_code}: {resp.text}")
                st.stop()

            data = resp.json()

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure the FastAPI server is running on localhost:8000.")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.divider()

    # ── Phase & Cycle Info ────────────────────────────────────────────────────
    if is_predict:
        phase_data  = data.get("phase", {})
        fertility   = data.get("fertility", {})
        timing      = data.get("timing", {})
        recs        = data.get("recommendations", {})
        rec_mode    = data.get("recommender_mode", "")
        ctx         = data.get("context_completeness", 0)

        phase_code  = phase_data.get("code", "")
        meta        = PHASE_META.get(phase_code, {"label": phase_code, "color": "#555", "emoji": "🌸"})
        confidence  = phase_data.get("confidence", 0)
        cycle_day   = phase_data.get("cycle_day")
        cycle_len   = phase_data.get("cycle_length")

        st.markdown(
            f'<div class="phase-badge" style="background:{meta["color"]}">'
            f'{meta["emoji"]} {meta["label"]} Phase'
            f'</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence",   f"{confidence * 100:.0f}%")
        m2.metric("Cycle Day",    cycle_day or "—")
        m3.metric("Cycle Length", f"{cycle_len} days" if cycle_len else "—")
        m4.metric("Context",      f"{ctx * 100:.0f}%")

        if rec_mode == "phase_blind":
            st.info("⚠️ Running in **phase-blind** mode — cycle confidence is low, recommendations are symptom-driven only.")

        fcol, tcol = st.columns(2)

        with fcol:
            st.markdown("#### Fertility")
            fstatus = fertility.get("status", "—").replace("_", " ").title()
            fconf   = fertility.get("signal_confidence", 0)
            fcolors = {
                "High Fertility": "#27ae60",
                "Fertile":        "#f39c12",
                "Low Fertility":  "#95a5a6",
                "Infertile":      "#bdc3c7",
            }
            fc = fcolors.get(fstatus, "#8e44ad")
            st.markdown(
                f'<span class="fertility-badge" style="background:{fc};color:white">{fstatus}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Signal confidence: {fconf * 100:.0f}%")
            for exp in fertility.get("explanations", []):
                st.caption(f"• {exp}")

        with tcol:
            st.markdown("#### Timing")
            if timing.get("predicted_next_period"):
                st.metric("Next Period", timing["predicted_next_period"])
            if timing.get("ovulation_date"):
                st.metric("Ovulation", timing["ovulation_date"])
            fw = timing.get("fertile_window")
            if fw and len(fw) == 2:
                st.caption(f"Fertile window: {fw[0]} → {fw[1]}")
            if timing.get("note"):
                st.caption(f"📝 {timing['note']}")

        phase_probs = phase_data.get("phase_probs", {})
        if phase_probs:
            with st.expander("Phase probabilities"):
                prob_cols = st.columns(len(phase_probs))
                for i, (code, prob) in enumerate(phase_probs.items()):
                    m = PHASE_META.get(code, {})
                    prob_cols[i].metric(f"{m.get('emoji','')}{code}", f"{prob * 100:.1f}%")

    else:
        recs = data.get("results", {})
        meta = PHASE_META.get(phase_key, {"label": PHASE_OPTS[phase_key], "color": "#8e44ad", "emoji": "🌸"})
        st.markdown(
            f'<div class="phase-badge" style="background:{meta["color"]}">'
            f'{meta["emoji"]} {meta["label"]} Phase (manual)'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Recommendations ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Your Recommendations")

    domains = [d for d in ["exercise", "nutrition", "sleep", "mental"] if d in recs]
    if not domains:
        st.warning("No recommendations returned.")
        st.stop()

    tabs = st.tabs([f"{DOMAIN_EMOJI.get(d,'📌')} {d.capitalize()}" for d in domains])

    for tab, domain in zip(tabs, domains):
        with tab:
            domain_data = recs[domain]
            categories  = list(domain_data.keys())

            cat_containers = st.columns(len(categories)) if len(categories) > 1 else [st.container()]

            for cat_col, category in zip(cat_containers, categories):
                with cat_col:
                    st.markdown(f'<div class="section-title">{category.capitalize()}</div>', unsafe_allow_html=True)
                    activities = domain_data[category]
                    if not activities:
                        st.caption("No activities returned.")
                        continue

                    for act in activities:
                        blocked   = act.get("is_blocked", False)
                        score     = act.get("final_score", 0)
                        intensity = act.get("intensity", "")
                        dur_min   = act.get("dur_min")
                        dur_max   = act.get("dur_max")
                        rank      = act.get("rank", "")
                        text      = act.get("text", "")

                        card_class   = "activity-card blocked" if blocked else "activity-card"
                        pill_class   = "score-pill blocked"    if blocked else "score-pill"
                        blocked_note = " 🚫 Blocked" if blocked else ""
                        dur_str      = f" · {dur_min}–{dur_max} min" if dur_min and dur_max else ""
                        int_color    = INTENSITY_COLOR.get(intensity, "#555")
                        int_str      = f'<span style="color:{int_color};font-weight:600">{intensity}</span>' if intensity else ""

                        st.markdown(
                            f"""
                            <div class="{card_class}">
                              <span class="{pill_class}">#{rank} &nbsp; {score:.0f}</span>{blocked_note}
                              <br/><span style="font-size:0.95rem">{text}</span>
                              <br/><span style="font-size:0.8rem;color:#888">{int_str}{dur_str}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # ── Personalization footer ────────────────────────────────────────────────
    if is_predict:
        pers = data.get("personalization")
        if pers:
            with st.expander("Personalization data"):
                st.json(pers)
