/* InBalance — editorial UI wired to FastAPI (/predict).
 *
 * State machine (simplified):
 *   boot  → no period_starts saved → "welcome"
 *          → period_starts saved   → fetch /predict → "app" (Today/Check-in/Guide/Cycle)
 *
 * Persistence: period_starts + user_id are stored in localStorage so the
 * backend's recent-logs / personalization pipeline can attach history.
 *
 * The prototype's PHASE_COPY (prose + fallback guidance) is kept as a
 * fallback when /predict returns no recommendations or the recommender is
 * in phase_blind mode with thin output. The live `.recommendations` take
 * precedence and are mapped into the Move/Eat/Rest/Notice rows.
 */

const { useState, useEffect, useMemo, useCallback } = React;

// ── constants ────────────────────────────────────────────────────────────────
const CYCLE_LEN_DEFAULT = 28;

const PHASES = [
  { key:"menstrual",  name:"Menstrual",  code:"MEN", start:1,  end:5,  seg:"m" },
  { key:"follicular", name:"Follicular", code:"FOL", start:6,  end:13, seg:"f" },
  { key:"ovulatory",  name:"Ovulatory",  code:"OVU", start:14, end:16, seg:"o" },
  { key:"luteal",     name:"Luteal",     code:"LUT", start:17, end:28, seg:"l" },
];
// phase_code → prototype key (for palette + prose fallback)
const CODE_TO_KEY = { MEN:"menstrual", FOL:"follicular", OVU:"ovulatory", LUT:"luteal", LL:"luteal" };

// Fallback prose per phase, lifted from the prototype. Used when the API
// returns no prose, so the editorial voice survives cold starts.
const PHASE_COPY = {
  menstrual: {
    prose: <>The body is <em>shedding</em>, not weak. Iron is leaving the system — make room for warmth, softness, and fewer decisions.</>,
    body: ["Energy is at its lowest; rest is productive.","Iron and magnesium stores are dropping.","Pain tolerance may be reduced."],
  },
  follicular: {
    prose: <>Estrogen is <em>climbing</em>. Planning, learning and lifting all tend to land more easily now — this is the week to try something new.</>,
    body: ["Estrogen rising; mood and cognition typically climb.","Strength and coordination improve through mid-phase.","Libido begins to rise mid-cycle."],
  },
  ovulatory: {
    prose: <>The fertile window is <em>open</em>. Output is high but volatile — protect your attention and use this energy on what matters.</>,
    body: ["Estrogen peaks, then a brief LH surge.","Basal temperature will rise about 0.3°C after ovulation.","Verbal fluency and social confidence often highest."],
  },
  luteal: {
    prose: <>Progesterone is <em>dominant</em>. The body is running warmer and hungrier; stress lands harder. Slow the pace on purpose.</>,
    body: ["Progesterone dominant; slight temperature elevation.","Appetite increases; insulin sensitivity drops.","Cortisol response is higher — stress hits harder."],
  },
};

// Map the recommender's 4 domains to the Move/Eat/Rest/Notice rows.
const DOMAIN_KIND = {
  exercise:  "Move",
  nutrition: "Eat",
  sleep:     "Rest",
  mental:    "Notice",
};
const PLAN_ORDER = ["exercise","nutrition","sleep","mental"];

// ── localStorage helpers ─────────────────────────────────────────────────────
const LS = {
  get: (k, fallback=null) => {
    try { const v = localStorage.getItem(k); return v==null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  del: (k)    => { try { localStorage.removeItem(k); } catch {} },
};

// Stable per-device user_id so the backend can attach history + personalization.
function ensureUserId(){
  let uid = LS.get("inbalance.user_id");
  if (uid) return uid;
  uid = "web-" + Math.random().toString(36).slice(2, 10) + "-" + Date.now().toString(36);
  LS.set("inbalance.user_id", uid);
  return uid;
}

// ── Lotus mark + icons ──────────────────────────────────────────────────────
function LotusMark(){
  return (
    <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 38 H42" />
      <path d="M24 38 C 19 30, 19 20, 24 10 C 29 20, 29 30, 24 38 Z" />
      <path d="M24 38 C 14 34, 8 26, 8 18 C 16 20, 22 28, 24 38 Z" />
      <path d="M24 38 C 34 34, 40 26, 40 18 C 32 20, 26 28, 24 38 Z" />
      <circle cx="24" cy="8" r="1.6" fill="currentColor" stroke="none"/>
    </svg>
  );
}
const Icon = ({d,size=20,fill="none"}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">{d}</svg>
);
const I = {
  today:  <Icon d={<><circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="9"/></>} />,
  check:  <Icon d={<><path d="M4 6h16M4 12h10M4 18h16"/></>} />,
  guide:  <Icon d={<><path d="M4 5h12a4 4 0 0 1 4 4v10H8a4 4 0 0 1-4-4Z"/><path d="M4 5v10a4 4 0 0 0 4 4"/></>} />,
  cycle:  <Icon d={<><path d="M12 3v3"/><path d="M12 18v3"/><path d="M3 12h3"/><path d="M18 12h3"/><circle cx="12" cy="12" r="7"/></>} />,
  arrow:  <Icon size={16} d={<path d="M5 12h14M13 6l6 6-6 6"/>} />,
  clock:  <Icon size={14} d={<><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>} />,
};

// ── Shared header ────────────────────────────────────────────────────────────
function Brand(){
  return (
    <div className="hdr">
      <div className="logo">
        <div className="logo-mark"><LotusMark/></div>
        <div className="brand">in<em>Balance</em></div>
      </div>
      <div className="label">v.06</div>
    </div>
  );
}

function TabBar({active, setActive}){
  const tabs = [
    { k:"today",  label:"Today",    icon:I.today },
    { k:"checkin",label:"Check-in", icon:I.check },
    { k:"guide",  label:"Guide",    icon:I.guide },
    { k:"cycle",  label:"Cycle",    icon:I.cycle },
  ];
  return (
    <div className="tabs">
      {tabs.map(t=>(
        <button key={t.k} className={`tab ${active===t.k?'active':''}`} onClick={()=>setActive(t.k)}>
          {t.icon}
          <span>{t.label}</span>
          <span className="tab-dot"></span>
        </button>
      ))}
    </div>
  );
}

// ── Ribbon (cycle visualisation) ─────────────────────────────────────────────
function Ribbon({day, cycleLen, small=false}){
  const total = cycleLen || CYCLE_LEN_DEFAULT;
  const segs = PHASES.map(p => ({...p, span: p.end - p.start + 1}));
  const safeDay = Math.max(1, Math.min(day || 1, total));
  const markerPct = ((safeDay-0.5) / total) * 100;
  const fStart = (12-1)/total*100;
  const fEnd   = (16/total)*100;
  return (
    <div className="cyc-ribbon">
      <div className="rb-h">
        <span>Day 1 · last period</span>
        <span>Day {total} · next period</span>
      </div>
      <div className="rb-track" style={{position:'relative'}}>
        <div className="rb-bg">
          {segs.map(s => (
            <div key={s.key} className={`s ${s.seg}`} style={{flex:s.span}}>
              {!small && <span>{s.name}</span>}
            </div>
          ))}
        </div>
        <div className="rb-fwindow" style={{left:`${fStart}%`, right:`${100-fEnd}%`}}></div>
        <div className="rb-marker" style={{left:`calc(${markerPct}% - 1px)`}}></div>
      </div>
      <div className="rb-days">
        <span>1</span><span>7</span><span>14</span><span>21</span><span>{total}</span>
      </div>
    </div>
  );
}

// ── API helpers ──────────────────────────────────────────────────────────────
async function callPredict(body){
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(()=> "");
    throw new Error(`predict ${res.status}: ${text.slice(0,200)}`);
  }
  return res.json();
}

// Default payload — one period start, no symptoms, just "give me what you've got".
function buildBasePayload(periodStarts, userId, extra={}){
  return {
    user_id: userId,
    period_starts: periodStarts,
    period_start_logged: false,
    cervical_mucus: "unknown",
    appetite: 0,
    exercise_level: 0,
    ENERGY: "Medium",
    STRESS: "Low",
    MOOD: "Medium",
    CRAMP: "None",
    HEADACHE: 0,
    BLOATING: "None",
    SLEEP_QUALITY: "Good",
    IRRITABILITY: "None",
    MOOD_SWING: "None",
    CRAVINGS: "None",
    FLOW: "None",
    BREAST_TENDERNESS: "None",
    INDIGESTION: "None",
    BRAIN_FOG: "None",
    top_n: 4,
    ...extra,
  };
}

// Map a cycle day onto the static phase calendar used by the ribbon. Luteal
// stretches to cycleLen so longer/shorter cycles still terminate cleanly.
function phaseFromDay(day, len){
  const total = len || CYCLE_LEN_DEFAULT;
  const d = Math.max(1, Math.min(day || 1, total));
  for (const p of PHASES) {
    const endAdj = p.key === "luteal" ? total : p.end;
    if (d >= p.start && d <= endAdj) return p;
  }
  return PHASES[PHASES.length - 1];
}

// Collapse /predict response into the shape the UI screens expect.
//
// Phase-selection policy:
//   trustClassifier=false → derive displayed phase from cycle_day (calendar).
//     Used in the thin-data regime (≤1 logged period start) where the L2
//     classifier is unreliable and disagreements with the calendar confuse
//     users more than they help.
//   trustClassifier=true  → use backend phase.code as authoritative.
//     Switched on once the user has ≥2 logged period starts — at that point
//     L1 calendar is grounded in real data and L2's fusion output is
//     trustworthy enough that a classifier/calendar divergence is signal,
//     not noise.
function projectPrediction(resp, { trustClassifier = false } = {}){
  const phase = resp.phase || {};
  const cycleLen = phase.cycle_length || CYCLE_LEN_DEFAULT;
  const cycleDay = phase.cycle_day ?? 1;

  let code, key, label;
  if (trustClassifier && phase.code) {
    code  = phase.code;
    key   = CODE_TO_KEY[code] || "luteal";
    const mapped = PHASES.find(p => p.code === code);
    label = phase.label || (mapped && mapped.name) || "Luteal";
  } else {
    const calP = phaseFromDay(cycleDay, cycleLen);
    code  = calP.code;
    key   = calP.key;
    label = calP.name;
  }

  // Flatten recommendations to plan rows. Prefer exercise→nutrition→sleep→mental.
  const planRows = [];
  const recs = resp.recommendations || {};
  for (const domain of PLAN_ORDER) {
    const cats = recs[domain];
    if (!cats) continue;
    // First item from any category under this domain that isn't blocked.
    let picked = null;
    for (const items of Object.values(cats)) {
      for (const it of (items || [])) {
        if (!it.is_blocked) { picked = it; break; }
      }
      if (picked) break;
    }
    if (picked) {
      planRows.push({
        kind:  DOMAIN_KIND[domain] || domain,
        title: picked.text,
        why:   `Confidence ${Math.round((picked.confidence || 0)*100)}% · domain ${domain}`,
        id:    picked.id,
      });
    }
  }

  // Confidence: backend returns "high" / "moderate" / "low". Map to % for the bar.
  const confWord = (phase.confidence || "").toLowerCase();
  const confPct = confWord === "high" ? 82 : confWord === "moderate" ? 62 : 42;

  return {
    raw:           resp,
    phaseCode:     code,
    phaseKey:      key,
    phaseLabel:    label,
    cycleDay,
    cycleLen,
    confWord:      confWord || "gathering",
    confPct,
    planRows,
    timing:        resp.timing || {},
    fertility:     resp.fertility || null,
  };
}

// ── Period-log sheet ─────────────────────────────────────────────────────────
// Modal for logging a new period start (or backfilling an old one). Writes to
// period_starts and, if date === today, fires period_start_logged=true which
// forces the fusion engine into period_start_override mode.
function PeriodLogSheet({onClose, onSave, saving, error}){
  const todayIso = new Date().toISOString().slice(0,10);
  const [date,setDate] = useState(todayIso);
  const [flow,setFlow] = useState(2); // default "Moderate"

  const isToday = date === todayIso;

  return (
    <div className="sheet-overlay" onClick={onClose}>
      <div className="sheet" onClick={e=>e.stopPropagation()}>
        <div className="sheet-head">
          <div className="eyebrow">Log period</div>
          <button className="sheet-x" onClick={onClose} aria-label="Close">×</button>
        </div>
        <h2 className="sheet-title">When did your period <em>start?</em></h2>
        <p className="sheet-sub">
          {isToday
            ? "Logging for today — your phase will reset to menstrual immediately."
            : "Logging a past start — predictions will recalibrate from this date."}
        </p>

        <label htmlFor="plsd">Start date</label>
        <input id="plsd" type="date" max={todayIso} value={date} onChange={e=>setDate(e.target.value)} />

        <div className="ci-group" style={{marginTop:16,paddingTop:0,borderTop:'none'}}>
          <div className="gh"><div className="gl">Flow</div><div className="gs">today</div></div>
          <div className="scale">
            {FLOW_SCALE.map((x,i)=>(
              <div key={x} className={`s ${flow===i?'on':''}`} onClick={()=>setFlow(i)}>
                {i===0 ? "—" : "●".repeat(Math.min(i,4))}
              </div>
            ))}
          </div>
          <div className="scale-legend">{FLOW_SCALE.map(x=><span key={x}>{x}</span>)}</div>
        </div>

        {error && <div className="err" style={{marginTop:12}}>{error}</div>}

        <div className="sheet-actions">
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" disabled={!date || saving} onClick={()=>onSave({ date, flow: FLOW_SCALE[flow] })}>
            {saving ? "Saving…" : <>Save {I.arrow}</>}
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Screens ──────────────────────────────────────────────────────────────────

function TodayScreen({proj, setTab, onQuickPulse, pulse, onLogPeriod}){
  const copy = PHASE_COPY[proj.phaseKey] || PHASE_COPY.luteal;
  const timing = proj.timing || {};
  const nextPeriod = timing.predicted_next_period;
  const today = new Date();
  const dateStr = today.toLocaleDateString("en-US", { weekday:"short", month:"short", day:"numeric", year:"numeric" }).replace(",", " ·");

  // "next period in Nd" — compute from predicted_next_period if present,
  // otherwise fall back to cycle-length math.
  let daysToPeriod = null;
  if (nextPeriod) {
    const np = new Date(nextPeriod);
    const diff = Math.round((np - today) / 86400000);
    if (diff >= 0) daysToPeriod = diff;
  }
  if (daysToPeriod == null) {
    daysToPeriod = Math.max(1, proj.cycleLen - proj.cycleDay + 1);
  }

  return (
    <div className="page fade-in" key={proj.phaseKey}>
      <div className="today-head">
        <div className="today-date">
          <div className="d">{dateStr}</div>
          <div className="cd">Cycle day {proj.cycleDay} / {proj.cycleLen}</div>
        </div>
      </div>

      <div className="phase-card">
        <div className="row1">
          <div>
            <div className="ph-name">You are in <em>{proj.phaseLabel.toLowerCase()}</em>.</div>
            <div className="ph-day">Day {proj.cycleDay} — {proj.phaseLabel.toLowerCase()}</div>
          </div>
          <div className="ph-tag"><span className="dot"></span>{proj.phaseLabel}</div>
        </div>
        <p className="prose">{copy.prose}</p>
        <div className="meta">
          <span><b>{proj.confPct}%</b> confidence</span>
          <span>Next period in <b>{daysToPeriod}d</b></span>
          {proj.raw.phase?.mode && <span>{proj.raw.phase.mode.replace(/_/g,' ')}</span>}
        </div>
        <div className="confidence">
          <div className="conf-track"><div className="conf-fill" style={{width:`${proj.confPct}%`}}></div></div>
          <div className="conf-label">{proj.confWord}</div>
        </div>
        <button className="log-period-chip" onClick={onLogPeriod}>
          <span className="dot"></span>Log period start
        </button>
      </div>

      <div className="sec-h">
        <div className="t">What your body may <em>need</em> today</div>
        <div className="a">{proj.planRows.length} of 4 · <button onClick={()=>setTab("guide")}>see plan</button></div>
      </div>
      <div className="guide-list">
        {proj.planRows.length === 0 ? (
          <div className="err-box">
            <b>no recommendations</b>
            Log today's symptoms from the Check-in tab to get phase-aware guidance.
          </div>
        ) : proj.planRows.slice(0,3).map((g,i)=>(
          <div className="guide-row" key={g.id || i}>
            <div className="num">0{i+1}</div>
            <div className="body">
              <div className="kind">{g.kind}</div>
              <div className="title">{g.title}</div>
              <div className="why">{g.why}</div>
            </div>
            <div className="arrow">{I.arrow}</div>
          </div>
        ))}
      </div>

      <div className="sec-h">
        <div className="t"><em>Pulse</em> check</div>
        <div className="a">10 seconds</div>
      </div>
      <div className="pulse">
        <div className="h">
          <div className="q">How are you, really?</div>
          <div className="t">Energy · 1–5</div>
        </div>
        <div className="opts">
          {[1,2,3,4,5].map(n=>(
            <button key={n} className={`opt ${pulse===n?'on':''}`} onClick={()=>onQuickPulse(n)}>{n}</button>
          ))}
        </div>
        <div className="legend"><span>Depleted</span><span>Steady</span><span>Charged</span></div>
      </div>

      <div className="sec-h">
        <div className="t">Where you are in the <em>cycle</em></div>
        <div className="a"><button onClick={()=>setTab("cycle")}>details</button></div>
      </div>
      <Ribbon day={proj.cycleDay} cycleLen={proj.cycleLen}/>
    </div>
  );
}

// ── CHECK-IN ────────────────────────────────────────────────────────────────
// Maps design chips/scales to UnifiedRequest fields.
// Moods are aggregated: if any negative ("Irritable","Anxious","Low","Tender") picked,
// it lifts the matching symptom signal. The design's multi-select doesn't 1:1 map
// to our enums — we collapse to the strongest signal per group.
const FLOW_SCALE = ["None","Light","Moderate","Heavy","Very Heavy"]; // index 0..4

// Cervical mucus types — ordered from dry (infertile) → eggwhite (peak fertile).
// Maps 1:1 to the backend's `cervical_mucus` enum. "unknown" is the skip state.
const MUCUS_ITEMS = [
  { k:"unknown",  label:"—"        },
  { k:"dry",      label:"Dry"      },
  { k:"sticky",   label:"Sticky"   },
  { k:"creamy",   label:"Creamy"   },
  { k:"watery",   label:"Watery"   },
  { k:"eggwhite", label:"Eggwhite" },
];

function CheckinScreen({proj, onSubmit, submitting}){
  const [state,setState] = useState({
    flow: 0,          // 0..4 index into FLOW_SCALE
    mucus: "unknown", // cervical_mucus enum
    energy: 3,        // 1..5
    mood: [],         // multi-select of mood tokens
    body: [],         // multi-select of body tokens
    sleep: null,      // one of sleep tokens
    note: "",
  });

  const MOOD_ITEMS = ["Calm","Focused","Irritable","Anxious","Low","Tender","Sharp","Social"];
  const BODY_ITEMS = ["Cramps","Bloating","Back pain","Breast tenderness","Headache","Nausea","Acne","Joint laxity"];
  const SLEEP_ITEMS = ["<5h","5–6h","7h","8h","9h+","Broken","Vivid dreams"];

  const toggle = (key, val) => setState(s => {
    if (Array.isArray(s[key])) {
      const has = s[key].includes(val);
      return { ...s, [key]: has ? s[key].filter(x=>x!==val) : [...s[key], val] };
    }
    return { ...s, [key]: val };
  });

  // Map UI state → /predict symptom fields.
  const toPayload = () => {
    const energyLevel = ({1:"Very Low",2:"Low",3:"Medium",4:"High",5:"Very High"})[state.energy] || "Medium";
    const has = (arr, x) => arr.includes(x);

    const mood =
      has(state.mood,"Low") ? "Low" :
      has(state.mood,"Tender") ? "Low" :
      has(state.mood,"Focused") || has(state.mood,"Calm") ? "High" :
      "Medium";
    const stress = has(state.mood,"Anxious") ? "High" : has(state.mood,"Irritable") ? "Moderate" : "Low";
    const irritability = has(state.mood,"Irritable") || has(state.mood,"Sharp") ? "Moderate" : "None";
    const moodSwing = (has(state.mood,"Irritable") && has(state.mood,"Low")) ? "Moderate" : "None";

    const cramp =
      has(state.body,"Cramps") ? "Moderate" : "None";
    const bloating =
      has(state.body,"Bloating") ? "Moderate" : "None";
    const breast =
      has(state.body,"Breast tenderness") ? "Moderate" : "None";
    const headache = has(state.body,"Headache") ? 4 : 0;
    const indigestion = has(state.body,"Nausea") ? "Moderate" : "None";

    const sleepQuality =
      state.sleep === "<5h" || state.sleep === "Broken" ? "Poor" :
      state.sleep === "5–6h" ? "Fair" :
      state.sleep == null ? "Good" : "Good";

    return {
      FLOW: FLOW_SCALE[state.flow] || "None",
      cervical_mucus: state.mucus || "unknown",
      ENERGY: energyLevel,
      MOOD: mood,
      STRESS: stress,
      IRRITABILITY: irritability,
      MOOD_SWING: moodSwing,
      CRAMP: cramp,
      BLOATING: bloating,
      BREAST_TENDERNESS: breast,
      HEADACHE: headache,
      INDIGESTION: indigestion,
      SLEEP_QUALITY: sleepQuality,
    };
  };

  return (
    <div className="page fade-in">
      <div className="ci-hero">
        <div className="eyebrow">Daily check-in · {proj.phaseLabel.toLowerCase()} · day {proj.cycleDay}</div>
        <h1>A few quick<br/><em>observations.</em></h1>
        <p>All optional. Whatever you log helps us sharpen the next prediction.</p>
      </div>

      <div className="ci-group first">
        <div className="gh"><div className="gl">Flow</div><div className="gs">today</div></div>
        <div className="scale">
          {FLOW_SCALE.map((x,i)=>(
            <div key={x} className={`s ${state.flow===i?'on':''}`} onClick={()=>toggle('flow', i)}>
              {i===0 ? "—" : "●".repeat(Math.min(i,4))}
            </div>
          ))}
        </div>
        <div className="scale-legend">
          {FLOW_SCALE.map(x=><span key={x}>{x}</span>)}
        </div>
      </div>

      <div className="ci-group">
        <div className="gh"><div className="gl">Cervical mucus</div><div className="gs">optional · fertility signal</div></div>
        <div className="chips">
          {MUCUS_ITEMS.map(m=>(
            <button key={m.k} className={`chip ${state.mucus===m.k?'on':''}`} onClick={()=>toggle('mucus', m.k)}>{m.label}</button>
          ))}
        </div>
      </div>

      <div className="ci-group">
        <div className="gh"><div className="gl">Energy</div><div className="gs">1 to 5</div></div>
        <div className="scale">
          {[1,2,3,4,5].map(n=>(
            <div key={n} className={`s ${state.energy===n?'on':''}`} onClick={()=>toggle('energy', n)}>{n}</div>
          ))}
        </div>
        <div className="scale-legend"><span>Depleted</span><span></span><span>Steady</span><span></span><span>Charged</span></div>
      </div>

      <div className="ci-group">
        <div className="gh"><div className="gl">Mood</div><div className="gs">tap any</div></div>
        <div className="chips">
          {MOOD_ITEMS.map(x=>(
            <button key={x} className={`chip ${state.mood.includes(x)?'on':''}`} onClick={()=>toggle('mood', x)}>{x}</button>
          ))}
        </div>
      </div>

      <div className="ci-group">
        <div className="gh"><div className="gl">Body</div><div className="gs">tap any</div></div>
        <div className="chips">
          {BODY_ITEMS.map(x=>(
            <button key={x} className={`chip ${state.body.includes(x)?'on':''}`} onClick={()=>toggle('body', x)}>{x}</button>
          ))}
        </div>
      </div>

      <div className="ci-group">
        <div className="gh"><div className="gl">Sleep</div><div className="gs">last night</div></div>
        <div className="chips">
          {SLEEP_ITEMS.map(x=>(
            <button key={x} className={`chip ${state.sleep===x?'on':''}`} onClick={()=>toggle('sleep', x)}>{x}</button>
          ))}
        </div>
      </div>

      <div className="ci-group">
        <div className="gh"><div className="gl">Note</div><div className="gs">optional</div></div>
        <textarea
          placeholder="Anything worth remembering — a pattern, a small win, something that hurt."
          value={state.note}
          onChange={e=>setState(s=>({...s, note:e.target.value}))}
          style={{width:'100%',minHeight:72,padding:12,border:'1px solid var(--rule-2)',borderRadius:2,background:'var(--paper)',fontFamily:'var(--serif)',fontSize:16,color:'var(--ink)',resize:'vertical'}} />
      </div>

      <div className="ci-submit">
        <button className="btn-ghost" onClick={()=>onSubmit(null)}>Skip</button>
        <button className="btn-primary" disabled={submitting} onClick={()=>onSubmit(toPayload())}>
          {submitting ? "Saving…" : <>Save check-in {I.arrow}</>}
        </button>
      </div>
    </div>
  );
}

// ── GUIDE ────────────────────────────────────────────────────────────────────
function GuideScreen({proj}){
  const copy = PHASE_COPY[proj.phaseKey] || PHASE_COPY.luteal;
  const [done,setDone] = useState({});
  const toggleDone = (i) => setDone(d => ({...d, [i]: !d[i]}));

  // Build a 4-item plan from proj.planRows, padding from PHASE_COPY guide if
  // the recommender returned fewer.
  const rows = proj.planRows.slice(0,4);

  return (
    <div className="page fade-in" key={proj.phaseKey}>
      <div className="guide-hero">
        <div className="k">Today · {proj.phaseLabel.toLowerCase()} phase</div>
        <h1>A plan, not a<br/><em>prescription.</em></h1>
      </div>
      <div className="plan">
        {rows.length === 0 ? (
          <div className="err-box">
            <b>gathering data</b>
            A few daily check-ins and we'll start surfacing a phase-specific plan here.
          </div>
        ) : rows.map((g,i)=>(
          <div key={g.id || i} className={`plan-item ${done[i]?'done':''}`} onClick={()=>toggleDone(i)}>
            <div className="cat">
              {g.kind}
              <span className="sm">0{i+1}</span>
            </div>
            <div>
              <div className="title">{g.title}</div>
              <div className="sub">{g.why}</div>
            </div>
            <div className="check">{done[i] && <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6.5 5 9l5-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>}</div>
          </div>
        ))}
      </div>

      <div className="sec-h" style={{marginTop:30}}>
        <div className="t">The <em>science</em>, briefly</div>
        <div className="a">what's happening</div>
      </div>
      <div style={{borderTop:'1px solid var(--ink)'}}>
        {copy.body.map((b,i)=>(
          <div key={i} style={{display:'grid',gridTemplateColumns:'28px 1fr',gap:14,padding:'16px 0',borderBottom:'1px solid var(--rule)'}}>
            <div style={{fontFamily:'var(--mono)',fontSize:10,letterSpacing:'.08em',color:'var(--ink-3)'}}>{String(i+1).padStart(2,'0')}</div>
            <div style={{fontFamily:'var(--serif)',fontSize:17,lineHeight:1.35,color:'var(--ink-2)'}}>{b}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── CYCLE ────────────────────────────────────────────────────────────────────
function CycleScreen({proj, periodStarts, onAddPeriod, onDeletePeriod, onLogPeriod}){
  const timing = proj.timing || {};
  const today = new Date();
  const todayIso = today.toISOString().slice(0,10);
  const [addDate,setAddDate] = useState("");

  // Sort period_starts newest-first for display. The cycle engine accepts any
  // order but the user expects to see "most recent" at the top.
  const startsSorted = [...(periodStarts || [])].sort().reverse();

  // Next-period countdown + label
  let nextInDays = Math.max(1, proj.cycleLen - proj.cycleDay + 1);
  let nextDateLabel = null;
  if (timing.predicted_next_period) {
    const np = new Date(timing.predicted_next_period);
    nextInDays = Math.max(0, Math.round((np - today) / 86400000));
    nextDateLabel = np.toLocaleDateString("en-US", { month:"short", day:"numeric" });
  }

  // Fertile window
  let fertileLabel = null;
  let fertileIn = null;
  const fw = timing.fertile_window;
  if (fw && fw.start) {
    const start = new Date(fw.start);
    const diff = Math.round((start - today) / 86400000);
    fertileIn = diff;
    fertileLabel = `${start.toLocaleDateString("en-US",{month:"short",day:"numeric"})}${fw.end ? " – " + new Date(fw.end).toLocaleDateString("en-US",{month:"short",day:"numeric"}) : ""}`;
  }

  return (
    <div className="page fade-in">
      <div className="cycle-head">
        <div className="eyebrow">Your cycle · {proj.cycleLen}d average</div>
        <h1>Cycle <em>intelligence.</em></h1>
        <p>What we know so far. Predictions tighten with every cycle you log.</p>
      </div>

      <Ribbon day={proj.cycleDay} cycleLen={proj.cycleLen}/>

      <div className="pred">
        <div className="cell">
          <div className="k">Next period</div>
          <div className="v">in <em>{nextInDays}</em> days</div>
          <div className="sub">{nextDateLabel || "moderate certainty"}</div>
        </div>
        <div className="cell">
          <div className="k">Fertile window</div>
          <div className="v">
            {fertileIn != null && fertileIn > 0 ? <>in <em>{fertileIn}</em> days</>
              : fertileIn != null && fertileIn <= 0 ? <em>now open</em>
              : <em>gathering</em>}
          </div>
          <div className="sub">{fertileLabel || "5-day window · moderate certainty"}</div>
        </div>
      </div>

      <div className="stats">
        <div className="c"><div className="k">avg cycle</div><div className="v">{proj.cycleLen}d</div><div className="t">estimated</div></div>
        <div className="c"><div className="k">avg period</div><div className="v">5d</div><div className="t">default</div></div>
        <div className="c"><div className="k">mode</div><div className="v" style={{fontSize:14}}>{(proj.raw.phase?.mode || "—").replace(/_/g,' ')}</div><div className="t">fusion</div></div>
      </div>

      <div className="sec-h" style={{marginTop:26}}>
        <div className="t">Period <em>history</em></div>
        <div className="a">
          <button onClick={onLogPeriod}>+ log start</button>
        </div>
      </div>
      <div className="period-list">
        {startsSorted.length === 0 ? (
          <div className="period-empty">No period starts logged yet.</div>
        ) : startsSorted.map(d => {
          const dt = new Date(d + "T00:00:00");
          const label = dt.toLocaleDateString("en-US",{weekday:"short", month:"short", day:"numeric", year:"numeric"});
          const daysAgo = Math.round((today - dt) / 86400000);
          return (
            <div key={d} className="period-row">
              <div className="period-date">{label}</div>
              <div className="period-age">{daysAgo === 0 ? "today" : `${daysAgo}d ago`}</div>
              <button className="period-del" onClick={()=>onDeletePeriod(d)} aria-label={`Delete ${d}`}>×</button>
            </div>
          );
        })}
        <div className="period-add">
          <input type="date" max={todayIso} value={addDate} onChange={e=>setAddDate(e.target.value)} placeholder="YYYY-MM-DD" />
          <button
            className="btn-ghost"
            disabled={!addDate || (periodStarts || []).includes(addDate)}
            onClick={()=>{ onAddPeriod(addDate); setAddDate(""); }}
          >
            Add date
          </button>
        </div>
      </div>

      <div className="sec-h" style={{marginTop:26}}>
        <div className="t">A note on <em>accuracy</em></div>
      </div>
      <p style={{fontFamily:'var(--serif)',fontSize:16,lineHeight:1.45,color:'var(--ink-2)',borderTop:'1px solid var(--rule)',paddingTop:14}}>
        Predictions tighten with every cycle you log. Fertility awareness is useful context, not contraception — we're honest about the difference.
      </p>
    </div>
  );
}

// ── Welcome + Onboarding date-picker ─────────────────────────────────────────
function WelcomeScreen({onBegin}){
  return (
    <div className="wl fade-in">
      <div className="mark"><LotusMark/></div>
      <div className="eyebrow" style={{marginTop:28}}>A cycle-aware wellness companion</div>
      <h1>Your body has <em>rhythm.</em><br/>So should the advice.</h1>
      <p>InBalance turns a 30-second daily check-in into calm, specific guidance for your phase — movement, food, sleep, mood, and fertility awareness.</p>
      <div className="tenets">
        <div className="tenet"><div className="n">01</div><div className="t">Quiet by default. Notifications only when the pattern earns it.</div></div>
        <div className="tenet"><div className="n">02</div><div className="t">Prediction with confidence intervals, not false precision.</div></div>
        <div className="tenet"><div className="n">03</div><div className="t">Body-literate language. No euphemisms, no infantilizing.</div></div>
        <div className="tenet"><div className="n">04</div><div className="t">Your data stays on device unless you say otherwise.</div></div>
      </div>
      <div className="cta">
        <button className="btn-primary" onClick={onBegin}>Begin {I.arrow}</button>
      </div>
    </div>
  );
}

function OnboardingDateScreen({onBack, onSubmit, submitting, error}){
  const [date,setDate] = useState("");
  const today = new Date().toISOString().slice(0,10);
  return (
    <div className="onb fade-in">
      <button className="back" onClick={onBack}>← back</button>
      <div className="eyebrow">Step 1 of 1</div>
      <h1>When did your<br/>last period <em>start?</em></h1>
      <p className="helper">One date is enough to get us going. The more cycles you log over time, the sharper the predictions get.</p>

      <label htmlFor="pd">First day of last period</label>
      <input id="pd" type="date" max={today} value={date} onChange={e=>setDate(e.target.value)} />
      {error && <div className="err">{error}</div>}

      <div className="bottom">
        <button className="btn-primary" disabled={!date || submitting} onClick={()=>onSubmit(date)}>
          {submitting ? "Calibrating…" : <>Continue {I.arrow}</>}
        </button>
      </div>
    </div>
  );
}

// ── App ──────────────────────────────────────────────────────────────────────
function App(){
  const [route,setRoute] = useState("boot");          // boot | welcome | onboard | app | error
  const [tab,setTab]     = useState("today");
  const [proj,setProj]   = useState(null);
  const [err,setErr]     = useState(null);
  const [submitting,setSubmitting] = useState(false);
  const [pulse,setPulse] = useState(null);
  const [sheet,setSheet] = useState(null);            // null | "period"
  const [sheetErr,setSheetErr] = useState(null);
  // periodStarts kept in React state so updates re-render screens that consume it.
  const [periodStarts,setPeriodStarts] = useState(() => LS.get("inbalance.period_starts", []));

  const userId = useMemo(ensureUserId, []);

  // Trust the backend's classifier once we have ≥2 period starts (i.e. at least
  // one full observed cycle). Before that, the calendar-derived phase is the
  // safer source of truth.
  const trustClassifier = periodStarts.length >= 2;

  // Apply accent to match live phase.
  useEffect(()=>{
    if (!proj) return;
    document.documentElement.style.setProperty('--accent', `var(--${proj.phaseKey})`);
  }, [proj]);

  const refresh = useCallback(async (extraSymptoms=null, overrideStarts=null) => {
    setErr(null);
    try {
      const starts = overrideStarts || LS.get("inbalance.period_starts", []);
      if (!starts.length) { setRoute("welcome"); return; }
      const body = buildBasePayload(starts, userId, extraSymptoms || {});
      const resp = await callPredict(body);
      setProj(projectPrediction(resp, { trustClassifier: starts.length >= 2 }));
      setRoute("app");
    } catch (e) {
      console.error(e);
      setErr(String(e.message || e));
      setRoute("error");
    }
  }, [userId]);

  // Initial boot: if we already have a period_start, go straight to app.
  useEffect(()=>{
    if (periodStarts.length) refresh();
    else setRoute("welcome");
  // eslint-disable-next-line
  },[]);

  const onOnboardSubmit = async (date) => {
    setSubmitting(true);
    setErr(null);
    try {
      const starts = [date];
      LS.set("inbalance.period_starts", starts);
      setPeriodStarts(starts);
      // `period_start_logged` means "a period started TODAY" — it forces the
      // fusion engine into period_start_override mode and pins the phase to
      // Menstrual regardless of actual cycle_day. Onboarding asks for the
      // LAST period start, which is usually days/weeks ago, so only set the
      // flag when the user actually picked today.
      const todayIso = new Date().toISOString().slice(0,10);
      const body = buildBasePayload(starts, userId, { period_start_logged: date === todayIso });
      const resp = await callPredict(body);
      setProj(projectPrediction(resp, { trustClassifier: starts.length >= 2 }));
      setRoute("app");
      setTab("today");
    } catch (e) {
      console.error(e);
      setErr(String(e.message || e));
    } finally {
      setSubmitting(false);
    }
  };

  // Merge a new period start into the list (dedup + sort ascending), persist,
  // and refresh the prediction. If the new date is today, fire
  // period_start_logged=true so the fusion engine pins phase to Menstrual.
  const onLogPeriodSave = async ({ date, flow }) => {
    setSubmitting(true);
    setSheetErr(null);
    try {
      const existing = LS.get("inbalance.period_starts", []);
      const merged = Array.from(new Set([...existing, date])).sort();
      LS.set("inbalance.period_starts", merged);
      setPeriodStarts(merged);
      const todayIso = new Date().toISOString().slice(0,10);
      const extra = {
        period_start_logged: date === todayIso,
        FLOW: flow || "None",
      };
      await refresh(extra, merged);
      setSheet(null);
      setTab("today");
    } catch (e) {
      console.error(e);
      setSheetErr(String(e.message || e));
    } finally {
      setSubmitting(false);
    }
  };

  // Quick-add from the Cycle screen's inline input (no flow captured).
  const onAddPeriod = async (date) => {
    const existing = LS.get("inbalance.period_starts", []);
    if (existing.includes(date)) return;
    const merged = Array.from(new Set([...existing, date])).sort();
    LS.set("inbalance.period_starts", merged);
    setPeriodStarts(merged);
    await refresh(null, merged);
  };

  // Remove a period start from history. Requires ≥1 start remaining — the
  // cycle engine can't compute anything without a reference date.
  const onDeletePeriod = async (date) => {
    const existing = LS.get("inbalance.period_starts", []);
    const next = existing.filter(d => d !== date);
    if (next.length === 0) {
      // Dropping the last date → back to welcome to re-onboard.
      LS.del("inbalance.period_starts");
      setPeriodStarts([]);
      setRoute("welcome");
      return;
    }
    LS.set("inbalance.period_starts", next);
    setPeriodStarts(next);
    await refresh(null, next);
  };

  const onCheckinSubmit = async (symptoms) => {
    if (!symptoms) { setTab("today"); return; }
    setSubmitting(true);
    try {
      await refresh(symptoms);
      setTab("today");
    } finally {
      setSubmitting(false);
    }
  };

  const onQuickPulse = async (n) => {
    setPulse(n);
    // Light-touch: send energy only as a quick update.
    const map = {1:"Very Low",2:"Low",3:"Medium",4:"High",5:"Very High"};
    refresh({ ENERGY: map[n] });
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  if (route === "boot") {
    return (
      <div className="app">
        <Brand/>
        <div className="loader"><div className="dot"></div><div>Opening…</div></div>
      </div>
    );
  }
  if (route === "welcome") {
    return (
      <div className="app">
        <WelcomeScreen onBegin={()=>setRoute("onboard")} />
      </div>
    );
  }
  if (route === "onboard") {
    return (
      <div className="app">
        <OnboardingDateScreen
          onBack={()=>setRoute("welcome")}
          onSubmit={onOnboardSubmit}
          submitting={submitting}
          error={err}
        />
      </div>
    );
  }
  if (route === "error") {
    return (
      <div className="app">
        <Brand/>
        <div className="page">
          <div className="err-box">
            <b>couldn't reach the engine</b>
            {err || "unknown error"}
          </div>
          <button className="btn-primary" onClick={refresh}>Try again {I.arrow}</button>
          <div style={{marginTop:24}}>
            <button className="btn-ghost" onClick={()=>{ LS.del("inbalance.period_starts"); setRoute("welcome"); }}>Reset and start over</button>
          </div>
        </div>
      </div>
    );
  }

  const openPeriodSheet = () => { setSheetErr(null); setSheet("period"); };

  // app with tabs
  const screen =
    tab === "today"   ? <TodayScreen proj={proj} setTab={setTab} onQuickPulse={onQuickPulse} pulse={pulse} onLogPeriod={openPeriodSheet}/> :
    tab === "checkin" ? <CheckinScreen proj={proj} onSubmit={onCheckinSubmit} submitting={submitting}/> :
    tab === "guide"   ? <GuideScreen proj={proj}/> :
    tab === "cycle"   ? <CycleScreen proj={proj} periodStarts={periodStarts} onAddPeriod={onAddPeriod} onDeletePeriod={onDeletePeriod} onLogPeriod={openPeriodSheet}/> :
    null;

  return (
    <div className="app">
      <Brand/>
      <div className="screen">{screen}</div>
      <TabBar active={tab} setActive={setTab}/>
      {sheet === "period" && (
        <PeriodLogSheet
          onClose={()=>setSheet(null)}
          onSave={onLogPeriodSave}
          saving={submitting}
          error={sheetErr}
        />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
