// Direction A — "Atlas": data-forward coach, Apple system language
const { useState, useEffect, useMemo } = React;

const CYCLE_LEN_A = 28;
const PHASES_A = [
  { key:"menstrual",  name:"Menstrual",  start:1,  end:5,  hue:8   },
  { key:"follicular", name:"Follicular", start:6,  end:13, hue:155 },
  { key:"ovulatory",  name:"Ovulatory",  start:14, end:16, hue:45  },
  { key:"luteal",     name:"Luteal",     start:17, end:28, hue:280 },
];
const phaseForA = (d)=>PHASES_A.find(p=>d>=p.start&&d<=p.end)||PHASES_A[0];

const COACH_A = {
  menstrual: { title:"Rest is the plan", body:"Energy is low. Iron is dropping. Choose warmth and fewer decisions.", do:["Walk 20 min outside","Iron + vitamin C at lunch","In bed by 10:30"], why:"Day 2 is typically your heaviest — plan accordingly." },
  follicular:{ title:"Push a little harder", body:"Estrogen is climbing. Strength, learning and recovery all land better now.", do:["Strength session, add weight","Protein-forward breakfast","Morning sunlight"], why:"You trained well in this window last cycle." },
  ovulatory: { title:"Protect your attention", body:"Output is high, but volatile. Use the energy on what matters.", do:["HIIT, 25 min capped","Hydrate + salt","Non-negotiable wind-down"], why:"Peak energy — past cycles support this." },
  luteal:    { title:"Slow the pace on purpose", body:"Progesterone dominant. Hungrier, warmer, stress lands harder.", do:["Zone 2, 30 min","Complex carbs at dinner","Bedtime 20 min earlier"], why:"Mood dips ~2 days before your period — pre-empt it." },
};

const SFProStack = `-apple-system, "SF Pro Text", "SF Pro Display", "Helvetica Neue", system-ui, sans-serif`;

// Small sparkline hormone curves
function HormoneCurves({day, height=60, width=320}){
  // three curves relative to 28-day cycle
  const est = (t)=> 0.2 + 0.8 * Math.max(0, Math.sin(Math.PI*(t/28)*1.6 - 0.2))**1.1;   // peaks mid
  const prg = (t)=> 0.1 + 0.85 * Math.max(0, Math.sin(Math.PI*((t-6)/28)*1.2))**2;      // peaks late
  const lh  = (t)=> { const x=(t-13.5)/1.4; return 0.1 + 0.9*Math.exp(-x*x); };
  const pts = (fn)=>Array.from({length:29},(_,i)=>i).map(i=>[i/28*width, height - fn(i)*height*0.85 - 4]);
  const path = (pp)=> pp.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
  const x = (day/28)*width;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} style={{display:'block'}}>
      <defs>
        <linearGradient id="estG" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="oklch(70% 0.15 155)" stopOpacity="0.35"/>
          <stop offset="1" stopColor="oklch(70% 0.15 155)" stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path d={path(pts(est))+` L ${width} ${height} L 0 ${height} Z`} fill="url(#estG)"/>
      <path d={path(pts(est))} stroke="oklch(60% 0.16 155)" strokeWidth="1.6" fill="none"/>
      <path d={path(pts(prg))} stroke="oklch(55% 0.18 300)" strokeWidth="1.6" fill="none"/>
      <path d={path(pts(lh))}  stroke="oklch(62% 0.18 45)"  strokeWidth="1.6" fill="none" strokeDasharray="2 2"/>
      <line x1={x} x2={x} y1="0" y2={height} stroke="currentColor" strokeWidth="1"/>
      <circle cx={x} cy={height - est(day)*height*0.85 - 4} r="3" fill="oklch(60% 0.16 155)"/>
    </svg>
  );
}

// Month calendar with phase bands + prediction shading
function MonthCalendar({day}){
  const today = 18; // Apr 18
  const daysInMonth = 30;
  const startWeekday = 2; // Apr 1 = Wed in our mock
  const cells = [];
  for (let i=0;i<startWeekday;i++) cells.push(null);
  for (let d=1; d<=daysInMonth; d++) cells.push(d);
  // cycle day mapping: today = cycle day `day`
  // period expected starts at cycle day 29 - i.e. (today + (28-day+1))
  const nextPeriodStart = today + (28 - day + 1);
  return (
    <div className="atlas-cal">
      <div className="atlas-cal-h">
        {["M","T","W","T","F","S","S"].map((d,i)=>(<div key={i} className="atlas-cal-d">{d}</div>))}
      </div>
      <div className="atlas-cal-grid">
        {cells.map((d,i)=>{
          if (!d) return <div key={i}/>;
          const cycleDayForDate = ((d - today) + day);
          const ph = (cycleDayForDate>=1 && cycleDayForDate<=28) ? phaseForA(cycleDayForDate) : null;
          const isToday = d===today;
          const isPredictedPeriod = d>=nextPeriodStart && d<=Math.min(daysInMonth, nextPeriodStart+4);
          const isFertile = cycleDayForDate>=12 && cycleDayForDate<=16;
          return (
            <div key={i} className={`atlas-cell ${isToday?'is-today':''}`}>
              {ph && <div className="atlas-cell-band" style={{background:`oklch(78% 0.10 ${ph.hue} / .35)`}}/>}
              {isFertile && <div className="atlas-cell-fert"/>}
              {isPredictedPeriod && <div className="atlas-cell-pred"/>}
              <span>{d}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Ring on Today showing cycle day position
function CycleRing({day, size=160}){
  const ph = phaseForA(day);
  const r = size/2 - 10;
  const c = size/2;
  const pct = day/28;
  const circ = 2*Math.PI*r;
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
      <circle cx={c} cy={c} r={r} fill="none" stroke="currentColor" strokeOpacity=".1" strokeWidth="8"/>
      <circle cx={c} cy={c} r={r} fill="none"
        stroke={`oklch(62% 0.14 ${ph.hue})`} strokeWidth="8"
        strokeDasharray={`${circ*pct} ${circ}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${c} ${c})`}/>
      <text x={c} y={c-4} textAnchor="middle" fontFamily={SFProStack} fontSize="40" fontWeight="600" fill="currentColor">{day}</text>
      <text x={c} y={c+18} textAnchor="middle" fontFamily={SFProStack} fontSize="10" fontWeight="500" letterSpacing="1.2" fill="currentColor" opacity=".6">DAY / 28</text>
    </svg>
  );
}

// Symptom heatmap
function Heatmap({rows=6, cols=28}){
  // pseudo-random mood intensity; luteal heavier
  const data = Array.from({length:rows}, (_,r)=>
    Array.from({length:cols}, (_,c)=>{
      const base = c>16 ? 0.55 : c<6 ? 0.65 : 0.2;
      return Math.max(0, Math.min(1, base + (Math.sin(r*3+c)+1)/4 - 0.2));
    })
  );
  return (
    <div className="atlas-heat">
      {data.map((row,ri)=>(
        <div key={ri} className="atlas-heat-row">
          {row.map((v,ci)=>(
            <div key={ci} className="atlas-heat-cell" style={{background:`oklch(62% ${0.02 + v*0.14} 300 / ${0.18 + v*0.75})`}}/>
          ))}
        </div>
      ))}
    </div>
  );
}

function AtlasTodayScreen({day, setTab}){
  const ph = phaseForA(day);
  const c = COACH_A[ph.key];
  return (
    <div className="atlas-page">
      <div className="atlas-hero">
        <div className="atlas-hero-l">
          <div className="atlas-eyebrow" style={{color:`oklch(62% 0.14 ${ph.hue})`}}>{ph.name} phase</div>
          <h1 className="atlas-h1">{c.title}.</h1>
          <p className="atlas-lede">{c.body}</p>
        </div>
        <div className="atlas-ring" style={{color:`oklch(62% 0.14 ${ph.hue})`}}>
          <CycleRing day={day}/>
        </div>
      </div>

      <div className="atlas-quicklog">
        {["Started period","Spotting","Cramps","Mood","+ More"].map((x,i)=>(
          <button key={i} className="atlas-ql">{x}</button>
        ))}
      </div>

      <div className="atlas-sec">
        <div className="atlas-sec-h">
          <span>Today's plan</span>
          <button onClick={()=>setTab('guide')} className="atlas-link">See full plan</button>
        </div>
        <div className="atlas-plan">
          {c.do.map((t,i)=>(
            <div key={i} className="atlas-plan-row">
              <div className="atlas-plan-ix">{String(i+1).padStart(2,'0')}</div>
              <div className="atlas-plan-t">{t}</div>
              <div className="atlas-plan-c">○</div>
            </div>
          ))}
          <div className="atlas-plan-why">Why: {c.why}</div>
        </div>
      </div>

      <div className="atlas-sec">
        <div className="atlas-sec-h">
          <span>Hormones, this cycle</span>
          <span className="atlas-sub">Day {day} / 28</span>
        </div>
        <div className="atlas-card">
          <HormoneCurves day={day}/>
          <div className="atlas-legend">
            <span><i style={{background:"oklch(60% 0.16 155)"}}/>Estrogen</span>
            <span><i style={{background:"oklch(55% 0.18 300)"}}/>Progesterone</span>
            <span><i style={{background:"oklch(62% 0.18 45)", borderStyle:"dashed"}}/>LH</span>
          </div>
        </div>
      </div>

      <div className="atlas-sec">
        <div className="atlas-sec-h">
          <span>Stats</span>
          <span className="atlas-sub">last 6 cycles</span>
        </div>
        <div className="atlas-stats">
          <div><div className="atlas-stat-v">28<small>d</small></div><div className="atlas-stat-k">avg cycle</div></div>
          <div><div className="atlas-stat-v">5<small>d</small></div><div className="atlas-stat-k">avg period</div></div>
          <div><div className="atlas-stat-v">±1.4<small>d</small></div><div className="atlas-stat-k">variance</div></div>
          <div><div className="atlas-stat-v">86<small>%</small></div><div className="atlas-stat-k">confidence</div></div>
        </div>
      </div>
    </div>
  );
}

function AtlasCycleScreen({day}){
  return (
    <div className="atlas-page">
      <div className="atlas-hero-simple">
        <div className="atlas-eyebrow">Your cycle · April</div>
        <h1 className="atlas-h1">Cycle map.</h1>
      </div>
      <div className="atlas-card atlas-card-pad">
        <MonthCalendar day={day}/>
        <div className="atlas-cal-legend">
          <span><i className="dot" style={{background:"oklch(60% 0.18 8)"}}/>Period</span>
          <span><i className="dot" style={{background:"oklch(70% 0.10 45)"}}/>Fertile</span>
          <span><i className="dot" style={{background:"oklch(70% 0.10 280)"}}/>Luteal</span>
          <span><i className="dot ring"/>Predicted</span>
        </div>
      </div>

      <div className="atlas-sec">
        <div className="atlas-sec-h">
          <span>Symptom heatmap</span>
          <span className="atlas-sub">mood · last 6 cycles</span>
        </div>
        <div className="atlas-card atlas-card-pad">
          <Heatmap/>
          <div className="atlas-heat-axis"><span>Day 1</span><span>14</span><span>28</span></div>
        </div>
      </div>

      <div className="atlas-sec">
        <div className="atlas-sec-h"><span>Patterns</span><span className="atlas-sub">auto-detected</span></div>
        <div className="atlas-card">
          {[
            ["Mood dips 2 days before your period, not on day 1.","3/3"],
            ["Luteal sleep is ~40 min shorter than follicular.","4/4"],
            ["Energy clusters on days 10–13. Plan intensity there.","5/6"],
            ["Heaviest flow is day 2, not day 1.","6/6"],
          ].map(([t,d],i)=>(
            <div key={i} className="atlas-pat">
              <div>{t}</div><div className="atlas-sub">{d}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AtlasLogScreen({day}){
  const [st,setSt] = useState({flow:null, energy:3, mood:[], body:[], sleep:null});
  const tog = (k,v)=>setSt(s=>Array.isArray(s[k])
    ? {...s,[k]:s[k].includes(v)?s[k].filter(x=>x!==v):[...s[k],v]}
    : {...s,[k]:v});
  const C = ({k, label, opts, multi=false, scale=false}) => (
    <div className="atlas-log-g">
      <div className="atlas-log-h"><span>{label}</span><span className="atlas-sub">{multi?'any':'pick one'}</span></div>
      <div className={scale?"atlas-log-scale":"atlas-log-chips"}>
        {opts.map(o=>(
          <button key={o} onClick={()=>tog(k,o)}
            className={`atlas-chip ${(multi?st[k].includes(o):st[k]===o)?'on':''}`}>{o}</button>
        ))}
      </div>
    </div>
  );
  return (
    <div className="atlas-page">
      <div className="atlas-hero-simple">
        <div className="atlas-eyebrow">Daily log · {new Date().toLocaleDateString('en',{month:'short',day:'numeric'})}</div>
        <h1 className="atlas-h1">How was today?</h1>
        <p className="atlas-lede">Takes 30 seconds. You can add more later.</p>
      </div>
      <C k="flow" label="Flow" opts={["None","Spotting","Light","Medium","Heavy"]} scale/>
      <C k="energy" label="Energy · 1–5" opts={[1,2,3,4,5]} scale/>
      <C k="mood" label="Mood" multi opts={["Calm","Focused","Irritable","Anxious","Low","Tender","Sharp","Social"]}/>
      <C k="body" label="Body" multi opts={["Cramps","Bloating","Back pain","Tender breasts","Headache","Nausea","Acne","Joint laxity"]}/>
      <C k="sleep" label="Sleep" opts={["<5h","5–6h","7h","8h","9h+","Broken"]}/>
      <div className="atlas-log-submit">
        <button className="atlas-btn-g">Skip today</button>
        <button className="atlas-btn">Save log →</button>
      </div>
    </div>
  );
}

function AtlasGuideScreen({day}){
  const ph = phaseForA(day);
  const c = COACH_A[ph.key];
  return (
    <div className="atlas-page">
      <div className="atlas-hero-simple">
        <div className="atlas-eyebrow" style={{color:`oklch(62% 0.14 ${ph.hue})`}}>{ph.name} plan</div>
        <h1 className="atlas-h1">{c.title}.</h1>
        <p className="atlas-lede">{c.body}</p>
      </div>
      <div className="atlas-card">
        {c.do.map((t,i)=>(
          <div key={i} className="atlas-plan-row big">
            <div className="atlas-plan-ix">{String(i+1).padStart(2,'0')}</div>
            <div className="atlas-plan-t">{t}</div>
            <div className="atlas-plan-c">○</div>
          </div>
        ))}
      </div>

      <div className="atlas-sec">
        <div className="atlas-sec-h"><span>Library · {ph.name}</span><span className="atlas-sub">curated</span></div>
        <div className="atlas-essays">
          {[
            ["Phase primer","Why luteal feels heavier — and why that's useful","5 min"],
            ["Nutrition","Magnesium before bed: small input, big return","4 min"],
            ["PCOS","Training when your cycles are irregular","7 min"],
            ["Peri","What perimenopause actually does to hormones","6 min"],
          ].map(([c,t,m],i)=>(
            <div key={i} className="atlas-essay">
              <div className="atlas-essay-thumb"/>
              <div><div className="atlas-essay-c">{c}</div><div className="atlas-essay-t">{t}</div></div>
              <div className="atlas-sub">{m}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const AtlasTabs = [
  { k:"today", label:"Today",  icon:(<path d="M12 3v2m0 14v2M3 12h2m14 0h2M5.6 5.6l1.4 1.4m10 10 1.4 1.4M5.6 18.4l1.4-1.4m10-10 1.4-1.4"/>) },
  { k:"cycle", label:"Cycle",  icon:(<><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 9h18M8 3v4m8-4v4"/></>) },
  { k:"log",   label:"Log",    icon:(<><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/></>) },
  { k:"guide", label:"Guide",  icon:(<path d="M4 5h12a4 4 0 0 1 4 4v10H8a4 4 0 0 1-4-4Zm0 0v10a4 4 0 0 0 4 4"/>) },
];

function AtlasApp({day}){
  const [tab,setTab] = useState('today');
  const ph = phaseForA(day);
  return (
    <div className="atlas" style={{'--ph-hue':ph.hue, '--ph':`oklch(62% 0.14 ${ph.hue})`}}>
      <div className="atlas-status">
        <span>9:41</span>
        <span className="atlas-status-r">
          <svg width="16" height="10" viewBox="0 0 16 10"><path d="M1 9h2V6H1zm4 0h2V4H5zm4 0h2V2H9zm4 0h2V0h-2z" fill="currentColor"/></svg>
          <svg width="24" height="10" viewBox="0 0 24 10"><rect x=".5" y=".5" width="20" height="9" rx="2" stroke="currentColor" fill="none"/><rect x="2" y="2" width="16" height="6" rx="1" fill="currentColor"/></svg>
        </span>
      </div>
      <div className="atlas-brand">
        <div className="atlas-brand-l"><strong>inBalance</strong><span className="atlas-sub">Atlas</span></div>
        <button className="atlas-avatar">A</button>
      </div>
      <div className="atlas-screen">
        {tab==='today' && <AtlasTodayScreen day={day} setTab={setTab}/>}
        {tab==='cycle' && <AtlasCycleScreen day={day}/>}
        {tab==='log'   && <AtlasLogScreen day={day}/>}
        {tab==='guide' && <AtlasGuideScreen day={day}/>}
      </div>
      <div className="atlas-tabs">
        {AtlasTabs.map(t=>(
          <button key={t.k} className={`atlas-tab ${tab===t.k?'on':''}`} onClick={()=>setTab(t.k)}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">{t.icon}</svg>
            <span>{t.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

window.AtlasApp = AtlasApp;
