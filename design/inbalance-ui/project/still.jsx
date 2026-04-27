// Direction B — "Still": quiet warm companion, visionOS-soft
const { useState, useEffect } = React;

const CYCLE_LEN_B = 28;
const PHASES_B = [
  { key:"menstrual",  name:"menstrual",  start:1,  end:5,  hue:8   },
  { key:"follicular", name:"follicular", start:6,  end:13, hue:155 },
  { key:"ovulatory",  name:"ovulatory",  start:14, end:16, hue:45  },
  { key:"luteal",     name:"luteal",     start:17, end:28, hue:280 },
];
const phaseForB = (d)=>PHASES_B.find(p=>d>=p.start&&d<=p.end)||PHASES_B[0];

const STILL_COPY = {
  menstrual:  { line:"Rest is the work today.",          one:"A short walk. Warmth. Less." },
  follicular: { line:"You can ask more of yourself.",    one:"Move — a little harder than yesterday." },
  ovulatory:  { line:"Energy is here. Spend it wisely.", one:"One thing that matters, done well." },
  luteal:     { line:"Slow, on purpose.",                one:"Finish early. Eat warm. Sleep sooner." },
};

function BreatheOrb({day, size=220}){
  const ph = phaseForB(day);
  const pct = day/28;
  const c = size/2;
  return (
    <div className="still-orb-wrap" style={{width:size, height:size}}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} className="still-orb">
        <defs>
          <radialGradient id={`orbG-${ph.key}`} cx="0.35" cy="0.35" r="0.75">
            <stop offset="0"   stopColor={`oklch(94% 0.05 ${ph.hue})`} stopOpacity="0.95"/>
            <stop offset="0.55" stopColor={`oklch(80% 0.10 ${ph.hue})`} stopOpacity="0.6"/>
            <stop offset="1"   stopColor={`oklch(60% 0.14 ${ph.hue})`} stopOpacity="0"/>
          </radialGradient>
        </defs>
        <circle cx={c} cy={c} r={c-6} fill="none" stroke="currentColor" strokeOpacity=".07"/>
        <circle cx={c} cy={c} r={c-22} fill={`url(#orbG-${ph.key})`} className="orb-breathe"/>
        <g transform={`rotate(${pct*360 - 90} ${c} ${c})`}>
          <circle cx={size-16} cy={c} r="4" fill="currentColor"/>
        </g>
      </svg>
    </div>
  );
}

function QuietCal({day}){
  const today = 18, total = 30, startWd = 2;
  const cells = [...Array(startWd).fill(null), ...Array.from({length:total},(_,i)=>i+1)];
  return (
    <div className="still-cal">
      <div className="still-cal-h">{["M","T","W","T","F","S","S"].map((d,i)=>(<span key={i}>{d}</span>))}</div>
      <div className="still-cal-g">
        {cells.map((d,i)=>{
          if (!d) return <div key={i}/>;
          const cd = (d - today) + day;
          const ph = (cd>=1 && cd<=28) ? phaseForB(cd) : null;
          const isToday = d===today;
          const fertile = cd>=12 && cd<=16;
          const predPeriod = cd>=29 && cd<=33;
          return (
            <div key={i} className={`still-cell ${isToday?'today':''}`}>
              <span>{d}</span>
              {ph && <div className="still-dot" style={{background:`oklch(70% 0.09 ${ph.hue})`, opacity: fertile?1:0.5}}/>}
              {predPeriod && <div className="still-dot ring"/>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StillToday({day, setTab}){
  const ph = phaseForB(day);
  const c = STILL_COPY[ph.key];
  return (
    <div className="still-page">
      <div className="still-date"><span>Saturday</span><span>April 18</span></div>
      <div className="still-hero" style={{color:`oklch(58% 0.14 ${ph.hue})`}}>
        <BreatheOrb day={day}/>
        <div className="still-phase">{ph.name}</div>
        <div className="still-dayn">day {day} of 28</div>
      </div>
      <div className="still-lede"><p className="still-line">{c.line}</p></div>
      <div className="still-one">
        <div className="still-one-k">one thing</div>
        <div className="still-one-t">{c.one}</div>
        <button className="still-one-b">Mark done ○</button>
      </div>
      <button className="still-log-quick" onClick={()=>setTab('log')}>+ Log today</button>
      <div className="still-foot">
        <button onClick={()=>setTab('cycle')}>Cycle</button>
        <span>·</span>
        <button onClick={()=>setTab('read')}>Read</button>
      </div>
    </div>
  );
}

function StillCycle({day}){
  const ph = phaseForB(day);
  return (
    <div className="still-page">
      <div className="still-eyeb">your cycle</div>
      <h2 className="still-h2">April</h2>
      <QuietCal day={day}/>
      <div className="still-legend">
        <span><i style={{background:`oklch(70% 0.09 8)`}}/>period</span>
        <span><i style={{background:`oklch(70% 0.09 45)`}}/>fertile</span>
        <span><i className="ring"/>predicted</span>
      </div>
      <div className="still-note">
        <div className="still-eyeb">pattern</div>
        <p>Your mood tends to dip <em>two days</em> before your period — not on day one. Noted across three of your last three cycles.</p>
      </div>
      <div className="still-note">
        <div className="still-eyeb">hormones</div>
        <p>You're in the <em>{ph.name}</em> phase. Progesterone is doing most of the talking right now — expect warmth, appetite, and a lower threshold for stress.</p>
      </div>
    </div>
  );
}

function StillLog({day, setTab}){
  const [st,setSt] = useState({flow:null, mood:[], body:[], energy:null});
  const tog=(k,v)=>setSt(s=>Array.isArray(s[k])
    ? {...s,[k]:s[k].includes(v)?s[k].filter(x=>x!==v):[...s[k],v]}
    : {...s,[k]:v});
  return (
    <div className="still-page">
      <div className="still-eyeb">today</div>
      <h2 className="still-h2">How are you?</h2>
      <p className="still-sub">Tap what fits. Skip what doesn't.</p>

      <div className="still-g">
        <div className="still-g-l">flow</div>
        <div className="still-chips">{["none","spotting","light","medium","heavy"].map(x=>
          <button key={x} className={`still-chip ${st.flow===x?'on':''}`} onClick={()=>tog('flow',x)}>{x}</button>)}</div>
      </div>
      <div className="still-g">
        <div className="still-g-l">energy</div>
        <div className="still-chips">{["low","steady","high"].map(x=>
          <button key={x} className={`still-chip ${st.energy===x?'on':''}`} onClick={()=>tog('energy',x)}>{x}</button>)}</div>
      </div>
      <div className="still-g">
        <div className="still-g-l">mood</div>
        <div className="still-chips">{["calm","focused","irritable","anxious","low","tender"].map(x=>
          <button key={x} className={`still-chip ${st.mood.includes(x)?'on':''}`} onClick={()=>tog('mood',x)}>{x}</button>)}</div>
      </div>
      <div className="still-g">
        <div className="still-g-l">body</div>
        <div className="still-chips">{["cramps","bloating","back","breast","headache","acne"].map(x=>
          <button key={x} className={`still-chip ${st.body.includes(x)?'on':''}`} onClick={()=>tog('body',x)}>{x}</button>)}</div>
      </div>
      <button className="still-save" onClick={()=>setTab('today')}>Save</button>
    </div>
  );
}

function StillRead({day}){
  const ph = phaseForB(day);
  return (
    <div className="still-page">
      <div className="still-eyeb">for {ph.name}</div>
      <h2 className="still-h2">Read</h2>
      {[
        ["Why luteal feels heavier","5 min"],
        ["Training when cycles are irregular","7 min"],
        ["Magnesium, briefly","3 min"],
        ["What perimenopause actually does","8 min"],
      ].map(([t,m],i)=>(
        <a key={i} className="still-art">
          <div className="still-art-t">{t}</div>
          <div className="still-sub">{m}</div>
        </a>
      ))}
    </div>
  );
}

function StillApp({day}){
  const [tab,setTab] = useState('today');
  const ph = phaseForB(day);
  return (
    <div className="still" style={{'--ph-hue':ph.hue, '--ph':`oklch(60% 0.12 ${ph.hue})`}}>
      <div className="still-status">
        <span>9:41</span>
        <span className="still-status-r">
          <svg width="14" height="10" viewBox="0 0 14 10"><path d="M1 9h2V6H1zm4 0h2V4H5zm4 0h2V2H9zm4 0h1V0h-1z" fill="currentColor"/></svg>
          <svg width="22" height="10" viewBox="0 0 22 10"><rect x=".5" y=".5" width="19" height="9" rx="2" stroke="currentColor" fill="none"/><rect x="2" y="2" width="15" height="6" rx="1" fill="currentColor"/></svg>
        </span>
      </div>
      <div className="still-brand">inBalance <span>· still</span></div>
      <div className="still-screen">
        {tab==='today' && <StillToday day={day} setTab={setTab}/>}
        {tab==='cycle' && <StillCycle day={day}/>}
        {tab==='log'   && <StillLog day={day} setTab={setTab}/>}
        {tab==='read'  && <StillRead day={day}/>}
      </div>
      <div className="still-tabs">
        {[
          ['today','Today'],['cycle','Cycle'],['log','Log'],['read','Read']
        ].map(([k,l])=>(
          <button key={k} className={`still-tab ${tab===k?'on':''}`} onClick={()=>setTab(k)}>{l}</button>
        ))}
      </div>
    </div>
  );
}

window.StillApp = StillApp;
