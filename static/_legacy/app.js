/* ── User identity ───────────────────────────────────────────────────────── */
function getUsername() { return localStorage.getItem('ib_username') || ''; }

function setUsername(name) {
  localStorage.setItem('ib_username', name.trim());
  refreshBanner();
}

function refreshBanner() {
  var name = getUsername();
  document.getElementById('user-greeting').textContent = name ? 'Hi, ' + name : 'Set your name →';
}

function openUsernameModal() {
  var modal = document.getElementById('username-modal');
  document.getElementById('username-input').value = getUsername();
  modal.classList.add('open');
  setTimeout(function() { document.getElementById('username-input').focus(); }, 50);
}

function confirmUsername() {
  var val = document.getElementById('username-input').value.trim();
  if (!val) return;
  setUsername(val);
  document.getElementById('username-modal').classList.remove('open');
}

// Allow Enter key in the input
document.getElementById('username-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') confirmUsername();
});

// Show modal on first visit
if (!getUsername()) openUsernameModal();
refreshBanner();

/* ── State ───────────────────────────────────────────────────────────────── */
var state = {
  phase:     null,
  vitals:    { ENERGY: null, MOOD: null, STRESS: null },
  results:   null,
  fertility: null,
  timing:    null,
  proximity: null,
};

var PHASE_MESSAGES = {
  MEN: ["Rest is productive today.", "Your body is renewing — be gentle.", "Flow with the slowdown, not against it."],
  FOL: ["Energy is rising. Lean in.", "A great time to start something new.", "Your strength is building — use it."],
  OVU: ["You're at your peak. Shine.", "High energy, high connection — enjoy it.", "Your most radiant phase is here."],
  LUT: ["Wind down with intention.", "Your body is preparing — nourish it.", "Rest well, you've earned it."],
  LL:  ["Tender care is needed today.", "Honor what your body is signaling.", "This phase asks for softness."],
};

var DOMAIN_ICONS = { exercise:'🏃', nutrition:'🥗', sleep:'🌙', mental:'🧠' };
var DOMAIN_ORDER = ['exercise','nutrition','sleep','mental'];

/* ── Tab routing ─────────────────────────────────────────────────────────── */
function switchTab(tab) {
  document.querySelectorAll('.screen').forEach(function(s) {
    s.classList.remove('active');
  });
  document.querySelectorAll('.nav-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  var screen = document.getElementById('screen-' + tab);
  if (screen) screen.classList.add('active');
}

document.querySelectorAll('.nav-btn').forEach(function(btn) {
  btn.addEventListener('click', function() { switchTab(btn.dataset.tab); });
});

/* ── Pill selectors ──────────────────────────────────────────────────────── */
document.querySelectorAll('.pill-selector').forEach(function(group) {
  group.querySelectorAll('.pill').forEach(function(pill) {
    pill.addEventListener('click', function() {
      group.querySelectorAll('.pill').forEach(function(p) { p.classList.remove('selected'); });
      pill.classList.add('selected');
    });
  });
});

// Default selections
function defaultPill(name, value) {
  var group = document.querySelector('.pill-selector[data-name="' + name + '"]');
  if (!group) return;
  var target = group.querySelector('[data-value="' + value + '"]');
  if (target) target.classList.add('selected');
}

defaultPill('CRAMP', 'None');
defaultPill('FLOW', 'None');
defaultPill('BLOATING', 'None');
defaultPill('MOOD_SWING', 'None');
defaultPill('IRRITABILITY', 'None');
defaultPill('BRAIN_FOG', 'None');
defaultPill('CRAVINGS', 'None');
defaultPill('SLEEP_QUALITY', 'Good');
defaultPill('cervical_mucus', 'unknown');
defaultPill('BREAST_TENDERNESS', 'None');
defaultPill('INDIGESTION', 'None');

/* ── Emoji vitals ────────────────────────────────────────────────────────── */
document.querySelectorAll('.emoji-scale').forEach(function(scale) {
  scale.querySelectorAll('.emoji-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      scale.querySelectorAll('.emoji-btn').forEach(function(b) { b.classList.remove('selected'); });
      btn.classList.add('selected');
      var signal = scale.dataset.signal;
      state.vitals[signal] = btn.dataset.value;
      checkVitalsComplete();
    });
  });
});

function checkVitalsComplete() {
  var allSet = state.vitals.ENERGY && state.vitals.MOOD && state.vitals.STRESS;
  document.getElementById('vitals-status').textContent = allSet ? 'Logged ✓' : 'Not logged today';
}

/* ── Extras toggle ───────────────────────────────────────────────────────── */
document.getElementById('extras-toggle').addEventListener('click', function() {
  var section = document.getElementById('extras-section');
  var open = section.classList.toggle('open');
  this.textContent = open ? '− Hide extra symptoms' : '+ More symptoms (breast tenderness, indigestion)';
});

/* ── Guide filter tabs ───────────────────────────────────────────────────── */
document.querySelectorAll('.filter-tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.filter-tab').forEach(function(t) { t.classList.remove('active'); });
    tab.classList.add('active');
    filterGuide(tab.dataset.domain);
  });
});

function filterGuide(domain) {
  var content = document.getElementById('guide-content');
  if (!content) return;
  content.querySelectorAll('.guide-domain').forEach(function(el) {
    el.style.display = (domain === 'all' || el.dataset.domain === domain) ? '' : 'none';
  });
}

/* ── Pill value helpers ──────────────────────────────────────────────────── */
function getPill(name) {
  var sel = document.querySelector('.pill-selector[data-name="' + name + '"] .pill.selected');
  return sel ? sel.dataset.value : null;
}

function getPillOrDefault(name, def) {
  return getPill(name) || def;
}

/* ── Build payload ───────────────────────────────────────────────────────── */
function buildPayload() {
  var raw = document.getElementById('period-dates').value.trim();
  var periodStarts = raw ? raw.split('\n').map(function(l){return l.trim();}).filter(function(l){return /^\d{4}-\d{2}-\d{2}$/.test(l);}) : [];

  return {
    user_id:             getUsername() || undefined,
    period_starts:       periodStarts,
    period_start_logged: document.getElementById('period-today').checked,
    cervical_mucus:      getPillOrDefault('cervical_mucus', 'unknown'),
    appetite:            0,
    exercise_level:      0,
    ENERGY:              state.vitals.ENERGY  || 'Medium',
    STRESS:              state.vitals.STRESS  || 'Low',
    MOOD:                state.vitals.MOOD    || 'Medium',
    CRAMP:               getPillOrDefault('CRAMP', 'None'),
    HEADACHE:            parseInt(document.getElementById('headache').value, 10),
    BLOATING:            getPillOrDefault('BLOATING', 'None'),
    SLEEP_QUALITY:       getPillOrDefault('SLEEP_QUALITY', 'Good'),
    IRRITABILITY:        getPillOrDefault('IRRITABILITY', 'None'),
    MOOD_SWING:          getPillOrDefault('MOOD_SWING', 'None'),
    CRAVINGS:            getPillOrDefault('CRAVINGS', 'None'),
    FLOW:                getPillOrDefault('FLOW', 'None'),
    BREAST_TENDERNESS:   getPillOrDefault('BREAST_TENDERNESS', 'None'),
    INDIGESTION:         getPillOrDefault('INDIGESTION', 'None'),
    BRAIN_FOG:           getPillOrDefault('BRAIN_FOG', 'None'),
    top_n: 3,
  };
}

/* ── Submit ──────────────────────────────────────────────────────────────── */
document.getElementById('log-form').addEventListener('submit', function(e) {
  e.preventDefault();
  runEngine();
});

function runEngine() {
  setLoading(true);
  clearError();

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(buildPayload()),
  })
  .then(function(res) {
    if (!res.ok) return res.json().catch(function(){return{};}).then(function(e){ throw new Error(e.detail || 'Error ' + res.status); });
    return res.json();
  })
  .then(function(data) {
    state.results   = data.recommendations;
    state.phase     = data.phase;
    state.fertility = data.fertility;
    state.timing    = data.timing;
    state.proximity = data.period_proximity;

    applyPhaseTheme(data.phase);
    updateHero(data.phase, data.timing);
    renderProximitySignal(data.period_proximity, data.phase);
    renderTodayPicks(data.recommendations);
    renderGuide(data.recommendations, data.phase);
    renderCycleTab(data.phase, data.fertility, data.timing);

    // Switch to today tab to show results
    switchTab('today');
  })
  .catch(function(err) { showError(err.message); })
  .finally(function() { setLoading(false); });
}

/* ── Phase theme ─────────────────────────────────────────────────────────── */
function applyPhaseTheme(phase) {
  if (!phase || !phase.code) return;
  document.querySelector('.app').dataset.phase = phase.code;
}

/* ── Hero update ─────────────────────────────────────────────────────────── */
function updateHero(phase, timing) {
  if (!phase) return;

  var pill = document.getElementById('hero-phase-pill');
  var day  = document.getElementById('hero-day');
  var msg  = document.getElementById('hero-message');

  pill.textContent = phase.label || phase.code;

  var dayStr = '';
  if (phase.cycle_day)   dayStr += 'Day ' + phase.cycle_day;
  if (phase.cycle_length) dayStr += (dayStr ? ' · ' : '') + Math.round(phase.cycle_length) + '-day cycle';
  day.textContent = dayStr;

  var messages = PHASE_MESSAGES[phase.code] || PHASE_MESSAGES['LUT'];
  msg.innerHTML = '"' + messages[Math.floor(Math.random() * messages.length)] + '"';
}

/* ── Period proximity signal ─────────────────────────────────────────────── */
function renderProximitySignal(proximity, phase) {
  var section = document.getElementById('proximity-section');
  var card    = document.getElementById('proximity-card');
  if (!section || !card) return;

  if (!proximity || proximity.signal === 'none') {
    section.style.display = 'none';
    return;
  }

  var isIrregular = phase && (phase.regularity === 'irregular' || phase.regularity === 'limited_history');
  var conf = proximity.probability >= 0.80 ? 'strong' : 'moderate';
  var headline = conf === 'strong'
    ? 'Period may be arriving soon'
    : 'Pre-menstrual symptoms detected';

  var subtext = proximity.note || '';
  if (isIrregular) {
    subtext += (subtext ? ' ' : '') + 'Based on symptoms — cycle timing is irregular.';
  }

  card.innerHTML =
    '<div class="proximity-icon">🗓</div>' +
    '<div class="proximity-body">' +
      '<div class="proximity-title">' + headline + '</div>' +
      (subtext ? '<div class="proximity-sub">' + subtext + '</div>' : '') +
    '</div>';

  section.style.display = '';
}

/* ── Today picks ─────────────────────────────────────────────────────────── */
function renderTodayPicks(results) {
  var container = document.getElementById('today-picks');
  if (!results) return;

  var picks = [];
  DOMAIN_ORDER.forEach(function(domain) {
    if (!results[domain]) return;
    var icon = DOMAIN_ICONS[domain];
    Object.values(results[domain]).forEach(function(items) {
      var visible = items.filter(function(i){ return !i.is_blocked; });
      if (visible.length) {
        var top = visible[0];
        picks.push({ icon: icon, domain: domain, item: top });
      }
    });
  });

  // Show top pick per domain (max 4)
  var seen = {};
  var filtered = picks.filter(function(p) {
    if (seen[p.domain]) return false;
    seen[p.domain] = true;
    return true;
  }).slice(0, 4);

  if (!filtered.length) {
    container.innerHTML = '<div class="picks-placeholder"><span>No recommendations available</span></div>';
    return;
  }

  container.innerHTML = filtered.map(function(p) {
    var dur = p.item.dur_max > 0 ? p.item.dur_min + '–' + p.item.dur_max + ' min' : '';
    return '<div class="pick-card">' +
      '<div class="pick-icon">' + p.icon + '</div>' +
      '<div class="pick-body">' +
        '<div class="pick-title">' + p.item.text + '</div>' +
        '<div class="pick-meta">' + capitalize(p.domain) + (dur ? ' · ' + dur : '') + '</div>' +
      '</div>' +
      (p.item.intensity ? '<span class="pick-tag">' + p.item.intensity + '</span>' : '') +
    '</div>';
  }).join('');

  // Cycle insight
  if (state.phase && state.phase.code) {
    var insightSection = document.getElementById('cycle-insight-section');
    var insightCard    = document.getElementById('cycle-insight-card');
    insightSection.style.display = '';
    insightCard.innerHTML =
      '<h4>Phase Insight</h4>' +
      '<p>' + phaseInsight(state.phase.code) + '</p>';
  }
}

function phaseInsight(code) {
  var insights = {
    MEN: 'Iron and rest are your priorities. Light movement like yoga or walking supports recovery without adding stress.',
    FOL: 'Rising estrogen means rising strength. This is your best window for new habits, workouts, and creative thinking.',
    OVU: 'Peak energy and social energy. Push your training, connect with people, eat antioxidant-rich foods.',
    LUT: 'Progesterone rises — prioritise sleep, reduce inflammation, and taper exercise intensity.',
    LL:  'PMS symptoms may peak. Magnesium, mood support, and gentle movement are your best tools right now.',
  };
  return insights[code] || '';
}

/* ── Guide tab ───────────────────────────────────────────────────────────── */
function renderGuide(results, phase) {
  var empty   = document.getElementById('guide-empty');
  var content = document.getElementById('guide-content');

  if (!results) { empty.classList.remove('hidden'); content.classList.add('hidden'); return; }

  empty.classList.add('hidden');
  content.classList.remove('hidden');

  if (phase) {
    document.getElementById('guide-phase-sub').textContent =
      (phase.label || phase.code) + (phase.cycle_day ? ' · Day ' + phase.cycle_day : '');
  }

  // Exercise mode banner
  var exerciseBanner = '';
  if (results.exercise) {
    exerciseBanner = buildExerciseBanner(results.exercise);
  }

  var domainsHtml = DOMAIN_ORDER
    .filter(function(d) { return results[d]; })
    .map(function(domain) { return renderGuideDomain(domain, results[domain]); })
    .join('');

  content.innerHTML = exerciseBanner + domainsHtml;

  // Reset filter to All
  document.querySelectorAll('.filter-tab').forEach(function(t) { t.classList.remove('active'); });
  document.querySelector('.filter-tab[data-domain="all"]').classList.add('active');
}

function buildExerciseBanner(exerciseData) {
  var recScore   = topScore(exerciseData.recovery || []);
  var trainScore = topScore(exerciseData.training || []);
  var gap        = recScore - trainScore;

  var mode, icon, sub;
  if (gap >= 15)       { mode = 'Recovery Day';  icon = '💚'; sub = 'Rest and gentle movement today.'; }
  else if (gap >= 5)   { mode = 'Lean Recovery'; icon = '🌿'; sub = 'Light activity preferred.'; }
  else if (gap <= -15) { mode = 'Training Day';  icon = '💪'; sub = 'Your body is primed — push it.'; }
  else if (gap <= -5)  { mode = 'Lean Training'; icon = '🏃'; sub = 'Moderate training conditions.'; }
  else                 { mode = 'Balanced';       icon = '⚖️'; sub = 'Either workout or rest works.'; }

  return '<div class="exercise-banner">' +
    '<div class="exercise-banner-icon">' + icon + '</div>' +
    '<div class="exercise-banner-text">' +
      '<div class="exercise-banner-mode">' + mode + '</div>' +
      '<div class="exercise-banner-sub">' + sub + '</div>' +
    '</div>' +
  '</div>';
}

function renderGuideDomain(domain, categories) {
  var icon = DOMAIN_ICONS[domain] || '';
  var catsHtml = Object.entries(categories)
    .map(function(e) { return renderGuideCategory(e[0], e[1]); })
    .filter(Boolean)
    .join('');

  if (!catsHtml) return '';

  return '<div class="guide-domain" data-domain="' + domain + '">' +
    '<div class="guide-domain-title">' +
      '<span class="guide-domain-icon">' + icon + '</span>' +
      capitalize(domain) +
    '</div>' +
    catsHtml +
  '</div>';
}

function renderGuideCategory(category, items) {
  var visible = items.filter(function(i) { return !i.is_blocked; });
  if (!visible.length) return '';

  var cardsHtml = visible.map(function(item) {
    var tags = '';
    if (item.intensity) tags += '<span class="rec-tag">' + item.intensity + '</span>';
    if (item.dur_max > 0) tags += '<span class="rec-tag duration">' + item.dur_min + '–' + item.dur_max + ' min</span>';
    return '<div class="rec-card">' +
      '<div class="rec-title">' + item.text + '</div>' +
      (tags ? '<div class="rec-tags">' + tags + '</div>' : '') +
    '</div>';
  }).join('');

  return '<div class="guide-category">' +
    '<div class="guide-category-label">' + capitalize(category) + '</div>' +
    cardsHtml +
  '</div>';
}

/* ── Cycle tab ───────────────────────────────────────────────────────────── */
function renderCycleTab(phase, fertility, timing) {
  var container = document.getElementById('cycle-content');
  if (!phase) return;

  var statsHtml = '';
  if (phase.cycle_day || phase.cycle_length) {
    statsHtml = '<div class="cycle-stat-grid">';
    if (phase.cycle_day) {
      statsHtml += '<div class="cycle-stat"><div class="cycle-stat-value">' + phase.cycle_day + '</div><div class="cycle-stat-label">Cycle Day</div></div>';
    }
    if (phase.cycle_length) {
      statsHtml += '<div class="cycle-stat"><div class="cycle-stat-value">' + Math.round(phase.cycle_length) + '</div><div class="cycle-stat-label">Cycle Length</div></div>';
    }
    statsHtml += '</div>';
  }

  var nextPeriod = '';
  if (timing && timing.predicted_next_period) {
    nextPeriod = '<div class="cycle-info-card"><div class="cycle-info-title">Next Period</div><div class="cycle-info-value">~' + timing.predicted_next_period + '</div></div>';
  }

  var fertileWindow = '';
  if (timing && timing.fertile_window) {
    fertileWindow = '<div class="cycle-info-card"><div class="cycle-info-title">Fertile Window</div><div class="cycle-info-value">' + timing.fertile_window + '</div></div>';
  }

  var fertilityHtml = '';
  if (fertility && fertility.status) {
    var fertClass = fertility.status === 'Red Day' ? 'fert-red' : fertility.status === 'Light Red Day' ? 'fert-lightred' : 'fert-green';
    fertilityHtml = '<div class="cycle-info-card"><div class="cycle-info-title">Fertility Status</div><div class="cycle-info-value"><span class="fertility-badge ' + fertClass + '">' + fertility.status + '</span></div></div>';
  }

  var probs = phase.phase_probs || {};
  var phaseOrder = ['Menstrual','Follicular','Fertility','Luteal'];
  var probsHtml = '<div class="cycle-info-card"><div class="cycle-info-title">Phase Probabilities</div><div class="cycle-prob-bars">';
  phaseOrder.forEach(function(p) {
    var pct = Math.round((probs[p] || 0) * 100);
    probsHtml += '<div class="prob-row"><span class="prob-label">' + p + '</span><div class="prob-bar-wrap"><div class="prob-bar" style="width:' + pct + '%"></div></div><span class="prob-pct">' + pct + '%</span></div>';
  });
  probsHtml += '</div></div>';

  var timingNote = '';
  if (timing && timing.note) {
    timingNote = '<div class="cycle-info-card"><div class="cycle-info-title">Timing Note</div><div class="cycle-info-value" style="font-size:.85rem;font-weight:400;color:var(--text-2)">' + timing.note + '</div></div>';
  }

  container.innerHTML = statsHtml + nextPeriod + fertileWindow + fertilityHtml + probsHtml + timingNote;
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function topScore(items) {
  if (!items || !items.length) return 0;
  var u = items.filter(function(i){ return !i.is_blocked; });
  return u.length ? Math.max.apply(null, u.map(function(i){ return i.final_score; })) : 0;
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

/* ── Loading ─────────────────────────────────────────────────────────────── */
function setLoading(on) {
  var btn = document.getElementById('submit-btn');
  var lbl = document.getElementById('submit-label');
  var gl  = document.getElementById('guide-loading');
  btn.disabled = on;
  lbl.textContent = on ? 'Loading…' : 'Get My Recommendations';
  if (gl) gl.classList.toggle('hidden', !on);
}

function showError(msg) {
  var bar = document.getElementById('error-bar');
  bar.textContent = 'Error: ' + msg;
  bar.classList.remove('hidden');
  setTimeout(function(){ bar.classList.add('hidden'); }, 5000);
}

function clearError() {
  document.getElementById('error-bar').classList.add('hidden');
}
