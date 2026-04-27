"""
InBalance Validation Matrix — HTML Report Generator
Run: python tests/validate_report.py
Opens: tests/validation_report.html
"""

import sys, os, json, webbrowser
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recommender import recommend

TOP_N   = 3
FETCH_N = 50

# ── re-use helpers from validate.py ──────────────────────────────────────────
def get_group(results, domain, category):
    return results.get(domain, {}).get(category, [])

def top_unblocked(items, n=TOP_N):
    return [i for i in items if not i["is_blocked"]][:n]

def tags_of(items):
    return {tag for item in items for tag in item.get("tags", [])}

def top_score(items):
    visible = [i for i in items if not i["is_blocked"]]
    return max((i["final_score"] for i in visible), default=0)

def check(scenario, results):
    failures = []
    for a in scenario["assertions"]:
        t = a["type"]
        items = get_group(results, a["domain"], a.get("category", ""))
        if t == "top_tag":
            if a["tag"] not in tags_of(top_unblocked(items)):
                failures.append(f"top_tag '{a['tag']}' missing from unblocked top-{TOP_N} of {a['domain']}/{a['category']}")
        elif t == "not_top_tag":
            found = [i["text"][:45] for i in top_unblocked(items) if a["tag"] in i.get("tags", [])]
            if found:
                failures.append(f"not_top_tag '{a['tag']}' found in top-{TOP_N} of {a['domain']}/{a['category']}")
        elif t == "blocked_tag":
            blocked_items = [i for i in items if i["is_blocked"] and a["tag"] in i.get("tags", [])]
            if not blocked_items:
                failures.append(f"blocked_tag '{a['tag']}' — no blocked item found in {a['domain']}/{a['category']}")
        elif t == "score_gap":
            items_a = get_group(results, a["domain"], a["cat_high"])
            items_b = get_group(results, a["domain"], a["cat_low"])
            score_a = top_score(items_a)
            score_b = top_score(items_b)
            gap = score_a - score_b
            if gap < a["min_gap"]:
                failures.append(f"score_gap {a['domain']}/{a['cat_high']}({score_a}) vs {a['cat_low']}({score_b}) gap={gap} < {a['min_gap']}")
    return failures

# ── import scenarios ──────────────────────────────────────────────────────────
from tests.validate import SCENARIOS

# ── run all scenarios and collect detail ──────────────────────────────────────
report = []
for scenario in SCENARIOS:
    results  = recommend(scenario["inputs"], top_n=FETCH_N)
    failures = check(scenario, results)

    # collect top-3 unblocked per domain for display
    domains_detail = {}
    for domain in ["exercise","nutrition","sleep","mental"]:
        dom_data = results.get(domain, {})
        cats = {}
        for cat, items in dom_data.items():
            ub = top_unblocked(items, TOP_N)
            bl = [i for i in items if i["is_blocked"]]
            cats[cat] = {
                "top": [{"text": i["text"], "score": i["final_score"], "tags": i["tags"]} for i in ub],
                "blocked_count": len(bl),
                "blocked_tags": sorted({t for i in bl for t in i["tags"] if t not in ["nutrition","exercise","mental","sleep","low_intensity","moderate_intensity","high_intensity"]}),
            }
        if cats:
            domains_detail[domain] = cats

    assertion_detail = []
    for a in scenario["assertions"]:
        desc = ""
        t = a["type"]
        if t in ("top_tag","not_top_tag","blocked_tag"):
            desc = f"{t}: '{a['tag']}' in {a['domain']}/{a['category']}"
        elif t == "score_gap":
            desc = f"score_gap: {a['domain']}/{a['cat_high']} beats {a['cat_low']} by ≥{a['min_gap']}"
        assertion_detail.append({"desc": desc, "pass": True})  # will mark failed below

    fail_descs = set(failures)
    for i, a in enumerate(scenario["assertions"]):
        t = a["type"]
        # check if this assertion failed
        for f in failures:
            tag_key = a.get("tag","") or a.get("cat_high","")
            if tag_key in f:
                assertion_detail[i]["pass"] = False
                break

    report.append({
        "name":       scenario["name"],
        "inputs":     scenario["inputs"],
        "passed":     len(failures) == 0,
        "failures":   failures,
        "assertions": assertion_detail,
        "domains":    domains_detail,
    })

passed = sum(1 for r in report if r["passed"])
total  = len(report)

# ── render HTML ───────────────────────────────────────────────────────────────
DOMAIN_COLOR = {
    "exercise":  "#2563eb",
    "nutrition": "#16a34a",
    "sleep":     "#7c3aed",
    "mental":    "#d97706",
}

def inp_html(inputs):
    badges = []
    for k, v in inputs.items():
        badges.append(f'<span class="badge">{k}: <strong>{v}</strong></span>')
    return " ".join(badges)

def assertion_html(assertions):
    rows = []
    for a in assertions:
        icon = "✅" if a["pass"] else "❌"
        cls  = "assert-pass" if a["pass"] else "assert-fail"
        rows.append(f'<div class="assert-row {cls}">{icon} {a["desc"]}</div>')
    return "\n".join(rows)

def top_items_html(domains):
    sections = []
    for domain, cats in domains.items():
        color = DOMAIN_COLOR.get(domain, "#888")
        cat_html = []
        for cat, data in cats.items():
            items_html = ""
            for item in data["top"]:
                tag_chips = " ".join(
                    f'<span class="chip">{t}</span>'
                    for t in item["tags"]
                    if t not in ["nutrition","exercise","mental","sleep"]
                )
                items_html += f'''
                <div class="rec-item">
                  <span class="rec-score">{item["score"]}</span>
                  <span class="rec-text">{item["text"][:70]}{"…" if len(item["text"])>70 else ""}</span>
                  <div class="chips">{tag_chips}</div>
                </div>'''
            blocked_note = ""
            if data["blocked_count"]:
                bt = ", ".join(data["blocked_tags"][:6])
                blocked_note = f'<div class="blocked-note">🚫 {data["blocked_count"]} blocked · tags: {bt}</div>'
            cat_html.append(f'''
            <div class="cat-block">
              <div class="cat-label">{cat}</div>
              {items_html}
              {blocked_note}
            </div>''')
        sections.append(f'''
        <div class="domain-block" style="border-left:3px solid {color}">
          <div class="domain-label" style="color:{color}">{domain.upper()}</div>
          {"".join(cat_html)}
        </div>''')
    return "\n".join(sections)

cards = []
for r in report:
    status_cls = "card-pass" if r["passed"] else "card-fail"
    status_icon = "✅ PASS" if r["passed"] else "❌ FAIL"
    fail_block = ""
    if r["failures"]:
        items = "".join(f"<li>{f}</li>" for f in r["failures"])
        fail_block = f'<div class="fail-block"><strong>Failures:</strong><ul>{items}</ul></div>'

    cards.append(f'''
    <details class="card {status_cls}">
      <summary>
        <span class="status-badge">{status_icon}</span>
        <span class="scenario-name">{r["name"]}</span>
        <span class="input-badges">{inp_html(r["inputs"])}</span>
      </summary>
      <div class="card-body">
        {fail_block}
        <div class="two-col">
          <div class="col-assertions">
            <h4>Assertions</h4>
            {assertion_html(r["assertions"])}
          </div>
          <div class="col-results">
            <h4>Top-{TOP_N} Results</h4>
            {top_items_html(r["domains"])}
          </div>
        </div>
      </div>
    </details>''')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>InBalance · Validation Report</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
       font-size: 14px; background: #f4f4f7; color: #1a1a2e; padding: 32px 20px 80px; }}
.container {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }}
.subtitle {{ color: #777; font-size: 0.85rem; margin-bottom: 24px; }}

/* Summary bar */
.summary {{ display: flex; align-items: center; gap: 16px; background: #fff;
            border: 1px solid #e4e4e8; border-radius: 10px; padding: 16px 20px;
            margin-bottom: 24px; }}
.summary-score {{ font-size: 2rem; font-weight: 800; color: #16a34a; }}
.summary-score.has-fail {{ color: #dc2626; }}
.summary-label {{ font-size: 0.8rem; color: #888; }}
.summary-bar {{ flex: 1; height: 10px; background: #f0f0f4; border-radius: 5px; overflow: hidden; }}
.summary-fill {{ height: 100%; background: #16a34a; border-radius: 5px;
                  width: {int(passed/total*100)}%; transition: width 0.4s; }}

/* Cards */
.card {{ background: #fff; border: 1px solid #e4e4e8; border-radius: 10px;
         margin-bottom: 10px; overflow: hidden; }}
.card-pass {{ border-left: 4px solid #16a34a; }}
.card-fail {{ border-left: 4px solid #dc2626; }}
summary {{ list-style: none; padding: 14px 18px; cursor: pointer;
           display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
summary::-webkit-details-marker {{ display: none; }}
summary:hover {{ background: #f9f9fb; }}
.status-badge {{ font-size: 0.8rem; font-weight: 700; white-space: nowrap; }}
.scenario-name {{ font-weight: 700; font-size: 0.95rem; min-width: 180px; }}
.input-badges {{ display: flex; flex-wrap: wrap; gap: 5px; }}
.badge {{ font-size: 0.72rem; background: #f0f0f4; border-radius: 4px; padding: 2px 7px; color: #555; }}

.card-body {{ padding: 16px 18px 20px; border-top: 1px solid #f0f0f4; }}
.fail-block {{ background: #fff5f5; border: 1px solid #fecaca; border-radius: 6px;
               padding: 10px 14px; margin-bottom: 14px; color: #b91c1c; font-size: 0.82rem; }}
.fail-block ul {{ margin-left: 18px; margin-top: 4px; }}

.two-col {{ display: grid; grid-template-columns: 280px 1fr; gap: 20px; }}
h4 {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: #aaa; margin-bottom: 10px; font-weight: 600; }}

/* Assertions */
.assert-row {{ font-size: 0.82rem; padding: 5px 8px; border-radius: 5px;
               margin-bottom: 4px; }}
.assert-pass {{ background: #f0fdf4; color: #166534; }}
.assert-fail {{ background: #fff5f5; color: #991b1b; font-weight: 600; }}

/* Results */
.domain-block {{ padding: 8px 12px; margin-bottom: 10px; border-radius: 6px;
                 background: #fafafa; }}
.domain-label {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
                 letter-spacing: 0.08em; margin-bottom: 8px; }}
.cat-block {{ margin-bottom: 10px; }}
.cat-label {{ font-size: 0.68rem; color: #bbb; text-transform: uppercase;
              letter-spacing: 0.06em; margin-bottom: 4px; font-weight: 600; }}
.rec-item {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px;
             padding: 5px 0; border-bottom: 1px solid #f0f0f4; }}
.rec-item:last-child {{ border-bottom: none; }}
.rec-score {{ font-size: 0.75rem; font-weight: 700; color: #6c5ce7;
              min-width: 30px; text-align: right; }}
.rec-text {{ font-size: 0.82rem; color: #333; flex: 1; min-width: 200px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 3px; width: 100%; padding-left: 36px; }}
.chip {{ font-size: 0.65rem; background: #ede9fe; color: #6c5ce7;
          border-radius: 3px; padding: 1px 5px; }}
.blocked-note {{ font-size: 0.72rem; color: #dc2626; margin-top: 6px;
                  background: #fff5f5; border-radius: 4px; padding: 4px 8px; }}

@media (max-width: 700px) {{
  .two-col {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="container">
  <h1>InBalance · Validation Report</h1>
  <p class="subtitle">Clinical scenario matrix — engine rules verification</p>

  <div class="summary">
    <div>
      <div class="summary-score {'has-fail' if passed < total else ''}">{passed}/{total}</div>
      <div class="summary-label">scenarios passing</div>
    </div>
    <div class="summary-bar"><div class="summary-fill"></div></div>
    <div style="font-size:0.82rem;color:#888">{int(passed/total*100)}% pass rate</div>
  </div>

  {"".join(cards)}
</div>
</body>
</html>"""

out_path = os.path.join(os.path.dirname(__file__), "validation_report.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✅  Report written → {out_path}")
print(f"    {passed}/{total} scenarios passed\n")
webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")
