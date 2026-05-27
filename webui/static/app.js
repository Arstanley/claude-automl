// AutoML dashboard — vanilla JS, no build step.

let currentRunId = null;
let eventSource = null;
let trainingChart = null;
let attemptLogCache = {}; // attempt_id -> array of log lines
let currentAttemptId = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const fmt = {
  pct: (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%"),
  pct3: (v) => (v == null ? "—" : (v * 100).toFixed(2) + "%"),
  ms: (v) => (v == null ? "—" : v < 1 ? v.toFixed(3) + " ms" : v.toFixed(2) + " ms"),
  bytes: (v) => {
    if (v == null) return "—";
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0; let n = v;
    while (n > 1024 && i < u.length - 1) { n /= 1024; i++; }
    return (i === 0 ? n.toFixed(0) : n.toFixed(1)) + " " + u[i];
  },
  short: (s, n = 60) => s == null ? "" : (s.length > n ? s.slice(0, n - 1) + "…" : s),
  num: (v, d = 3) => v == null ? "—" : (typeof v === "number" ? v.toFixed(d) : String(v)),
  time: (iso) => iso ? String(iso).slice(11, 19) : "",
  date: (iso) => iso ? String(iso).slice(0, 10) : "",
  duration: (a, b) => {
    if (!a || !b) return "—";
    const s = (new Date(b) - new Date(a)) / 1000;
    if (s < 60) return s.toFixed(1) + "s";
    if (s < 3600) return Math.round(s / 60) + "m " + Math.round(s % 60) + "s";
    return Math.floor(s / 3600) + "h " + Math.round((s % 3600) / 60) + "m";
  },
};

// ============================================================
// bootstrap
// ============================================================

window.addEventListener("DOMContentLoaded", () => {
  $("#refresh-runs").addEventListener("click", loadRuns);
  $$("nav#tabs button").forEach(b => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  loadRuns();
  tickClock();
  setInterval(tickClock, 1000);
});

function tickClock() {
  const el = $("#server-clock"); if (!el) return;
  const d = new Date();
  el.textContent = [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map(x => String(x).padStart(2, "0")).join(":");
}

// ============================================================
// run list
// ============================================================

async function loadRuns() {
  const res = await fetch("/api/runs");
  const runs = await res.json();
  const list = $("#run-list");
  list.innerHTML = "";
  if (runs.length === 0) {
    list.innerHTML = `<li class="run-list-empty">
      <div class="run-id-row">— empty —</div>
      <div class="run-prompt-row">No runs yet. Commission one with /automl in Claude Code.</div>
    </li>`;
    return;
  }
  for (const r of runs) {
    const li = document.createElement("li");
    if (r.run_id === currentRunId) li.classList.add("active");
    li.innerHTML = `
      <div class="run-id-row">${esc(r.run_id)}</div>
      <div class="run-prompt-row">${esc(fmt.short(r.prompt, 90))}</div>
      <div class="run-meta-row">
        <span class="status-pill ${r.status}">${esc(r.status)}</span>
        <span>${r.n_attempts} attempt${r.n_attempts === 1 ? "" : "s"}</span>
      </div>`;
    li.addEventListener("click", () => selectRun(r.run_id));
    list.appendChild(li);
  }
}

// ============================================================
// select run
// ============================================================

async function selectRun(runId) {
  currentRunId = runId;
  $$("#run-list li").forEach(li => li.classList.remove("active"));
  const target = $$("#run-list li").find(li => li.querySelector(".run-id-row")?.textContent === runId);
  if (target) target.classList.add("active");

  $("#empty-state").hidden = true;
  $("#run-view").hidden = false;

  if (eventSource) { eventSource.close(); eventSource = null; }
  attemptLogCache = {};

  const state = await (await fetch(`/api/runs/${runId}`)).json();
  renderState(state);

  eventSource = new EventSource(`/api/runs/${runId}/events`);
  eventSource.addEventListener("state", e => renderState(JSON.parse(e.data)));
  eventSource.addEventListener("attempt_log", e => {
    const d = JSON.parse(e.data);
    if (!attemptLogCache[d.attempt_id]) attemptLogCache[d.attempt_id] = [];
    attemptLogCache[d.attempt_id].push(d.line);
    if (currentAttemptId === d.attempt_id) renderTrainingChart(d.attempt_id);
  });
}

// ============================================================
// top-level renderer
// ============================================================

function renderState(state) {
  if (!state || !state.run_id) return;

  // --- header ---
  safe("header", () => {
    $("#run-id").textContent = state.run_id;
    $("#run-prompt").textContent = state.prompt || "";
    $("#run-status-pill").innerHTML = `<span class="status-pill ${esc(state.status)}">${esc(state.status)}</span>`;
  });

  safe("byline",        () => renderByline(state));
  safe("kpi",           () => renderKpiStrip(state));
  safe("gate",          () => renderGateBanner(state));

  // --- §1 plan ---
  safe("task-identity", () => renderTaskIdentity(state));
  safe("classes",       () => renderClasses(state));
  safe("hard-rules",    () => renderHardRules(state));
  safe("budgets",       () => renderBudgets(state));
  safe("examples",      () => renderExamples(state));
  safe("plan-summary",  () => renderPlanSummary(state));
  safe("plan-attempts", () => renderPlanAttempts(state));
  safe("eval-protocol", () => renderEvalProtocol(state));
  safe("research",      () => renderResearch(state));

  // --- §2 data ---
  safe("dataset",       () => renderDataset(state));

  // --- §3 attempts ---
  safe("leaderboard",   () => renderLeaderboard(state));

  // --- §4 thoughts ---
  safe("thoughts",      () => renderThoughtsTimeline(state));

  // --- §5 report ---
  safe("report",        () => renderReport(state));
}

function safe(name, fn) {
  try { fn(); }
  catch (e) { console.error(`[render:${name}]`, e); }
}

// ============================================================
// header byline + KPI strip
// ============================================================

function renderByline(s) {
  const el = $("#run-byline");
  const parts = [];
  if (s.mode) parts.push(`<span class="by-item">mode<strong>${esc(s.mode)}</strong></span>`);
  if (s.phase) parts.push(`<span class="by-item">phase<strong>${esc(s.phase)}</strong></span>`);
  if (s.created_at) parts.push(`<span class="by-item">started<strong>${esc(s.created_at.slice(0,16).replace("T", " "))}</strong></span>`);
  if (s.created_at && s.updated_at) parts.push(`<span class="by-item">duration<strong>${fmt.duration(s.created_at, s.updated_at)}</strong></span>`);
  if (s.attempts) parts.push(`<span class="by-item">attempts<strong>${s.attempts.length}</strong></span>`);
  el.innerHTML = parts.join("");
}

function renderKpiStrip(s) {
  const el = $("#kpi-strip");
  el.innerHTML = "";
  const best = (s.attempts || []).find(a => a.id === s.best_attempt_id) || null;
  const ev = best?.eval;
  const tiles = [];

  // primary metric (best)
  if (ev?.primary_metric) {
    tiles.push(tile(
      ev.primary_metric.name || "primary",
      `<span class="kpi-value amber">${fmt.num(ev.primary_metric.value, 4)}</span>`,
      best ? `${best.id}` : "",
      true
    ));
  } else if (s.constraints?.accuracy_targets?.primary_metric) {
    tiles.push(tile("Primary metric", `<span class="kpi-value">${esc(s.constraints.accuracy_targets.primary_metric.split(" on ")[0] || s.constraints.accuracy_targets.primary_metric)}</span>`, "—"));
  }

  // hard rules pass count
  if (s.constraints?.hard_rules && best?.eval?.meets_constraints?.checks) {
    const checks = best.eval.meets_constraints.checks;
    const passed = checks.filter(c => c.pass).length;
    const total = checks.length;
    const allOk = best.eval.meets_constraints.overall;
    tiles.push(tile(
      "Hard rules",
      `<span class="kpi-value ${allOk ? "moss" : "rust"}">${passed}/${total}</span>`,
      allOk ? "all passed" : "violations"
    ));
  } else if (s.constraints?.hard_rules) {
    tiles.push(tile("Hard rules", `<span class="kpi-value">${s.constraints.hard_rules.length}</span>`, "defined"));
  }

  // latency
  if (ev?.latency?.p99_ms != null) {
    const limit = s.constraints?.latency?.max_ms;
    const ok = limit ? ev.latency.p99_ms <= limit : null;
    tiles.push(tile(
      "p99 latency",
      `<span class="kpi-value ${ok === true ? "moss" : ok === false ? "rust" : ""}">${fmt.ms(ev.latency.p99_ms)}</span>`,
      limit ? `≤ ${limit} ms target` : "CPU"
    ));
  } else if (s.constraints?.latency?.max_ms) {
    tiles.push(tile("Latency target", `<span class="kpi-value">≤ ${s.constraints.latency.max_ms} ms</span>`, "p99 / CPU"));
  }

  // model size
  if (ev?.model_size_bytes != null || best?.model_size_bytes != null) {
    const sz = ev?.model_size_bytes ?? best?.model_size_bytes;
    const limit = s.constraints?.model_size?.max_bytes;
    const ok = limit ? sz <= limit : null;
    tiles.push(tile(
      "Model size",
      `<span class="kpi-value ${ok === true ? "moss" : ok === false ? "rust" : ""}">${fmt.bytes(sz)}</span>`,
      limit ? `≤ ${fmt.bytes(limit)}` : ""
    ));
  } else if (s.constraints?.model_size?.max_bytes) {
    tiles.push(tile("Size cap", `<span class="kpi-value">${fmt.bytes(s.constraints.model_size.max_bytes)}</span>`, ""));
  }

  // dataset size
  if (s.dataset?.total_examples) {
    tiles.push(tile("Dataset", `<span class="kpi-value">${(s.dataset.total_examples).toLocaleString()}</span>`, `${Object.keys(s.dataset.splits || {}).length} splits`));
  }

  el.innerHTML = tiles.join("");
}

function tile(label, valueHtml, sub, accent = false) {
  return `<div class="kpi-tile${accent ? " accent" : ""}">
    <div class="kpi-label">${esc(label)}</div>
    ${valueHtml}
    ${sub ? `<div class="kpi-sub">${esc(sub)}</div>` : ""}
  </div>`;
}

// ============================================================
// gate banner
// ============================================================

function renderGateBanner(s) {
  const banner = $("#gate-banner");
  const pending = (s.gates || []).find(g => g.status === "pending");
  if (!pending) { banner.innerHTML = ""; return; }
  banner.innerHTML = `
    <div class="gate-banner">
      <h3>Awaiting your decision · <code>${esc(pending.id)}</code></h3>
      <p>${esc(pending.summary || "")}</p>
      <div class="gate-actions">
        <button class="approve" data-gate="${esc(pending.id)}" data-action="approve">Approve</button>
        <button class="reject"  data-gate="${esc(pending.id)}" data-action="reject">Reject</button>
        <button class="edit"    data-gate="${esc(pending.id)}" data-action="edit">Request edit</button>
      </div>
    </div>`;
  banner.querySelectorAll("button").forEach(b => b.addEventListener("click", () => gateAction(s.run_id, b.dataset.gate, b.dataset.action)));
}

// ============================================================
// §1 PLAN
// ============================================================

function renderTaskIdentity(s) {
  const c = s.constraints || {};
  const items = [];
  if (c.task_type) items.push(idItem("Task type", c.task_type));
  if (c.domain) items.push(idItem("Domain", c.domain));
  if (c.input?.modality) items.push(idItem("Input", c.input.modality + (c.input.typical_length ? ` · ${c.input.typical_length}` : "")));
  if (c.output?.type) items.push(idItem("Output", c.output.type));
  if (c.accuracy_targets?.primary_metric) items.push(idItem("Primary metric", c.accuracy_targets.primary_metric));
  if (c.accuracy_targets?.target) items.push(idItem("Target", c.accuracy_targets.target));
  $("#task-identity").innerHTML = items.length
    ? `<div class="identity-grid">${items.join("")}</div>`
    : `<div class="empty-line">No constraints parsed yet.</div>`;
}
function idItem(label, val, mono = false) {
  return `<div class="identity-item">
    <div class="label">${esc(label)}</div>
    <div class="val ${mono ? "mono" : ""}">${esc(val)}</div>
  </div>`;
}

function renderClasses(s) {
  const c = s.constraints || {};
  const classes = c.classes || [];
  $("#classes-count").textContent = classes.length ? `${classes.length} labels` : "";
  if (!classes.length) {
    $("#classes-by-tier").innerHTML = `<div class="empty-line">No label space defined.</div>`;
    $("#special-classes").innerHTML = "";
    return;
  }
  const tiers = c.class_tiers || {};
  const tierOrder = ["tier1", "tier2", "tier3", "additional", "special"];
  const tierLabel = {tier1: "Tier 1", tier2: "Tier 2", tier3: "Tier 3", additional: "Additional", special: "Special"};
  let html = "";
  const assigned = new Set();
  for (const t of tierOrder) {
    const ms = tiers[t] || [];
    if (!ms.length) continue;
    ms.forEach(m => assigned.add(m));
    html += `<div class="tier-block">
      <div class="tier-head">
        <span class="tier-name">${esc(tierLabel[t] || t)}</span>
        <span class="tier-count">${ms.length}</span>
        <span class="tier-rule"></span>
      </div>
      <div class="chip-row">${ms.map(m => `<span class="chip ${t}">${esc(m)}</span>`).join("")}</div>
    </div>`;
  }
  const unassigned = classes.filter(c => !assigned.has(c));
  if (unassigned.length && !Object.keys(tiers).length) {
    // no tiers — render all as flat
    html = `<div class="chip-row">${classes.map(m => `<span class="chip">${esc(m)}</span>`).join("")}</div>`;
  } else if (unassigned.length) {
    html += `<div class="tier-block">
      <div class="tier-head"><span class="tier-name">Unclassified</span><span class="tier-count">${unassigned.length}</span><span class="tier-rule"></span></div>
      <div class="chip-row">${unassigned.map(m => `<span class="chip">${esc(m)}</span>`).join("")}</div>
    </div>`;
  }
  $("#classes-by-tier").innerHTML = html;

  // special-classes callouts (with descriptions)
  const sp = c.special_classes || {};
  const keys = Object.keys(sp);
  if (keys.length) {
    $("#special-classes").innerHTML = keys.map(k =>
      `<div class="special-class"><span class="key">${esc(k)}</span>${esc(sp[k])}</div>`
    ).join("");
  } else {
    $("#special-classes").innerHTML = "";
  }
}

function renderHardRules(s) {
  const rules = s.constraints?.hard_rules || [];
  const el = $("#hard-rules");
  $("#hard-rules-count").textContent = rules.length ? `${rules.length} checks` : "";
  if (!rules.length) { el.innerHTML = `<div class="empty-line">No hard rules defined.</div>`; return; }

  // pull check status from the best attempt's eval (if any), or aggregate across all
  const best = (s.attempts || []).find(a => a.id === s.best_attempt_id);
  const checksById = {};
  if (best?.eval?.meets_constraints?.checks) {
    for (const c of best.eval.meets_constraints.checks) checksById[c.id] = c;
  }

  el.innerHTML = rules.map(r => {
    const ch = checksById[r.id];
    const klass = ch ? (ch.pass ? "passed" : "failed") : "pending";
    const label = ch ? (ch.pass ? "passed" : "failed") : "pending";
    const actualLine = ch ? `<div class="rule-test"><strong>actual</strong>${esc(String(ch.actual ?? ""))}</div>` : "";
    return `<div class="rule-card ${klass}">
      <div class="rule-head">
        <span class="rule-id">${esc(r.id)}</span>
        <span class="rule-status">${label}</span>
      </div>
      <div class="rule-desc">${esc(r.description || "")}</div>
      ${r.test ? `<div class="rule-test"><strong>test</strong>${esc(r.test)}</div>` : ""}
      ${actualLine}
    </div>`;
  }).join("");
}

function renderBudgets(s) {
  const c = s.constraints || {};
  const tiles = [];
  if (c.latency) tiles.push(budgetTile("Latency", `${c.latency.max_ms} ms`, `${c.latency.p || "p99"} · ${c.latency.device || "CPU"}`));
  if (c.model_size) tiles.push(budgetTile("Model size", fmt.bytes(c.model_size.max_bytes), c.model_size.format || ""));
  if (c.compute_budget) tiles.push(budgetTile("Compute", `${c.compute_budget.max_hours} h`, c.compute_budget.preferred_device || ""));
  if (c.deliverables?.length) tiles.push(budgetTile("Deliverables", c.deliverables.length, c.deliverables.join(", ")));

  $("#budgets-tiles").innerHTML = tiles.length
    ? `<div class="budget-grid">${tiles.join("")}</div>`
    : `<div class="empty-line">No budgets set.</div>`;
}
function budgetTile(label, value, sub) {
  return `<div class="budget-tile">
    <div class="b-label">${esc(label)}</div>
    <div class="b-value">${esc(value)}</div>
    ${sub ? `<div class="b-sub">${esc(sub)}</div>` : ""}
  </div>`;
}

function renderExamples(s) {
  const ex = s.constraints?.input?.examples;
  const panel = $("#examples-panel");
  if (!ex || !ex.length) { panel.hidden = true; return; }
  panel.hidden = false;
  $("#examples-list").innerHTML = `<div class="examples-list">${ex.map(x => `<div class="example-bubble">${esc(x)}</div>`).join("")}</div>`;
}

function renderPlanSummary(s) {
  if (s.plan?.summary) {
    $("#plan-summary").innerHTML = `<div class="markdown-render">${marked.parse(s.plan.summary)}</div>`;
  } else {
    $("#plan-summary").innerHTML = `<div class="empty-line">Planner has not run yet.</div>`;
  }
}

function renderPlanAttempts(s) {
  const attempts = s.plan?.attempts || [];
  const el = $("#plan-attempts-cards");
  if (!attempts.length) { el.innerHTML = `<div class="empty-line">No attempts proposed.</div>`; return; }

  el.innerHTML = attempts.map(a => {
    const hp = a.initial_hyperparams || {};
    const hpHtml = Object.keys(hp).length
      ? `<div class="hp-grid">${Object.entries(hp).map(([k, v]) =>
          `<div class="hp-item"><div class="hp-key">${esc(k)}</div><div class="hp-val">${esc(formatHpVal(v))}</div></div>`
        ).join("")}</div>`
      : "";
    const strengths = (a.expected_strengths || []).map(x => `<span class="tag good">${esc(x)}</span>`).join("");
    const weaknesses = (a.expected_weaknesses || []).map(x => `<span class="tag bad">${esc(x)}</span>`).join("");
    const tagsHtml = (strengths || weaknesses) ? `
      <div>
        ${strengths ? `<span class="tag-label">Strengths</span>` : ""}
        <div class="tag-row" style="display:inline-flex">${strengths}</div>
      </div>
      ${weaknesses ? `<div style="margin-top:6px">
        <span class="tag-label">Risks</span>
        <div class="tag-row" style="display:inline-flex">${weaknesses}</div>
      </div>` : ""}` : "";
    const est = [
      a.est_train_time_min != null ? `~${a.est_train_time_min} min train` : null,
      a.est_model_size_mb != null ? `~${a.est_model_size_mb} MB` : null,
    ].filter(Boolean).join(" · ");
    return `<div class="attempt-card">
      <div class="ac-head">
        <span class="ac-id">${esc(a.id)}</span>
        <span class="ac-name">${esc(a.name || "")}</span>
        ${a.priority != null ? `<span class="ac-prio">prio ${a.priority}</span>` : ""}
      </div>
      ${a.method_family ? `<div class="ac-method">${esc(a.method_family)}${est ? ` · ${esc(est)}` : ""}</div>` : ""}
      ${a.rationale ? `<div class="ac-rationale">${esc(a.rationale)}</div>` : ""}
      ${tagsHtml}
      ${hpHtml}
    </div>`;
  }).join("");
}

function formatHpVal(v) {
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString();
    return String(v);
  }
  if (Array.isArray(v)) return v.join(", ");
  if (v == null) return "—";
  return String(v);
}

function renderEvalProtocol(s) {
  const ep = s.plan?.eval_protocol;
  if (!ep) { $("#eval-protocol").innerHTML = `<div class="empty-line">Not defined.</div>`; return; }
  const items = [];
  if (ep.primary_metric) items.push(evalItem("Primary metric", `<div class="val">${esc(ep.primary_metric)}</div>`));
  if (ep.secondary_metrics?.length) items.push(evalItem("Secondary metrics", `<ul>${ep.secondary_metrics.map(m => `<li>${esc(m)}</li>`).join("")}</ul>`));
  if (ep.slices?.length) {
    items.push(evalItem("Slices", `<ul>${ep.slices.map(sl => {
      if (typeof sl === "string") return `<li>${esc(sl)}</li>`;
      return `<li><strong>${esc(sl.name)}</strong>${sl.primary ? `<span class="primary-tag">primary</span>` : ""}${sl.filter ? ` · <code>${esc(sl.filter)}</code>` : ""}${sl.notes ? `<br><span style="color:var(--muted)">${esc(sl.notes)}</span>` : ""}</li>`;
    }).join("")}</ul>`));
  }
  $("#eval-protocol").innerHTML = `<div class="eval-block">${items.join("")}</div>`;
}
function evalItem(label, html) {
  return `<div class="eval-item"><div class="label">${esc(label)}</div>${html}</div>`;
}

function renderResearch(s) {
  if (s.research) {
    $("#research-panel").hidden = false;
    $("#research-content").innerHTML = `<pre>${esc(JSON.stringify(s.research, null, 2))}</pre>`;
  } else {
    $("#research-panel").hidden = true;
  }
}

// ============================================================
// §2 DATA
// ============================================================

function renderDataset(s) {
  const d = s.dataset;
  if (!d) {
    $("#dataset-summary").innerHTML = `<div class="empty-line">Data engineer has not run yet.</div>`;
    $("#dataset-sources-panel").hidden = true;
    $("#dataset-splits-panel").hidden = true;
    $("#dataset-slices-panel").hidden = true;
    $("#dataset-card-panel").hidden = true;
    return;
  }

  // summary KPIs
  const summaryTiles = [];
  if (d.total_examples != null) summaryTiles.push(`<div class="budget-tile"><div class="b-label">Total examples</div><div class="b-value">${d.total_examples.toLocaleString()}</div></div>`);
  if (d.splits) summaryTiles.push(`<div class="budget-tile"><div class="b-label">Splits</div><div class="b-value">${Object.keys(d.splits).length}</div></div>`);
  if (d.slices) summaryTiles.push(`<div class="budget-tile"><div class="b-label">Slices</div><div class="b-value">${d.slices.length}</div></div>`);
  if (d.warnings?.length) summaryTiles.push(`<div class="budget-tile"><div class="b-label">Warnings</div><div class="b-value" style="color:var(--oxide)">${d.warnings.length}</div></div>`);
  $("#dataset-summary").innerHTML = summaryTiles.length ? `<div class="budget-grid">${summaryTiles.join("")}</div>` : "";

  // sources cards — prefer planner's data_sources (with metadata) over the engineer's flat list
  const planSources = s.plan?.data_sources || [];
  const usedNames = new Set((d.sources_used || []).map(x => String(x).toLowerCase()));
  const sourceList = planSources.length
    ? planSources.filter(ps => !usedNames.size || usedNames.has(String(ps.name).toLowerCase()) || true)
    : (d.sources_used || []).map(n => ({name: n}));
  if (sourceList.length) {
    $("#dataset-sources-panel").hidden = false;
    $("#dataset-sources").innerHTML = `<div class="sources-grid">${sourceList.map(src => `
      <div class="source-card">
        <div class="src-name">${esc(src.name)}</div>
        <div class="src-meta">
          ${src.license ? `<span class="pill">${esc(src.license)}</span>` : ""}
          ${src.size_estimate ? `<span class="pill">${esc(src.size_estimate)}</span>` : ""}
          ${src.url ? `<a href="${esc(src.url)}" target="_blank" rel="noopener">link</a>` : ""}
        </div>
        ${src.use_for ? `<div class="src-use">${esc(src.use_for)}</div>` : ""}
      </div>`).join("")}</div>`;
  } else {
    $("#dataset-sources-panel").hidden = true;
  }

  // splits as bars
  if (d.splits && Object.keys(d.splits).length) {
    $("#dataset-splits-panel").hidden = false;
    const max = Math.max(...Object.values(d.splits).map(x => x.n || 0));
    $("#dataset-splits").innerHTML = `<div class="splits-wrap">${Object.entries(d.splits).map(([name, sp]) => {
      const pct = max ? (sp.n / max * 100) : 0;
      return `<div class="split-row ${esc(name)}">
        <span class="split-name">${esc(name)}</span>
        <span class="split-bar"><span class="split-fill" style="width:${pct.toFixed(2)}%"></span></span>
        <span class="split-n">${(sp.n || 0).toLocaleString()}</span>
      </div>`;
    }).join("")}</div>`;
  } else {
    $("#dataset-splits-panel").hidden = true;
  }

  // slices
  if (d.slices?.length) {
    $("#dataset-slices-panel").hidden = false;
    $("#dataset-slices").innerHTML = `
      <table class="slices-table">
        <thead><tr><th>Slice</th><th>N</th><th>Description</th></tr></thead>
        <tbody>${d.slices.map(sl => `<tr>
          <td><span class="slice-name-row">${esc(sl.name)}${sl.primary ? `<span class="primary-tag">primary</span>` : ""}</span></td>
          <td class="n">${(sl.n || 0).toLocaleString()}</td>
          <td>${esc(sl.description || sl.notes || sl.filter || "")}</td>
        </tr>`).join("")}</tbody>
      </table>`;
  } else {
    $("#dataset-slices-panel").hidden = true;
  }

  // dataset card markdown
  if (d.card_path) {
    $("#dataset-card-panel").hidden = false;
    fetch(`/api/runs/${s.run_id}/file?path=${encodeURIComponent(d.card_path)}`)
      .then(r => r.ok ? r.text() : "")
      .then(t => { $("#dataset-card-content").innerHTML = `<div class="markdown-render">${marked.parse(t || "")}</div>`; });
  } else {
    $("#dataset-card-panel").hidden = true;
  }
}

// ============================================================
// §3 ATTEMPTS — leaderboard
// ============================================================

function renderLeaderboard(s) {
  const attempts = (s.attempts || []).slice();
  const el = $("#attempts-leaderboard");
  if (!attempts.length) { el.innerHTML = `<div class="empty-line">No attempts yet.</div>`; return; }

  // sort by eval primary desc, fallback to ranking from eval_summary
  const ranking = s.eval_summary?.ranking || [];
  const rankIdx = {};
  ranking.forEach((r, i) => { rankIdx[r.attempt_id] = i; });
  attempts.sort((a, b) => {
    const ra = rankIdx[a.id] ?? 999;
    const rb = rankIdx[b.id] ?? 999;
    if (ra !== rb) return ra - rb;
    const pa = a.eval?.primary_metric?.value ?? -Infinity;
    const pb = b.eval?.primary_metric?.value ?? -Infinity;
    return pb - pa;
  });

  el.innerHTML = `<div class="lb-list">${attempts.map((a, i) => {
    const isWinner = a.id === s.best_attempt_id;
    const rank = i + 1;
    const ev = a.eval || {};
    const prim = ev.primary_metric?.value;
    const p99 = ev.latency?.p99_ms;
    const sz = ev.model_size_bytes ?? a.model_size_bytes;
    const ok = ev.meets_constraints?.overall;
    const okClass = ok === true ? "pass" : ok === false ? "fail" : "muted";
    const okGlyph = ok === true ? "✓" : ok === false ? "✗" : "—";
    return `<div class="lb-row ${isWinner ? "winner" : ""}" data-attempt-id="${esc(a.id)}">
      <div class="lb-rank">${rank}</div>
      <div class="lb-body">
        <div class="lb-name">${esc(a.name || a.id)}${isWinner ? ` <span class="winner-tag">Winner</span>` : ""}</div>
        <div class="lb-id"><span>${esc(a.id)}</span><span class="status-pill ${esc(a.status)}">${esc(a.status)}</span></div>
        ${a.method_family ? `<div class="lb-method">${esc(a.method_family)}</div>` : ""}
      </div>
      <div class="lb-stats">
        <div class="lb-stat primary"><div class="s-label">primary</div><div class="s-value">${fmt.num(prim, 4)}</div></div>
        <div class="lb-stat"><div class="s-label">p99</div><div class="s-value">${fmt.ms(p99)}</div></div>
        <div class="lb-stat"><div class="s-label">size</div><div class="s-value">${fmt.bytes(sz)}</div></div>
        <div class="lb-stat"><div class="s-label">rules</div><div class="s-value ${okClass}">${okGlyph}</div></div>
      </div>
    </div>`;
  }).join("")}</div>`;

  el.querySelectorAll(".lb-row").forEach(row => {
    row.addEventListener("click", () => showAttemptDetail(s, row.dataset.attemptId));
  });
}

async function showAttemptDetail(state, attemptId) {
  currentAttemptId = attemptId;
  const a = (state.attempts || []).find(x => x.id === attemptId);
  if (!a) return;
  $("#attempt-detail").hidden = false;
  $("#attempt-detail-title").textContent = `${a.id} — ${a.name || ""}`;
  const meta = [];
  if (a.method_family) meta.push(`<div><strong>Method</strong>${esc(a.method_family)}</div>`);
  if (a.rationale) meta.push(`<div><strong>Rationale</strong>${esc(a.rationale)}</div>`);
  if (a.notes) meta.push(`<div><strong>Notes</strong>${esc(a.notes)}</div>`);
  if (a.failure_note) meta.push(`<div><strong class="fail">Failure</strong>${esc(a.failure_note)}</div>`);

  // slices results
  if (a.eval?.slices?.length) {
    let t = `<h4>Per-slice results</h4><table><thead><tr><th>Slice</th><th>N</th><th>Macro-F1</th></tr></thead><tbody>`;
    for (const sl of a.eval.slices) {
      t += `<tr><td>${esc(sl.name)}${sl.primary ? ` <span class="primary-tag">primary</span>` : ""}</td><td>${(sl.n || 0).toLocaleString()}</td><td>${fmt.num(sl.macro_f1 ?? sl.value, 4)}</td></tr>`;
    }
    t += `</tbody></table>`;
    meta.push(t);
  }

  // constraint checks
  if (a.eval?.meets_constraints?.checks) {
    let t = `<h4>Constraint checks</h4><table><thead><tr><th>ID</th><th>Target</th><th>Actual</th><th>Pass</th></tr></thead><tbody>`;
    for (const c of a.eval.meets_constraints.checks) {
      t += `<tr><td><code>${esc(c.id)}</code></td><td>${esc(c.target)}</td><td>${esc(c.actual)}</td><td>${c.pass ? '<span class="pass">pass</span>' : '<span class="fail">fail</span>'}</td></tr>`;
    }
    t += `</tbody></table>`;
    meta.push(t);
  }

  $("#attempt-detail-meta").innerHTML = meta.join("");
  await renderTrainingChart(attemptId);
  $("#attempt-detail").scrollIntoView({behavior: "smooth", block: "nearest"});
}

async function renderTrainingChart(attemptId) {
  let lines = attemptLogCache[attemptId];
  if (!lines || lines.length === 0) {
    const res = await fetch(`/api/runs/${currentRunId}/attempts/${attemptId}/log?tail=2000`);
    const data = await res.json();
    lines = data.lines || [];
    attemptLogCache[attemptId] = lines;
  }
  if (lines.length === 0) {
    if (trainingChart) { trainingChart.destroy(); trainingChart = null; }
    return;
  }

  const xs = lines.map((l, i) => l.step ?? l.epoch ?? i);
  const trainLoss = lines.map(l => l.loss ?? l.train_loss);
  const valLoss = lines.map(l => l.val_loss);
  const valAcc = lines.map(l => l.val_acc ?? l.val_f1);

  const ctx = $("#training-chart").getContext("2d");
  if (trainingChart) trainingChart.destroy();
  const AMBER = "#e8a23a", CYAN = "#5cbbc4", MOSS = "#95ad60";
  const MUTED = "#968874", PAPER = "#f0e8d6", GRID = "#2a231d";
  const fontFamily = "'JetBrains Mono', ui-monospace, monospace";
  trainingChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: xs,
      datasets: [
        {label: "train loss", data: trainLoss, borderColor: AMBER, backgroundColor: AMBER + "1f", yAxisID: "y", spanGaps: true, pointRadius: 0, borderWidth: 1.6, tension: 0.15},
        {label: "val loss", data: valLoss, borderColor: CYAN, backgroundColor: CYAN + "1f", yAxisID: "y", spanGaps: true, pointRadius: 0, borderWidth: 1.6, tension: 0.15, borderDash: [4, 3]},
        {label: "val acc/f1", data: valAcc, borderColor: MOSS, backgroundColor: MOSS + "1f", yAxisID: "y2", spanGaps: true, pointRadius: 2, pointBackgroundColor: MOSS, borderWidth: 1.6, tension: 0.15},
      ]
    },
    options: {
      responsive: true,
      animation: false,
      interaction: {mode: "index", intersect: false},
      scales: {
        x: {title: {display: true, text: "step / epoch", color: MUTED, font: {family: fontFamily, size: 10}}, ticks: {color: MUTED, font: {family: fontFamily, size: 10}}, grid: {color: GRID, drawTicks: false}},
        y: {title: {display: true, text: "loss", color: MUTED, font: {family: fontFamily, size: 10}}, ticks: {color: MUTED, font: {family: fontFamily, size: 10}}, grid: {color: GRID, drawTicks: false}, position: "left"},
        y2: {title: {display: true, text: "acc / f1", color: MUTED, font: {family: fontFamily, size: 10}}, ticks: {color: MUTED, font: {family: fontFamily, size: 10}}, grid: {drawOnChartArea: false}, position: "right", min: 0, max: 1},
      },
      plugins: {
        legend: {labels: {color: PAPER, font: {family: fontFamily, size: 11}, boxWidth: 14, boxHeight: 2, usePointStyle: false}, position: "bottom", align: "start"},
        tooltip: {backgroundColor: "#1b1812", borderColor: "#3a3128", borderWidth: 1, titleColor: AMBER, titleFont: {family: fontFamily, size: 11, weight: "500"}, bodyColor: PAPER, bodyFont: {family: fontFamily, size: 11}, padding: 10, cornerRadius: 0, displayColors: true}
      }
    }
  });
}

// ============================================================
// §4 THOUGHTS — phase timeline
// ============================================================

const PHASE_ORDER = ["init", "planning", "research_data", "data_ready", "training", "eval", "reporting", "done"];
const PHASE_LABEL = {
  init: "Initialization",
  planning: "Planning",
  research_data: "Research + data",
  data_ready: "Data ready",
  training: "Training",
  eval: "Evaluation",
  reporting: "Reporting",
  done: "Complete",
};

function renderThoughtsTimeline(s) {
  const ths = s.thoughts || [];
  const el = $("#thoughts-timeline");
  if (!ths.length) { el.innerHTML = `<div class="empty-line">No agent activity yet.</div>`; return; }

  // group by phase, preserving order of appearance
  const groups = new Map();
  for (const t of ths) {
    const ph = t.phase || "other";
    if (!groups.has(ph)) groups.set(ph, []);
    groups.get(ph).push(t);
  }
  // sort phases by PHASE_ORDER, then by first ts of group
  const sorted = Array.from(groups.entries()).sort((a, b) => {
    const ai = PHASE_ORDER.indexOf(a[0]); const bi = PHASE_ORDER.indexOf(b[0]);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    const at = a[1][0]?.ts || ""; const bt = b[1][0]?.ts || "";
    return at.localeCompare(bt);
  });

  el.innerHTML = `<div class="timeline">${sorted.map(([phase, items]) => `
    <div class="phase-group">
      <div class="phase-head">
        <span class="phase-name">${esc(PHASE_LABEL[phase] || phase)}</span>
        <span class="phase-rule"></span>
      </div>
      <ul>${items.map(t => `<li>
        <span class="ts">${esc(fmt.time(t.ts))}</span>
        <span class="agent">${esc(t.agent || "")}</span>
        <span class="msg">${esc(t.message || "")}</span>
      </li>`).join("")}</ul>
    </div>`).join("")}</div>`;
}

// ============================================================
// §5 REPORT
// ============================================================

function renderReport(s) {
  if (s.report_path) {
    fetch(`/api/runs/${s.run_id}/file?path=${encodeURIComponent(s.report_path)}`)
      .then(r => r.ok ? r.text() : "")
      .then(t => { $("#report-content").innerHTML = `<div class="markdown-render">${marked.parse(t || "")}</div>`; });
  } else {
    $("#report-content").innerHTML = `<div class="empty-line">Report not yet written.</div>`;
  }
}

// ============================================================
// tabs + gate actions + utils
// ============================================================

function selectTab(name) {
  $$("nav#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${name}`));
}

async function gateAction(runId, gateId, action) {
  let note = null;
  if (action === "edit" || action === "reject") {
    note = prompt(`Optional note for ${action}:`);
  }
  const res = await fetch(`/api/runs/${runId}/gates/${gateId}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({action, note: note || undefined}),
  });
  if (!res.ok) alert("Failed to update gate: " + await res.text());
}

function esc(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
