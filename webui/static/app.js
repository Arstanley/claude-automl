// AutoML dashboard — vanilla JS, no build step.

let currentRunId = null;
let eventSource = null;
let trainingChart = null;
let attemptLogCache = {}; // attempt_id -> array of log lines

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const fmt = {
  pct: (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%"),
  ms: (v) => (v == null ? "—" : v.toFixed(2) + " ms"),
  bytes: (v) => {
    if (v == null) return "—";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0; let n = v;
    while (n > 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(1) + " " + u[i];
  },
  short: (s, n = 60) => s == null ? "" : (s.length > n ? s.slice(0, n - 1) + "…" : s),
};

// ---------- bootstrap ----------

window.addEventListener("DOMContentLoaded", () => {
  $("#refresh-runs").addEventListener("click", loadRuns);
  $$("nav#tabs button").forEach(b => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  loadRuns();
});

// ---------- run list ----------

async function loadRuns() {
  const res = await fetch("/api/runs");
  const runs = await res.json();
  const list = $("#run-list");
  list.innerHTML = "";
  if (runs.length === 0) {
    list.innerHTML = `<li style="padding:16px;color:var(--muted)">No runs yet.</li>`;
    return;
  }
  for (const r of runs) {
    const li = document.createElement("li");
    if (r.run_id === currentRunId) li.classList.add("active");
    li.innerHTML = `
      <div class="run-id-row">${r.run_id}</div>
      <div class="run-prompt-row">${escapeHtml(fmt.short(r.prompt, 80))}</div>
      <div class="run-meta-row">
        <span class="status-pill ${r.status}">${r.status}</span>
        <span>${r.n_attempts} attempt${r.n_attempts === 1 ? "" : "s"}</span>
      </div>`;
    li.addEventListener("click", () => selectRun(r.run_id));
    list.appendChild(li);
  }
}

// ---------- select run ----------

async function selectRun(runId) {
  currentRunId = runId;
  $$("#run-list li").forEach(li => li.classList.remove("active"));
  const target = $$("#run-list li").find(li => li.querySelector(".run-id-row")?.textContent === runId);
  if (target) target.classList.add("active");

  $("#empty-state").hidden = true;
  $("#run-view").hidden = false;

  // tear down old stream
  if (eventSource) { eventSource.close(); eventSource = null; }
  attemptLogCache = {};

  // initial state
  const state = await (await fetch(`/api/runs/${runId}`)).json();
  renderState(state);

  // subscribe to live events
  eventSource = new EventSource(`/api/runs/${runId}/events`);
  eventSource.addEventListener("state", e => {
    const s = JSON.parse(e.data);
    renderState(s);
  });
  eventSource.addEventListener("attempt_log", e => {
    const d = JSON.parse(e.data);
    if (!attemptLogCache[d.attempt_id]) attemptLogCache[d.attempt_id] = [];
    attemptLogCache[d.attempt_id].push(d.line);
    // refresh chart only if the currently expanded attempt matches
    if (currentAttemptId === d.attempt_id) renderTrainingChart(d.attempt_id);
  });
}

// ---------- render state ----------

let currentAttemptId = null;

function renderState(state) {
  if (!state || !state.run_id) return;
  $("#run-id").textContent = state.run_id;
  $("#run-prompt").textContent = state.prompt || "";
  const pill = $("#run-status-pill");
  pill.innerHTML = `<span class="status-pill ${state.status}">${state.status}</span>`;

  // gate banner (most urgent: any pending gate)
  const banner = $("#gate-banner");
  const pendingGate = (state.gates || []).find(g => g.status === "pending");
  if (pendingGate) {
    banner.innerHTML = `
      <div class="gate-banner">
        <h3>⏸ Approval needed: ${pendingGate.id}</h3>
        <p>${escapeHtml(pendingGate.summary || "")}</p>
        <div class="gate-actions">
          <button class="approve" data-gate="${pendingGate.id}" data-action="approve">Approve</button>
          <button class="reject" data-gate="${pendingGate.id}" data-action="reject">Reject</button>
          <button class="edit" data-gate="${pendingGate.id}" data-action="edit">Request edit</button>
        </div>
      </div>`;
    banner.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      gateAction(state.run_id, b.dataset.gate, b.dataset.action);
    }));
  } else {
    banner.innerHTML = "";
  }

  // constraints
  $("#constraints-json").textContent = state.constraints ? JSON.stringify(state.constraints, null, 2) : "(planner has not run yet)";

  // plan
  if (state.plan) {
    $("#plan-summary").innerHTML = `<div class="markdown-render">${marked.parse(state.plan.summary || "")}</div>`;
    renderPlanAttempts(state.plan.attempts || []);
  } else {
    $("#plan-summary").textContent = "(planner has not run yet)";
    $("#plan-attempts-table").innerHTML = "";
  }

  // research
  if (state.research) {
    $("#research-panel").hidden = false;
    $("#research-content").innerHTML = `<pre>${escapeHtml(JSON.stringify(state.research, null, 2))}</pre>`;
  } else {
    $("#research-panel").hidden = true;
  }

  // dataset
  if (state.dataset) {
    const d = state.dataset;
    const sources = (d.sources_used || []).join(", ");
    const splits = d.splits || {};
    let html = `<p><strong>Sources:</strong> ${escapeHtml(sources)}</p>`;
    html += `<p><strong>Total examples:</strong> ${d.total_examples ?? "—"}</p>`;
    if (Object.keys(splits).length) {
      html += `<table><thead><tr><th>Split</th><th>N</th><th>Path</th></tr></thead><tbody>`;
      for (const [name, s] of Object.entries(splits)) {
        html += `<tr><td>${name}</td><td>${s.n}</td><td><code>${escapeHtml(s.path)}</code></td></tr>`;
      }
      html += `</tbody></table>`;
    }
    if (d.slices && d.slices.length) {
      html += `<h4>Slices</h4><table><thead><tr><th>Name</th><th>N</th></tr></thead><tbody>`;
      for (const s of d.slices) html += `<tr><td>${s.name}</td><td>${s.n}</td></tr>`;
      html += `</tbody></table>`;
    }
    if (d.warnings && d.warnings.length) {
      html += `<h4>Warnings</h4><ul>` + d.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join("") + `</ul>`;
    }
    $("#dataset-summary").innerHTML = html;
    if (d.card_path) {
      fetch(`/api/runs/${state.run_id}/file?path=${encodeURIComponent(d.card_path)}`)
        .then(r => r.ok ? r.text() : "")
        .then(t => { $("#dataset-card-content").innerHTML = `<div class="markdown-render">${marked.parse(t || "")}</div>`; });
    }
  } else {
    $("#dataset-summary").textContent = "(data engineer has not run yet)";
    $("#dataset-card-content").textContent = "";
  }

  // attempts
  renderAttemptsTable(state);

  // thoughts
  const tlist = $("#thoughts-list");
  tlist.innerHTML = "";
  for (const t of (state.thoughts || []).slice().reverse()) {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="ts">${escapeHtml((t.ts || "").slice(11, 19))}</span>
      <span class="agent">${escapeHtml(t.agent || "")}</span>
      <span class="msg">${escapeHtml(t.message || "")}</span>`;
    tlist.appendChild(li);
  }

  // report
  if (state.report_path) {
    fetch(`/api/runs/${state.run_id}/file?path=${encodeURIComponent(state.report_path)}`)
      .then(r => r.ok ? r.text() : "")
      .then(t => { $("#report-content").innerHTML = `<div class="markdown-render">${marked.parse(t || "")}</div>`; });
  } else {
    $("#report-content").textContent = "(report not yet written)";
  }
}

function renderPlanAttempts(attempts) {
  const tbl = $("#plan-attempts-table");
  if (!attempts.length) { tbl.innerHTML = ""; return; }
  let html = `<thead><tr><th>ID</th><th>Name</th><th>Method</th><th>Rationale</th><th>Priority</th></tr></thead><tbody>`;
  for (const a of attempts) {
    html += `<tr>
      <td><code>${escapeHtml(a.id)}</code></td>
      <td>${escapeHtml(a.name)}</td>
      <td>${escapeHtml(a.method_family || "")}</td>
      <td>${escapeHtml(a.rationale || "")}</td>
      <td>${a.priority ?? ""}</td>
    </tr>`;
  }
  html += `</tbody>`;
  tbl.innerHTML = html;
}

function renderAttemptsTable(state) {
  const tbody = $("#attempts-table tbody");
  tbody.innerHTML = "";
  const attempts = state.attempts || [];
  for (const a of attempts) {
    const tr = document.createElement("tr");
    tr.className = "clickable";
    const valMetric = a.final_metrics?.val_acc ?? a.final_metrics?.val_loss;
    const evalPrim = a.eval?.primary_metric?.value;
    const p99 = a.eval?.latency?.p99_ms;
    const size = a.eval?.model_size_bytes ?? a.model_size_bytes;
    const constraintsOk = a.eval?.meets_constraints?.overall;
    tr.innerHTML = `
      <td><code>${escapeHtml(a.id)}</code></td>
      <td>${escapeHtml(a.name || "")}</td>
      <td><span class="status-pill ${a.status}">${a.status}</span></td>
      <td>${valMetric != null ? valMetric.toFixed(3) : "—"}</td>
      <td>${evalPrim != null ? evalPrim.toFixed(3) : "—"}</td>
      <td>${fmt.ms(p99)}</td>
      <td>${fmt.bytes(size)}</td>
      <td>${constraintsOk === true ? '<span class="pass">✓</span>' : constraintsOk === false ? '<span class="fail">✗</span>' : '<span class="neutral">—</span>'}</td>`;
    tr.addEventListener("click", () => showAttemptDetail(state, a.id));
    tbody.appendChild(tr);
  }
}

async function showAttemptDetail(state, attemptId) {
  currentAttemptId = attemptId;
  const a = (state.attempts || []).find(x => x.id === attemptId);
  if (!a) return;
  $("#attempt-detail").hidden = false;
  $("#attempt-detail-title").textContent = `${a.id} — ${a.name || ""}`;
  const meta = [];
  if (a.method_family) meta.push(`<strong>Method:</strong> ${escapeHtml(a.method_family)}`);
  if (a.rationale) meta.push(`<strong>Rationale:</strong> ${escapeHtml(a.rationale)}`);
  if (a.notes) meta.push(`<strong>Notes:</strong> ${escapeHtml(a.notes)}`);
  if (a.failure_note) meta.push(`<strong class="fail">Failure:</strong> ${escapeHtml(a.failure_note)}`);
  if (a.eval?.meets_constraints?.checks) {
    let t = `<h4>Constraint checks</h4><table><thead><tr><th>ID</th><th>Target</th><th>Actual</th><th>Pass</th></tr></thead><tbody>`;
    for (const c of a.eval.meets_constraints.checks) {
      t += `<tr><td>${escapeHtml(c.id)}</td><td>${escapeHtml(c.target)}</td><td>${escapeHtml(c.actual)}</td><td>${c.pass ? '<span class="pass">✓</span>' : '<span class="fail">✗</span>'}</td></tr>`;
    }
    t += `</tbody></table>`;
    meta.push(t);
  }
  $("#attempt-detail-meta").innerHTML = meta.join("<br>");

  // fetch log + render chart
  await renderTrainingChart(attemptId);
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
  trainingChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: xs,
      datasets: [
        {label: "train loss", data: trainLoss, borderColor: "#6ea8fe", backgroundColor: "#6ea8fe33", yAxisID: "y", spanGaps: true, pointRadius: 0, borderWidth: 1.5},
        {label: "val loss", data: valLoss, borderColor: "#f4a261", backgroundColor: "#f4a26133", yAxisID: "y", spanGaps: true, pointRadius: 0, borderWidth: 1.5},
        {label: "val acc/f1", data: valAcc, borderColor: "#6ab04c", backgroundColor: "#6ab04c33", yAxisID: "y2", spanGaps: true, pointRadius: 0, borderWidth: 1.5},
      ]
    },
    options: {
      responsive: true,
      animation: false,
      interaction: {mode: "index", intersect: false},
      scales: {
        x: {title: {display: true, text: "step / epoch", color: "#8a8f9c"}, ticks: {color: "#8a8f9c"}, grid: {color: "#2a2d3a"}},
        y: {title: {display: true, text: "loss", color: "#8a8f9c"}, ticks: {color: "#8a8f9c"}, grid: {color: "#2a2d3a"}, position: "left"},
        y2: {title: {display: true, text: "acc/f1", color: "#8a8f9c"}, ticks: {color: "#8a8f9c"}, grid: {drawOnChartArea: false}, position: "right", min: 0, max: 1},
      },
      plugins: {legend: {labels: {color: "#e6e6e6"}}}
    }
  });
}

// ---------- tab nav ----------

function selectTab(name) {
  $$("nav#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${name}`));
}

// ---------- gate actions ----------

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
  if (!res.ok) {
    alert("Failed to update gate: " + await res.text());
  }
  // state will refresh via SSE
}

// ---------- utils ----------

function escapeHtml(s) {
  if (s == null) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
