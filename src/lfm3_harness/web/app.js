// Zero-build vanilla ES module SPA for Romanaccio EIC7700X Harness.
import { MicRecorder, canRecordInBrowser, speakText, uploadAudioBlob } from "/static/audio.js";

const state = {
  workspaceId: null,
  sessionId: null,
  models: [],
  defaultModel: "lfm3-1.7b",
  capabilities: {},
  d1Info: {},
  attachedImage: null,
  streaming: false,
  ttsEngine: null,
  sessions: [],
  recorder: null,
  recording: false,
};

const $ = (id) => document.getElementById(id);

function escHtml(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderSafeMarkdown(text) {
  let out = escHtml(text);
  out = out.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  out = out.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  return out;
}

async function apiJson(path, opts = {}) {
  const resp = await fetch(path, { credentials: "same-origin", ...opts });
  if (resp.status === 401) {
    $("auth-modal").classList.remove("hidden");
    throw new Error("Authentication required");
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  return data;
}

async function checkAuthAndBoot() {
  const st = await apiJson("/api/auth/status");
  if (st.auth_enabled) {
    $("logout-btn").classList.remove("hidden");
    if (!st.authenticated) {
      $("auth-modal").classList.remove("hidden");
      return false;
    }
  }
  $("auth-modal").classList.add("hidden");
  await initData();
  return true;
}

async function initData() {
  const [wsRes, modRes, tmplRes] = await Promise.all([
    apiJson("/api/workspaces"),
    apiJson("/api/models"),
    apiJson("/api/missions/templates"),
  ]);
  state.workspaceId = wsRes.default_workspace_id;
  const ws = (wsRes.workspaces || []).find((w) => w.id === state.workspaceId);
  $("workspace-pill").textContent = ws ? ws.root_path : "-";

  state.models = modRes.models || [];
  state.defaultModel = modRes.default_model || "lfm3-1.7b";
  state.capabilities = modRes.capabilities || {};
  state.d1Info = modRes.d1 || {};
  state.ttsEngine = modRes.capabilities?.tts_engine || null;

  const actorSel = $("actor-select");
  actorSel.innerHTML = "";
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = `${m.name} (:${m.port}${m.reachable ? "" : " offline"})`;
    actorSel.appendChild(opt);
  }
  actorSel.value = state.defaultModel;
  syncActorControls();

  const tmplSel = $("mission-template-select");
  tmplSel.innerHTML = "";
  for (const t of tmplRes.templates || []) {
    const opt = document.createElement("option");
    opt.value = t.id;
    opt.textContent = `${t.name} (${t.semantic_mode})`;
    tmplSel.appendChild(opt);
  }

  $("voice-reply-label").textContent = state.ttsEngine ? `Voice (${state.ttsEngine})` : "Voice unavailable";
  $("voice-reply-toggle").disabled = !state.ttsEngine;
  const ttsSel = $("tts-engine-select");
  if (ttsSel) {
    ttsSel.innerHTML = "";
    const autoOpt = document.createElement("option");
    autoOpt.value = "auto";
    autoOpt.textContent = state.ttsEngine ? `TTS: Auto (${state.ttsEngine})` : "TTS: Auto";
    ttsSel.appendChild(autoOpt);
    for (const eng of state.capabilities.tts_engines || []) {
      const o = document.createElement("option");
      o.value = eng;
      o.textContent = eng;
      ttsSel.appendChild(o);
    }
    ttsSel.disabled = !state.ttsEngine;
  }
  $("hud-tts-badge").textContent = state.ttsEngine ? `TTS: ${state.ttsEngine}` : "TTS: off";
  if (state.capabilities.voice_asr) {
    const asrOk = state.capabilities.asr_reachable !== false;
    $("hud-asr-badge").textContent = asrOk ? `ASR: ${state.capabilities.asr_engine || "lfm3-audio"}` : "ASR: unreachable";
    $("hud-asr-badge").className = `badge ${asrOk ? "ok" : "warn"}`;
  } else {
    $("hud-asr-badge").textContent = "ASR: off";
    $("hud-asr-badge").className = "badge subtle";
  }
  if (state.capabilities.vision) {
    $("image-upload-btn").classList.remove("hidden");
    $("hud-vision-badge").textContent = state.capabilities.vision_reachable ? `Vision: ${state.capabilities.vision_model || "on"}` : "Vision: unreachable";
    $("hud-vision-badge").className = `badge ${state.capabilities.vision_reachable ? "ok" : "warn"}`;
  } else {
    $("image-upload-btn").classList.add("hidden");
    $("hud-vision-badge").textContent = "Vision: off";
    $("hud-vision-badge").className = "badge subtle";
  }
  if (state.d1Info.separate_daemon) {
    $("hud-d1-badge").textContent = `d1 :${state.d1Info.port}${state.d1Info.reachable ? "" : " offline"}`;
    $("hud-d1-badge").className = `badge ${state.d1Info.reachable ? "ok" : "danger"}`;
  } else {
    $("hud-d1-badge").textContent = `d1: shared`;
    $("hud-d1-badge").className = "badge subtle";
  }
  if (!canRecordInBrowser()) {
    $("mic-btn").title = "Mic requires HTTPS/localhost Secure Context; use +Audio on plain HTTP";
  }

  await Promise.all([refreshSessions(), refreshFiles(), refreshMissions(), refreshHud()]);
}

function syncActorControls() {
  const chosen = state.models.find((m) => m.name === ($("actor-select").value || state.defaultModel));
  const supportsThink = chosen ? Boolean(chosen.supports_thinking) : true;
  $("think-toggle").disabled = !supportsThink;
  if (!supportsThink) {
    $("think-toggle").checked = false;
  }
}

function renderAttachedImage() {
  const bar = $("attached-image-bar");
  if (!state.attachedImage) {
    bar.classList.add("hidden");
    bar.innerHTML = "";
    return;
  }
  const im = state.attachedImage;
  bar.classList.remove("hidden");
  bar.innerHTML = `<span>Image attached: <strong class="mono">${escHtml(im.filename)}</strong> (${im.width}x${im.height}, ${im.num_tokens} vis tok)</span>`;
  const rm = document.createElement("button");
  rm.className = "btn tiny";
  rm.textContent = "Remove";
  rm.onclick = () => {
    state.attachedImage = null;
    renderAttachedImage();
  };
  bar.appendChild(rm);
}

async function refreshFiles() {
  if (!state.workspaceId) return;
  try {
    const data = await apiJson(`/api/workspaces/${state.workspaceId}/files`);
    const list = $("workspace-files");
    list.innerHTML = "";
    for (const ent of data.entries || []) {
      const div = document.createElement("div");
      div.className = "file-item";
      div.textContent = `${ent.type === "dir" ? "[D]" : "[F]"} ${ent.name}`;
      list.appendChild(div);
    }
  } catch (err) {
    $("workspace-files").textContent = String(err.message || err);
  }
}

async function refreshSessions() {
  const data = await apiJson("/api/sessions");
  state.sessions = data.sessions || [];
  if (!state.sessionId && state.sessions.length > 0) {
    state.sessionId = state.sessions[0].id;
  }
  if (!state.sessionId) {
    const created = await apiJson("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "Workspace Session", model_name: $("actor-select").value || state.defaultModel }),
    });
    state.sessionId = created.session.id;
    state.sessions.unshift(created.session);
  }
  renderSessionList();
  await loadSession(state.sessionId);
}

function renderSessionList() {
  const el = $("session-list");
  el.innerHTML = "";
  for (const s of state.sessions) {
    const item = document.createElement("div");
    item.className = `session-item ${s.id === state.sessionId ? "active" : ""}`;
    item.onclick = () => loadSession(s.id);
    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = s.title || s.id;
    const tag = document.createElement("span");
    tag.className = "muted";
    tag.style.fontSize = "11px";
    tag.textContent = s.model_name.startsWith("lfm2") ? "8B" : "1.7B";
    item.append(title, tag);
    el.appendChild(item);
  }
}

async function loadSession(sid) {
  state.sessionId = sid;
  renderSessionList();
  const data = await apiJson(`/api/sessions/${sid}`);
  if (data.session?.model_name) {
    $("actor-select").value = data.session.model_name;
    syncActorControls();
  }
  const tl = $("chat-timeline");
  tl.innerHTML = "";
  for (const m of data.messages || []) {
    if (m.role === "tool") continue;
    appendMessageCard(m);
  }
  renderPendingActions(data.actions || []);
  tl.scrollTop = tl.scrollHeight;
}

function appendMessageCard(msg) {
  const card = document.createElement("div");
  card.className = `msg ${msg.role === "user" ? "user" : "assistant"}`;
  const header = document.createElement("div");
  header.className = "msg-header";

  const roleTag = document.createElement("strong");
  roleTag.textContent = msg.role === "user" ? "User" : "Assistant";
  header.appendChild(roleTag);

  let telemetry = {};
  try {
    telemetry = typeof msg.telemetry_json === "string" ? JSON.parse(msg.telemetry_json || "{}") : msg.telemetry || {};
  } catch {
    telemetry = {};
  }
  if (telemetry.model) {
    const mb = document.createElement("span");
    mb.className = "badge subtle";
    mb.textContent = telemetry.model;
    header.appendChild(mb);
  }
  if (telemetry.cache_status) {
    const cb = document.createElement("span");
    cb.className = `badge ${telemetry.cache_status.startsWith("HIT") ? "ok" : "subtle"}`;
    cb.textContent = `${telemetry.cache_status} (${telemetry.cached_tokens || 0}/${telemetry.prompt_tokens || 0}, ${Number(telemetry.cache_restore_ms || 0).toFixed(2)}ms)`;
    header.appendChild(cb);
  }
  if (msg.role === "assistant" && state.ttsEngine && msg.content) {
    const spBtn = document.createElement("button");
    spBtn.className = "btn tiny";
    spBtn.textContent = "Speak";
    spBtn.onclick = () => speakText(msg.content).catch((e) => ($("audio-status-bar").textContent = e.message));
    header.appendChild(spBtn);
  }

  card.appendChild(header);

  if (msg.thinking) {
    const det = document.createElement("details");
    det.className = "think-box";
    const sum = document.createElement("summary");
    sum.textContent = "Thought process";
    const body = document.createElement("div");
    body.className = "tool-body";
    body.textContent = msg.thinking;
    det.append(sum, body);
    card.appendChild(det);
  }

  const contentEl = document.createElement("div");
  contentEl.className = "msg-body";
  contentEl.innerHTML = renderSafeMarkdown(msg.content || "");
  card.appendChild(contentEl);

  $("chat-timeline").appendChild(card);
  return { card, header, contentEl };
}

function renderDiffHtml(diffText) {
  return String(diffText || "")
    .split("\n")
    .map((line) => {
      const e = escHtml(line);
      if (line.startsWith("+") && !line.startsWith("+++")) return `<span class="diff-add">${e}</span>`;
      if (line.startsWith("-") && !line.startsWith("---")) return `<span class="diff-del">${e}</span>`;
      return e;
    })
    .join("\n");
}

function renderPendingActions(actions) {
  const dock = $("actions-dock");
  dock.innerHTML = "";
  for (const act of actions) {
    if (act.status !== "pending") continue;
    const card = document.createElement("div");
    card.className = "approval-card";
    card.innerHTML = `
      <div class="approval-header">
        <strong>Pending Write Approval: <span class="mono">${escHtml(act.rel_path)}</span> (${escHtml(act.tool_name)})</strong>
        <span class="badge warn">expires in ~${act.expires_in_s || 600}s</span>
      </div>
      <div class="diff-pre">${renderDiffHtml(act.diff_text)}</div>
      <div class="approval-actions">
        <button class="btn ok small" data-allow="${escHtml(act.id)}">Allow once</button>
        <button class="btn danger small" data-deny="${escHtml(act.id)}">Deny</button>
      </div>
    `;
    card.querySelector("[data-allow]").onclick = async () => {
      await apiJson(`/api/actions/${act.id}/allow`, { method: "POST" });
      await Promise.all([loadSession(state.sessionId), refreshFiles()]);
    };
    card.querySelector("[data-deny]").onclick = async () => {
      await apiJson(`/api/actions/${act.id}/deny`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      await loadSession(state.sessionId);
    };
    dock.appendChild(card);
  }
}

async function sendChatMessage() {
  const text = $("chat-input").value.trim();
  if (!text || !state.sessionId || state.streaming) return;
  const attachedImg = state.attachedImage;
  state.attachedImage = null;
  renderAttachedImage();
  $("chat-input").value = "";
  appendMessageCard({
    role: "user",
    content: attachedImg ? `[Image attached: ${attachedImg.filename}]\n${text}` : text,
  });

  state.streaming = true;
  $("send-btn").disabled = true;
  $("stop-btn").classList.remove("hidden");

  const card = document.createElement("div");
  card.className = "msg assistant";
  const header = document.createElement("div");
  header.className = "msg-header";
  header.innerHTML = `<strong>Assistant</strong> <span class="badge subtle">streaming...</span>`;
  const thinkDet = document.createElement("details");
  thinkDet.className = "think-box hidden";
  thinkDet.innerHTML = `<summary>Thinking...</summary><div class="tool-body"></div>`;
  const toolWrap = document.createElement("div");
  const bodyEl = document.createElement("div");
  bodyEl.className = "msg-body";
  card.append(header, thinkDet, toolWrap, bodyEl);
  const tl = $("chat-timeline");
  tl.appendChild(card);
  tl.scrollTop = tl.scrollHeight;

  let accText = "";
  let accThink = "";

  try {
    const reqBody = {
      content: text,
      model_name: $("actor-select").value || state.defaultModel,
      enable_thinking: $("think-toggle").checked,
    };
    if (attachedImg?.image_id) {
      reqBody.image_id = attachedImg.image_id;
    }
    const resp = await fetch(`/api/sessions/${state.sessionId}/messages`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    const handleEv = (evName, payload) => {
      if (evName === "route") {
        const pill = $("route-status-pill");
        pill.textContent = `d1: ${payload.intent} (${Math.round((payload.confidence || 0) * 100)}%, ${payload.source})`;
        pill.className = `badge ${payload.source === "decision" ? "ok" : "warn"}`;
      } else if (evName === "prefill_done") {
        header.innerHTML = `<strong>Assistant</strong> <span class="badge ${String(payload.cache_status || "").startsWith("HIT") ? "ok" : "subtle"}">${escHtml(payload.cache_status || "MISS")} ${payload.cached_tokens || 0}/${payload.prompt_tokens || 0} (${Number(payload.cache_restore_ms || 0).toFixed(2)}ms)</span>`;
      } else if (evName === "thinking") {
        thinkDet.classList.remove("hidden");
        accThink += payload.text || "";
        thinkDet.querySelector(".tool-body").textContent = accThink;
      } else if (evName === "token") {
        accText += payload.text || "";
        bodyEl.textContent = accText;
        tl.scrollTop = tl.scrollHeight;
      } else if (evName === "tool_call") {
        const d = document.createElement("details");
        d.className = "tool-box";
        d.innerHTML = `<summary>Tool call: <span class="mono">${escHtml(payload.name)}</span></summary><div class="tool-body">${escHtml(JSON.stringify(payload.arguments, null, 2))}</div>`;
        toolWrap.appendChild(d);
      } else if (evName === "tool_result") {
        const d = document.createElement("details");
        d.className = "tool-box";
        d.innerHTML = `<summary>Tool result: <span class="mono">${escHtml(payload.name)}</span> (${payload.ok ? "ok" : "error"})</summary><div class="tool-body">${escHtml(payload.result)}</div>`;
        toolWrap.appendChild(d);
      } else if (evName === "evidence_gate" && payload.advisory) {
        const b = document.createElement("div");
        b.className = "advisory-banner";
        b.textContent = payload.advisory;
        toolWrap.appendChild(b);
      } else if (evName === "write_gate_hold") {
        const b = document.createElement("div");
        b.className = "advisory-banner";
        b.textContent = payload.revision_note || "Write held by WriteGate";
        toolWrap.appendChild(b);
      } else if (evName === "approval_card" && payload.action) {
        renderPendingActions([payload.action]);
      } else if (evName === "error") {
        bodyEl.innerHTML += `<div class="error-text">${escHtml(payload.message)}</div>`;
      } else if (evName === "done") {
        bodyEl.innerHTML = renderSafeMarkdown(accText || (payload.status === "awaiting_approval" ? "Proposed file modification staged for approval." : ""));
        if ($("voice-reply-toggle").checked && state.ttsEngine && accText) {
          const eng = $("tts-engine-select")?.value || "auto";
          speakText(accText, { engine: eng }).catch(() => {});
        }
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let evName = "message";
        let dataLine = "{}";
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) evName = line.slice(7).trim();
          else if (line.startsWith("data: ")) dataLine = line.slice(6);
        }
        try {
          handleEv(evName, JSON.parse(dataLine));
        } catch {
          // ignore malformed keepalive
        }
      }
    }
  } catch (err) {
    bodyEl.innerHTML += `<div class="error-text">${escHtml(err.message)}</div>`;
  } finally {
    state.streaming = false;
    $("send-btn").disabled = false;
    $("stop-btn").classList.add("hidden");
    await Promise.all([loadSession(state.sessionId), refreshHud()]);
  }
}

async function refreshMissions() {
  const data = await apiJson("/api/missions");
  const list = $("mission-list");
  list.innerHTML = "";
  for (const m of data.missions || []) {
    let res = {};
    try {
      res = JSON.parse(m.result_json || "{}");
    } catch {
      res = {};
    }
    const checks = res.checks || [];
    const passedChecks = checks.filter((c) => c.passed).length;
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="section-row">
        <strong class="mono">${escHtml(m.id)}</strong>
        <span class="badge ${m.status === "passed" ? "ok" : m.status === "running" ? "warn" : "danger"}">${escHtml(m.status)} (attempt ${m.attempt}/${m.max_attempts})</span>
      </div>
      <div class="muted" style="font-size:12px;margin-top:4px;">spec SHA-256: <span class="mono">${escHtml((m.spec_sha256 || "").slice(0, 16))}</span>${checks.length ? ` | checks: ${passedChecks}/${checks.length}` : ""}</div>
    `;
    list.appendChild(card);
  }
}

async function startMission() {
  const tmpl = $("mission-template-select").value || "desk_task";
  const model = $("actor-select").value || state.defaultModel;
  const started = await apiJson("/api/missions/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ template: tmpl, model_name: model }),
  });
  const liveBox = $("mission-live-box");
  liveBox.classList.remove("hidden");
  liveBox.innerHTML = `<strong>Running ${escHtml(started.run_id)}</strong> (spec <span class="mono">${escHtml(started.spec_sha256.slice(0, 12))}</span>)<div id="mission-live-rounds" class="tool-body"></div>`;
  const roundsEl = $("mission-live-rounds");

  const resp = await fetch(`/api/missions/${started.run_id}/stream`, { credentials: "same-origin" });
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const blk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let evName = "message";
      let dataLine = "{}";
      for (const line of blk.split("\n")) {
        if (line.startsWith("event: ")) evName = line.slice(7).trim();
        else if (line.startsWith("data: ")) dataLine = line.slice(6);
      }
      try {
        const d = JSON.parse(dataLine);
        if (evName === "round") {
          roundsEl.textContent += `r${d.round_num} [${d.phase}] cached=${d.cached_tokens}/${d.prompt_tokens} (${d.cache_status} ${Number(d.cache_restore_ms || 0).toFixed(2)}ms) calls=${JSON.stringify(d.tool_calls)}\n`;
        } else if (evName === "mission_done") {
          roundsEl.textContent += `DONE status=${d.status} checks=${(d.checks || []).filter((c) => c.passed).length}/${(d.checks || []).length}\n`;
        }
      } catch {
        // ignore
      }
    }
  }
  await Promise.all([refreshMissions(), refreshHud()]);
}

async function refreshHud() {
  try {
    const [cacheRes, sysRes] = await Promise.all([apiJson("/api/cache/stats"), apiJson("/api/system/health")]);
    const cards = $("cache-cards");
    cards.innerHTML = "";
    const models = cacheRes.models || {};
    const summaryParts = [];
    for (const [name, st] of Object.entries(models)) {
      const c = document.createElement("div");
      c.className = "card";
      if (!st.reachable) {
        c.innerHTML = `<div class="section-row"><strong>${escHtml(name)}</strong><span class="badge danger">offline</span></div>`;
      } else {
        const mb = ((st.disk_bytes_used || 0) / (1024 * 1024)).toFixed(1);
        summaryParts.push(`${name}: ${st.resident_blocks || 0}b (${mb}M)`);
        c.innerHTML = `
          <div class="section-row"><strong>${escHtml(name)}</strong><span class="badge ok">online</span></div>
          <div class="tool-body">resident=${st.resident_blocks || 0} (pinned=${st.pinned_blocks || 0})
disk=${mb} MiB | tier0_hits=${st.tier0_hits || 0} (live_tok=${st.tier0_live_tokens || 0})
tokens_saved=${st.total_tokens_saved || 0} | lookups=${st.total_lookups || 0}</div>
        `;
      }
      cards.appendChild(c);
    }
    $("hud-cache-badge").textContent = summaryParts.length ? summaryParts.join(" | ") : "Cache: offline";
    const f0 = sysRes.npu_freq_hz?.die0 ? `${(sysRes.npu_freq_hz.die0 / 1e9).toFixed(2)}G` : "-";
    const f1 = sysRes.npu_freq_hz?.die1 ? `${(sysRes.npu_freq_hz.die1 / 1e9).toFixed(2)}G` : "-";
    $("hud-npu-badge").textContent = `NPU: ${f0}/${f1} | RSS ${sysRes.rss_mib} MiB`;
    $("system-health-card").innerHTML = `<strong>Host / Board Telemetry</strong><div class="tool-body">${escHtml(JSON.stringify(sysRes, null, 2))}</div>`;
  } catch {
    // ignore transient poll failures
  }
}

function wireEvents() {
  for (const tabBtn of document.querySelectorAll(".tab")) {
    tabBtn.onclick = () => {
      for (const b of document.querySelectorAll(".tab")) b.classList.remove("active");
      for (const p of document.querySelectorAll(".tab-panel")) p.classList.remove("active");
      tabBtn.classList.add("active");
      $(`tab-${tabBtn.dataset.tab}`).classList.add("active");
    };
  }

  $("auth-login-btn").onclick = async () => {
    try {
      await apiJson("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: $("auth-token-input").value }),
      });
      $("auth-error").textContent = "";
      await checkAuthAndBoot();
    } catch (err) {
      $("auth-error").textContent = err.message;
    }
  };
  $("logout-btn").onclick = async () => {
    await apiJson("/api/auth/logout", { method: "POST" });
    location.reload();
  };

  $("new-session-btn").onclick = async () => {
    const res = await apiJson("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New Chat", model_name: $("actor-select").value || state.defaultModel }),
    });
    state.sessionId = res.session.id;
    await refreshSessions();
  };
  $("actor-select").onchange = async () => {
    syncActorControls();
    const chosen = $("actor-select").value;
    if (state.sessionId && chosen) {
      await apiJson(`/api/sessions/${state.sessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: chosen }),
      }).catch(() => {});
      const row = state.sessions.find((s) => s.id === state.sessionId);
      if (row) row.model_name = chosen;
      renderSessionList();
    }
  };
  $("image-file-input").onchange = async (ev) => {
    const f = ev.target.files?.[0];
    if (!f) return;
    $("audio-status-bar").textContent = `Encoding image ${f.name}...`;
    try {
      const fd = new FormData();
      fd.append("file", f, f.name);
      const res = await apiJson(`/api/images/upload?filename=${encodeURIComponent(f.name)}`, {
        method: "POST",
        body: fd,
      });
      state.attachedImage = { image_id: res.image_id, ...res.metadata };
      renderAttachedImage();
      const visActor = state.models.find((m) => m.supports_vision);
      if (visActor && $("actor-select").value !== visActor.name) {
        $("actor-select").value = visActor.name;
        $("actor-select").onchange();
      }
      $("audio-status-bar").textContent = `Vision encoded ${res.metadata.filename}: ${res.metadata.width}x${res.metadata.height} -> ${res.metadata.num_tokens} tokens`;
    } catch (err) {
      $("audio-status-bar").textContent = `Image error: ${err.message}`;
    } finally {
      ev.target.value = "";
    }
  };
  $("refresh-files-btn").onclick = refreshFiles;
  $("send-btn").onclick = sendChatMessage;
  $("stop-btn").onclick = async () => {
    if (state.sessionId) {
      await apiJson(`/api/sessions/${state.sessionId}/stop`, { method: "POST" }).catch(() => {});
    }
  };
  $("chat-input").onkeydown = (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      sendChatMessage();
    }
  };

  $("audio-file-input").onchange = async (ev) => {
    const f = ev.target.files?.[0];
    if (!f) return;
    $("audio-status-bar").textContent = `Uploading ${f.name}...`;
    try {
      const res = await uploadAudioBlob(f, f.name);
      const m = res.metadata;
      const tr = (res.transcript || "").trim();
      $("audio-status-bar").textContent =
        `Normalized ${m.filename}: ${m.duration_s}s, ${m.sample_rate}Hz, ${m.channels}ch` +
        (tr ? ` -> transcribed (${res.asr_engine || "asr"})` : state.capabilities.voice_asr ? " (no speech detected)" : " (ASR disabled)");
      if (tr) {
        $("chat-input").value = ($("chat-input").value ? $("chat-input").value + " " : "") + tr;
      }
    } catch (err) {
      $("audio-status-bar").textContent = `Audio error: ${err.message}`;
    } finally {
      ev.target.value = "";
    }
  };

  $("mic-btn").onclick = async () => {
    if (!state.recording) {
      try {
        state.recorder = new MicRecorder();
        await state.recorder.start();
        state.recording = true;
        $("mic-btn").textContent = "Stop Mic";
        $("mic-btn").classList.add("danger");
        $("audio-status-bar").textContent = "Recording 16kHz audio...";
      } catch (err) {
        $("audio-status-bar").textContent = err.message;
      }
    } else {
      state.recording = false;
      $("mic-btn").textContent = "Mic";
      $("mic-btn").classList.remove("danger");
      try {
        const wav = await state.recorder.stop();
        const res = await uploadAudioBlob(wav, "mic.wav");
        const tr = (res.transcript || "").trim();
        $("audio-status-bar").textContent =
          `Mic clip ready: ${res.metadata.duration_s}s @ ${res.metadata.sample_rate}Hz` +
          (tr ? ` -> transcribed (${res.asr_engine || "asr"})` : state.capabilities.voice_asr ? " (no speech detected)" : " (ASR disabled)");
        if (tr) {
          $("chat-input").value = ($("chat-input").value ? $("chat-input").value + " " : "") + tr;
        }
      } catch (err) {
        $("audio-status-bar").textContent = `Mic error: ${err.message}`;
      }
    }
  };

  $("run-mission-btn").onclick = () => startMission().catch((e) => ($("mission-live-box").textContent = e.message));
  $("refresh-hud-btn").onclick = refreshHud;
  $("clear-unpinned-btn").onclick = async () => {
    await apiJson("/api/cache/clear", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ include_pinned: false }) });
    await refreshHud();
  };
  $("clear-all-btn").onclick = async () => {
    await apiJson("/api/cache/clear", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ include_pinned: true }) });
    await refreshHud();
  };
}

wireEvents();
checkAuthAndBoot().catch(() => {});
