"use strict";

const $ = (id) => document.getElementById(id);

let pollTimer = null;
let currentJob = null;

// ---------------------------------------------------------------- backend toggle
document.querySelectorAll(".backend").forEach((el) => {
  el.addEventListener("click", () => {
    document.querySelectorAll(".backend").forEach((b) => b.classList.remove("selected"));
    el.classList.add("selected");
    el.querySelector("input").checked = true;
    const isMeshy = el.dataset.backend === "meshy";
    $("meshy-opts").classList.toggle("hidden", !isMeshy);
    $("blender-opts").classList.toggle("hidden", isMeshy);
  });
});

$("iters").addEventListener("input", (e) => { $("iters-out").value = e.target.value; });

function selectedBackend() {
  return document.querySelector('input[name="backend"]:checked').value;
}

// ---------------------------------------------------------------- generate
$("go").addEventListener("click", async () => {
  const backend = selectedBackend();
  const prompt = $("prompt").value.trim();
  const file = $("image").files[0];
  const err = $("form-error");
  err.classList.add("hidden");

  if (!prompt && !file) {
    err.textContent = "Provide a prompt, an image, or both.";
    err.classList.remove("hidden");
    return;
  }
  if (backend === "blender" && $("coder").value === $("critic").value) {
    err.textContent = "Coder and critic must be different models — a model critiquing its own output rubber-stamps it.";
    err.classList.remove("hidden");
    return;
  }

  const fd = new FormData();
  fd.append("backend", backend);
  fd.append("prompt", prompt);
  if (file) fd.append("image", file);
  fd.append("art_style", $("art_style").value);
  fd.append("lowpoly", $("lowpoly").checked);
  fd.append("refine", $("refine").checked);
  fd.append("enable_pbr", $("enable_pbr").checked);
  fd.append("iters", $("iters").value);
  fd.append("pick", $("pick").checked);
  fd.append("coder", $("coder").value);
  fd.append("critic", $("critic").value);

  $("go").disabled = true;
  $("result").classList.add("hidden");
  $("picker").classList.add("hidden");
  $("log").textContent = "Starting…";

  try {
    const res = await fetch("/api/generate", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to start the job.");
    currentJob = data.job_id;
    poll();
  } catch (e) {
    err.textContent = e.message;
    err.classList.remove("hidden");
    $("go").disabled = false;
    $("log").textContent = "Idle.";
  }
});

// ---------------------------------------------------------------- polling
function poll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    if (!currentJob) return;
    try {
      const res = await fetch(`/api/job/${currentJob}`);
      const job = await res.json();
      render(job);
      if (job.status === "running" || job.status === "awaiting_pick") poll();
      else $("go").disabled = false;
    } catch {
      poll(); // transient network hiccup; keep trying
    }
  }, 900);
}

function render(job) {
  const badge = $("status-badge");
  badge.textContent = job.status.replace("_", " ");
  badge.className = "badge " + job.status;
  badge.classList.remove("hidden");

  const log = $("log");
  const pinned = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  log.textContent = (job.log || []).join("\n") || "…";
  if (pinned) log.scrollTop = log.scrollHeight;

  if (job.status === "awaiting_pick") renderPicker(job);
  else $("picker").classList.add("hidden");

  if (job.status === "done" && job.result) {
    showResult(job);
    loadLibrary();
  }
  if (job.status === "error") {
    const err = $("form-error");
    err.textContent = job.error || "Generation failed.";
    err.classList.remove("hidden");
  }
}

// ---------------------------------------------------------------- iteration picker
function renderPicker(job) {
  const picker = $("picker");
  picker.classList.remove("hidden");

  $("sheet-wrap").innerHTML = job.contact_sheet_url
    ? `<img src="${job.contact_sheet_url}" alt="All iterations compared">`
    : "";

  $("candidates").innerHTML = (job.candidates || []).map((c) => `
    <div class="cand">
      <img src="${c.render_url}" alt="iteration ${c.iter} preview">
      <div class="n">iter ${c.iter} — ${c.n_diffs === null ? "critic n/a" : c.n_diffs + " diff(s)"}</div>
      <button data-iter="${c.iter}">Export this</button>
    </div>`).join("");

  $("candidates").querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", async () => {
      $("candidates").querySelectorAll("button").forEach((x) => (x.disabled = true));
      await fetch(`/api/job/${currentJob}/pick`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ iter: Number(b.dataset.iter) }),
      });
      $("picker").classList.add("hidden");
    });
  });
}

// ---------------------------------------------------------------- result
function showResult(job) {
  const r = job.result;
  $("result").classList.remove("hidden");
  $("viewer").src = r.glb_url;

  const bits = [`<strong>${job.backend}</strong>`];
  if (r.credits_used != null) bits.push(`${r.credits_used} credits`);
  if (r.best_iter != null) bits.push(`exported iteration ${r.best_iter}`);
  if (r.task_ids) bits.push(`tasks: ${r.task_ids.join(", ")}`);
  $("result-meta").innerHTML = bits.join(" &middot; ");

  const links = [`<a href="${r.glb_url}?download=1" download>Download GLB</a>`];
  if (r.script_url) links.push(`<a href="${r.script_url}?download=1" download>Download .py</a>`);
  if (job.contact_sheet_url) links.push(`<a href="${job.contact_sheet_url}" target="_blank">Iteration sheet</a>`);
  $("downloads").innerHTML = links.join("");
}

// ---------------------------------------------------------------- library
async function loadLibrary() {
  const box = $("library");
  try {
    const assets = await (await fetch("/api/library")).json();
    if (!assets.length) {
      box.innerHTML = '<div class="empty">No assets yet — generate one above.</div>';
      return;
    }
    box.innerHTML = assets.map((a) => `
      <div class="asset">
        <div class="row">
          <span class="tag ${a.backend}">${a.backend}</span>
          <span class="when">${new Date(a.created_at).toLocaleString()}</span>
        </div>
        <div class="prompt">${escapeHtml(a.prompt || "(untitled)")}</div>
        <div class="links">
          <button onclick="viewAsset('${a.glb_url}')">View</button>
          <a href="${a.glb_url}?download=1" download>GLB</a>
          ${a.script_url ? `<a href="${a.script_url}?download=1" download>.py</a>` : ""}
        </div>
      </div>`).join("");
  } catch {
    box.innerHTML = '<div class="empty">Could not load the library.</div>';
  }
}

window.viewAsset = (url) => {
  $("result").classList.remove("hidden");
  $("viewer").src = url;
  $("result").scrollIntoView({ behavior: "smooth", block: "center" });
};

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

loadLibrary();
