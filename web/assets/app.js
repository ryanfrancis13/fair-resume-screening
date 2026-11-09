const form = document.querySelector("#screen-form");
const fileInput = document.querySelector("#resume");
const jdInput = document.querySelector("#job-description");
const jdHint = document.querySelector("#jd-hint");
const uploadBox = document.querySelector("#upload-box");
const uploadTitle = document.querySelector("#upload-title");
const uploadDetail = document.querySelector("#upload-detail");
const formError = document.querySelector("#form-error");
const button = document.querySelector("#submit-button");
const result = document.querySelector("#result");

function setFormError(msg) {
  if (!msg) {
    formError.textContent = "";
    formError.classList.add("hidden");
  } else {
    formError.textContent = msg;
    formError.classList.remove("hidden");
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    uploadTitle.textContent = file.name;
    uploadDetail.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · Ready to evaluate`;
    uploadBox.classList.add("has-file");
    setFormError("");
  } else {
    uploadBox.classList.remove("has-file");
  }
});

jdInput.addEventListener("input", () => {
  const len = jdInput.value.trim().length;
  jdHint.textContent = len >= 40 ? `${len} chars` : `${len}/40 min chars`;
  jdInput.setCustomValidity("");
  setFormError("");
});

jdInput.addEventListener("blur", () => {
  const len = jdInput.value.trim().length;
  if (len > 0 && len < 40) {
    jdInput.setCustomValidity("Job description must be at least 40 characters.");
    setFormError("Job description must be at least 40 characters.");
  }
});

["dragenter", "dragover"].forEach((evt) =>
  uploadBox.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadBox.classList.add("dragging");
  })
);

["dragleave", "drop"].forEach((evt) =>
  uploadBox.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadBox.classList.remove("dragging");
  })
);

uploadBox.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[ch]));
}

function cleanText(value) {
  return String(value ?? "")
    .replaceAll("ﬁ", "fi")
    .replaceAll("ﬂ", "fl")
    .replace(/\s+/g, " ")
    .trim();
}

function renderResult(data) {
  const score = Number(data.overall_score);
  const roundedScore = Number.isFinite(score) ? Math.round(score) : 0;
  const confidence = Number(data.confidence);
  const ringColor = roundedScore >= 75 ? "var(--emerald)" : roundedScore >= 55 ? "var(--accent)" : "var(--amber)";

  const strengths = (data.strengths || []).filter(Boolean);
  const gaps = [
    ...(data.missing_requirements || []),
    ...(data.weaknesses || []),
  ].filter(Boolean);

  const insightsHtml =
    strengths.length || gaps.length
      ? `<div class="insights-grid">
          ${
            strengths.length
              ? `<section class="insight-box strengths">
                  <h3>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    Key Strengths
                  </h3>
                  <ul class="insight-list">${strengths.map((s) => `<li>${escapeHtml(cleanText(s))}</li>`).join("")}</ul>
                </section>`
              : ""
          }
          ${
            gaps.length
              ? `<section class="insight-box gaps">
                  <h3>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    Gaps &amp; Missing Requirements
                  </h3>
                  <ul class="insight-list">${gaps.slice(0, 5).map((g) => `<li>${escapeHtml(cleanText(g))}</li>`).join("")}</ul>
                </section>`
              : ""
          }
        </div>`
      : "";

  const dimensions = (data.dimensions || [])
    .map((item) => {
      const evidence = (item.evidence || [])
        .map(
          (q) => `
        <div class="evidence-item">
          <span class="quote">"${escapeHtml(cleanText(q.quote))}"</span>
        </div>`
        )
        .join("");
      return `<article class="dimension">
        <div class="dimension-head">
          <span class="dimension-title">${escapeHtml(item.name.replaceAll("_", " "))}</span>
          <span class="dimension-score">${Math.round(item.score)}/100</span>
        </div>
        <div class="bar"><i style="width:${Math.max(0, Math.min(100, item.score))}%"></i></div>
        <p>${escapeHtml(cleanText(item.rationale))}</p>
        ${
          evidence
            ? `<div class="evidence-block">
                <span class="evidence-label">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>
                  Verified Resume Evidence
                </span>
                <div class="evidence">${evidence}</div>
              </div>`
            : ""
        }
      </article>`;
    })
    .join("");

  result.innerHTML = `
    ${data.fallback_used ? `<div class="fallback"><strong>Local Scorer Used:</strong> ${escapeHtml(data.fallback_reason)}</div>` : ""}
    <div class="panel overview-card">
      <div class="score-ring-wrap">
        <div class="score-ring" style="--pct:${Math.max(0, Math.min(100, roundedScore))}; --ring-color:${ringColor}">
          <div class="score-ring-inner">
            <span class="score-value">${Number.isFinite(score) ? roundedScore : "-"}</span>
            <span class="score-max">/ 100</span>
          </div>
        </div>
        <span class="score-caption">Composite Fit</span>
      </div>
      <div class="overview-body">
        <h2 class="overview-title">Evaluation Summary</h2>
        <p class="summary">${escapeHtml(cleanText(data.summary))}</p>
        <div class="meta-pills">
          <span class="meta-pill">Model: <strong>${escapeHtml(data.model)}</strong></span>
          <span class="meta-pill">Evidence Grounding: <strong>${Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : "-"}</strong></span>
          <span class="meta-pill">Latency: <strong>${data.latency_ms ? `${Math.round(data.latency_ms)} ms` : "-"}</strong></span>
        </div>
      </div>
    </div>

    ${insightsHtml}

    <div class="panel">
      <div class="section-header">
        <h2>Rubric Breakdown</h2>
        <span>Weighted across 6 dimensions</span>
      </div>
      <div class="dimensions">${dimensions}</div>
    </div>

    <div class="panel" id="baseline-panel">
      <div class="second-opinion">
        <div>
          <strong>Rule-Based Baseline Comparison</strong>
          <p>Run the deterministic lexical rubric scorer on the same inputs to compare scores.</p>
        </div>
        <button class="secondary-action" id="offline-button" type="button">Compare Baseline</button>
      </div>
    </div>`;
  result.classList.remove("hidden");
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

function submissionData() {
  const body = new FormData();
  body.append("resume", fileInput.files[0], fileInput.files[0].name);
  body.append("job_description", jdInput.value);
  return body;
}

function renderOfflineComparison(primary, offline) {
  const delta = Number(offline.overall_score) - Number(primary.overall_score);
  const deltaLabel = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts`;
  const comparison = document.createElement("div");
  comparison.className = "offline-comparison";
  comparison.innerHTML = `
    <div class="comparison-grid">
      <div><span>Primary Score (${escapeHtml(primary.model)})</span><strong>${Math.round(primary.overall_score)} / 100</strong></div>
      <div><span>Rule-Based Baseline</span><strong>${Math.round(offline.overall_score)} / 100</strong></div>
      <div><span>Score Delta</span><strong>${escapeHtml(deltaLabel)}</strong></div>
    </div>
    <p class="comparison-note">Note: The baseline is computed locally via exact keyword and tenure rules with zero LLM calls, so it may under-score transferable or differently phrased experience.</p>`;
  const baselinePanel = document.querySelector("#baseline-panel");
  const existing = baselinePanel.querySelector(".offline-comparison");
  if (existing) existing.replaceWith(comparison);
  else baselinePanel.appendChild(comparison);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files[0]) {
    setFormError("Please select a resume file to evaluate.");
    return;
  }
  if (jdInput.value.trim().length < 40) {
    setFormError("Job description must be at least 40 characters.");
    jdInput.focus();
    return;
  }
  setFormError("");
  button.disabled = true;
  button.querySelector("span").textContent = "Evaluating...";
  result.classList.add("hidden");
  try {
    const response = await fetch("/api/screen", { method: "POST", body: submissionData() });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Evaluation failed.");
    renderResult(data);

    const offlineButton = document.querySelector("#offline-button");
    offlineButton.addEventListener("click", async () => {
      offlineButton.disabled = true;
      offlineButton.textContent = "Running...";
      try {
        const offlineResponse = await fetch("/api/screen/offline", {
          method: "POST",
          body: submissionData(),
        });
        const offlineData = await offlineResponse.json();
        if (!offlineResponse.ok) throw new Error(offlineData.detail || "Baseline comparison failed.");
        renderOfflineComparison(data, offlineData);
      } catch (error) {
        setFormError(error.message);
      } finally {
        offlineButton.disabled = false;
        offlineButton.textContent = "Compare Baseline";
      }
    });
  } catch (error) {
    result.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
    result.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Evaluate Candidate";
  }
});
