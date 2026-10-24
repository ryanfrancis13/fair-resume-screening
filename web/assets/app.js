const form = document.querySelector("#screen-form");
const fileInput = document.querySelector("#resume");
const uploadBox = document.querySelector("#upload-box");
const uploadTitle = document.querySelector("#upload-title");
const uploadDetail = document.querySelector("#upload-detail");
const button = document.querySelector("#submit-button");
const result = document.querySelector("#result");

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    uploadTitle.textContent = file.name;
    uploadDetail.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB selected`;
  }
});
["dragenter", "dragover"].forEach((eventName) => uploadBox.addEventListener(eventName, (event) => {
  event.preventDefault();
  uploadBox.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => uploadBox.addEventListener(eventName, (event) => {
  event.preventDefault();
  uploadBox.classList.remove("dragging");
}));
uploadBox.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    fileInput.files = event.dataTransfer.files;
    fileInput.dispatchEvent(new Event("change"));
  }
});

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[char]));
}

function cleanDisplayText(value) {
  return String(value ?? "")
    .replaceAll("ﬁ", "fi")
    .replaceAll("ﬂ", "fl")
    .replace(/\s+/g, " ")
    .trim();
}

function renderResult(data) {
  const score = Number(data.overall_score);
  const confidence = Number(data.confidence);
  const dimensions = (data.dimensions || []).map((item) => {
    const evidence = (item.evidence || []).map((quote) => `
      <div class="evidence-item">
        <span class="quote">“${escapeHtml(cleanDisplayText(quote.quote))}”</span>
        <span class="quote-support">Supports: ${escapeHtml(cleanDisplayText(quote.supports || "This dimension"))}</span>
      </div>`).join("");
    return `<article class="dimension">
      <div class="dimension-head">
        <span>${escapeHtml(item.name.replaceAll("_", " "))}</span>
        <strong>${Math.round(item.score)}/100</strong>
      </div>
      <div class="bar"><i style="width:${Math.max(0, Math.min(100, item.score))}%"></i></div>
      <p>${escapeHtml(cleanDisplayText(item.rationale))}</p>
      ${evidence ? `<div class="evidence-block"><span class="evidence-label">Resume evidence</span><div class="evidence">${evidence}</div></div>` : ""}
    </article>`;
  }).join("");
  result.innerHTML = `
    <div class="result-top">
      <div>
        <p class="result-label">Screening result</p>
        <div class="score">${Number.isFinite(score) ? Math.round(score) : "—"}<small>/100</small></div>
      </div>
      <div class="result-note">Score for human review<br><strong>No automatic decision</strong></div>
    </div>
    ${data.fallback_used ? `<div class="fallback"><strong>Offline fallback used.</strong> Gemini was unavailable, so this result was generated locally. Reason: ${escapeHtml(data.fallback_reason)}</div>` : ""}
    <div class="result-meta">
      <div><span>Provider</span><strong>${escapeHtml(data.provider)}</strong></div>
      <div><span>Model</span><strong>${escapeHtml(data.model)}</strong></div>
      <div><span>Confidence</span><strong>${Number.isFinite(confidence) ? `${Math.round(confidence * 100)}%` : "—"}</strong></div>
    </div>
    <div class="result-section">
      <h3>Summary</h3>
      <p class="summary">${escapeHtml(cleanDisplayText(data.summary))}</p>
    </div>
    <div class="result-section">
      <h3>Assessment breakdown</h3>
      <div class="dimensions">${dimensions}</div>
    </div>
    ${(data.missing_requirements || []).length ? `<div class="missing"><h3>Not clearly demonstrated</h3><ul>${data.missing_requirements.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
    <div class="second-opinion">
      <div>
        <strong>Prefer a local comparison?</strong>
        <p>Run the deterministic offline scorer without sending another request to Gemini.</p>
      </div>
      <button class="secondary-action" id="offline-button" type="button">Get offline second opinion</button>
    </div>
    <p class="privacy">Decision support only — review this result with a human and do not use it as the sole basis for an employment decision.</p>`;
  result.classList.remove("hidden");
  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

function submissionData() {
  const body = new FormData();
  body.append("resume", fileInput.files[0], fileInput.files[0].name);
  body.append("job_description", document.querySelector("#job-description").value);
  return body;
}

function renderOfflineComparison(primary, offline) {
  const delta = Number(offline.overall_score) - Number(primary.overall_score);
  const deltaLabel = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} points`;
  const comparison = document.createElement("div");
  comparison.className = "offline-comparison";
  comparison.innerHTML = `
    <div class="comparison-heading">
      <div><span class="result-label">Offline second opinion</span><h3>Local scorer comparison</h3></div>
      <span class="offline-badge">No API call</span>
    </div>
    <div class="comparison-grid">
      <div><span>Gemini score</span><strong>${Math.round(primary.overall_score)}/100</strong></div>
      <div><span>Offline score</span><strong>${Math.round(offline.overall_score)}/100</strong></div>
      <div><span>Difference</span><strong>${escapeHtml(deltaLabel)}</strong></div>
    </div>
    <p>Compare the two scores as review input. A difference is not proof that either result is correct.</p>
    <p class="privacy">The offline scorer is a deterministic local baseline, not the system-level fairness auditor. It does not prove that this individual assessment is fair.</p>`;
  const existing = document.querySelector(".offline-comparison");
  if (existing) existing.replaceWith(comparison);
  else result.insertBefore(comparison, result.querySelector(".privacy"));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files[0]) return;
  button.disabled = true;
  button.innerHTML = "Evaluating…";
  result.classList.add("hidden");
  try {
    const response = await fetch("/api/screen", { method: "POST", body: submissionData() });
    const responseText = await response.text();
    let data;
    try {
      data = JSON.parse(responseText);
    } catch {
      throw new Error(
        response.ok
          ? "The screening service returned an invalid response."
          : `The screening service failed (${response.status}). Try again.`
      );
    }
    if (!response.ok) throw new Error(data.detail || "The screening request failed.");
    renderResult(data);
    const offlineButton = document.querySelector("#offline-button");
    offlineButton.addEventListener("click", async () => {
      offlineButton.disabled = true;
      offlineButton.textContent = "Comparing…";
      try {
        const offlineResponse = await fetch("/api/screen/offline", {
          method: "POST",
          body: submissionData(),
        });
        const offlineText = await offlineResponse.text();
        let offlineData;
        try {
          offlineData = JSON.parse(offlineText);
        } catch {
          throw new Error(
            offlineResponse.status === 404
              ? "The running server does not support offline comparisons yet. Restart the web app and try again."
              : `The offline comparison failed (${offlineResponse.status}).`
          );
        }
        if (!offlineResponse.ok) throw new Error(offlineData.detail || "The offline comparison failed.");
        renderOfflineComparison(data, offlineData);
      } catch (error) {
        const errorBox = document.createElement("div");
        errorBox.className = "error";
        errorBox.textContent = error.message;
        result.insertBefore(errorBox, result.firstChild);
      } finally {
        offlineButton.disabled = false;
        offlineButton.textContent = "Get offline second opinion";
      }
    });
  } catch (error) {
    result.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
    result.classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.innerHTML = 'Evaluate resume <span>→</span>';
  }
});
