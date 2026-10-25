(() => {
  const audioExtensions = new Set([".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm", ".wma"]);
  const modeInputs = {
    single: document.getElementById("single-input"),
    folder: document.getElementById("folder-input"),
    zip: document.getElementById("zip-input"),
  };
  const dropZone = document.getElementById("drop-zone");
  const fileQueue = document.getElementById("file-queue");
  const startButton = document.getElementById("start-button");
  const uploadError = document.getElementById("upload-error");
  const emptyRun = document.getElementById("empty-run");
  const runContent = document.getElementById("run-content");
  const jobStatus = document.getElementById("job-status");
  const callList = document.getElementById("call-list");
  const detailsPanel = document.getElementById("details-panel");
  const detailContent = document.getElementById("detail-content");

  let mode = "single";
  let selectedFiles = [];
  let activeJobId = null;
  let currentCalls = [];
  let selectedCall = null;
  let selectedDetail = "transcript";
  let isBusy = false;

  const supportedAudio = (name) => audioExtensions.has(`.${name.split(".").pop().toLowerCase()}`);
  const prettyBytes = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  };
  const basename = (name) => name.replaceAll("\\", "/").split("/").filter(Boolean).pop() || name;
  const filePath = (file) => file.webkitRelativePath || file.name;
  const stem = (name) => basename(name).replace(/\.[^.]+$/, "");

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clearError() {
    uploadError.textContent = "";
    uploadError.classList.add("hidden");
  }

  function showUploadError(message) {
    uploadError.textContent = message;
    uploadError.classList.remove("hidden");
  }

  function setMode(nextMode) {
    if (isBusy) return;
    mode = nextMode;
    document.querySelectorAll(".mode-tab").forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    const hints = {
      single: ["Drop an audio file here", "or browse your files", "MP3, WAV, M4A and more · Folder or ZIP supported"],
      folder: ["Choose a folder of recordings", "or browse for a folder", "Audio files inside the selected folder will be processed"],
      zip: ["Drop a ZIP archive here", "or browse for an archive", "ZIP archives containing supported audio files"],
    }[mode];
    document.getElementById("drop-title").textContent = hints[0];
    const subtitle = document.getElementById("drop-subtitle");
    subtitle.replaceChildren(document.createTextNode("or "), el("span", "", hints[1].replace(/^or\s+/i, "")));
    document.getElementById("format-note").textContent = hints[2];
    selectedFiles = [];
    Object.values(modeInputs).forEach((input) => { input.value = ""; });
    renderQueue();
    clearError();
  }

  function setFiles(fileList, sourceMode = mode) {
    clearError();
    const files = Array.from(fileList || []);
    if (sourceMode === "zip") {
      if (files.length !== 1 || !files[0].name.toLowerCase().endsWith(".zip")) {
        showUploadError("Choose one .zip archive at a time.");
        return;
      }
      selectedFiles = files;
    } else {
      const audioFiles = files.filter((file) => supportedAudio(file.name));
      if (!audioFiles.length) {
        showUploadError("No supported audio files were found. Choose MP3, WAV, M4A, OGG, FLAC, AAC, OPUS, WEBM or WMA.");
        return;
      }
      if (sourceMode === "single" && audioFiles.length > 1) {
        showUploadError("Choose a single audio file or switch to Folder mode.");
        return;
      }
      selectedFiles = sourceMode === "single" ? audioFiles.slice(0, 1) : audioFiles;
      if (files.length !== audioFiles.length) {
        document.getElementById("format-note").textContent = `${files.length - audioFiles.length} non-audio file(s) ignored`;
      }
    }
    renderQueue();
  }

  function renderQueue() {
    const count = selectedFiles.length;
    startButton.disabled = count === 0 || isBusy;
    if (!count) {
      fileQueue.classList.add("hidden");
      fileQueue.replaceChildren();
      return;
    }
    fileQueue.classList.remove("hidden");
    fileQueue.replaceChildren();
    const top = el("div", "queue-topline");
    top.append(el("span", "", `${count} ${count === 1 ? "file" : "audio files"} selected`));
    top.append(el("span", "", prettyBytes(selectedFiles.reduce((total, file) => total + file.size, 0))));
    fileQueue.append(top);
    selectedFiles.slice(0, 5).forEach((file) => {
      const row = el("div", "queue-file");
      row.append(el("span", "file-mini-icon", mode === "zip" ? "◇" : "♫"));
      row.append(el("span", "queue-name", filePath(file)));
      row.append(el("span", "queue-size", prettyBytes(file.size)));
      fileQueue.append(row);
    });
    if (count > 5) fileQueue.append(el("div", "queue-topline", `and ${count - 5} more…`));
  }

  function setBusy(busy) {
    isBusy = busy;
    startButton.disabled = busy || selectedFiles.length === 0;
    document.querySelectorAll(".mode-tab").forEach((button) => { button.disabled = busy; });
    dropZone.disabled = busy;
    dropZone.classList.toggle("is-disabled", busy);
  }

  function setRunStatus(status, label) {
    jobStatus.className = `status-badge ${status}`;
    jobStatus.replaceChildren(el("span", "status-dot"), document.createTextNode(label));
  }

  function showRun() {
    emptyRun.classList.add("hidden");
    runContent.classList.remove("hidden");
  }

  function updateProgress(percent, label, stage, currentFile) {
    const value = Math.max(0, Math.min(100, percent || 0));
    document.getElementById("job-percent").textContent = `${value}%`;
    document.getElementById("progress-bar").style.width = `${value}%`;
    document.getElementById("job-label").textContent = label;
    document.getElementById("job-stage").textContent = stage || "Working through the pipeline";
    const row = document.getElementById("current-file-row");
    if (currentFile) {
      row.classList.remove("hidden");
      document.getElementById("current-file").textContent = currentFile;
    } else {
      row.classList.add("hidden");
    }
  }

  function requestUpload(jobId, index, file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", `/api/jobs/${jobId}/files/${index}`);
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(event.loaded, event.total);
      };
      xhr.onerror = () => reject(new Error("Upload connection failed."));
      xhr.onload = () => {
        let response = {};
        try { response = JSON.parse(xhr.responseText); } catch (_) { /* server error handled below */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(response);
        else reject(new Error(response.error || `Upload failed (${xhr.status}).`));
      };
      xhr.send(file);
    });
  }

  async function beginAnalysis() {
    if (!selectedFiles.length || isBusy) return;
    clearError();
    setBusy(true);
    showRun();
    detailsPanel.classList.add("hidden");
    selectedCall = null;
    currentCalls = [];
    setRunStatus("working", "Uploading");
    document.getElementById("job-error").classList.add("hidden");
    document.getElementById("call-list").replaceChildren(el("p", "no-calls", "Your recordings will appear here when analysis begins."));
    const totalBytes = selectedFiles.reduce((total, file) => total + file.size, 0);
    updateProgress(0, `Uploading ${selectedFiles.length} ${selectedFiles.length === 1 ? "file" : "files"}`, "Preparing secure local upload");

    try {
      const createResponse = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files: selectedFiles.map((file) => ({ name: filePath(file), size: file.size })) }),
      });
      const created = await createResponse.json();
      if (!createResponse.ok) throw new Error(created.error || "Could not create the analysis job.");
      activeJobId = created.id;

      let bytesBeforeFile = 0;
      for (let index = 0; index < selectedFiles.length; index += 1) {
        const file = selectedFiles[index];
        updateProgress(
          Math.round(100 * bytesBeforeFile / Math.max(1, totalBytes)),
          `Uploading file ${index + 1} of ${selectedFiles.length}`,
          "Sending files to the local pipeline",
          filePath(file),
        );
        await requestUpload(activeJobId, index, file, (loaded) => {
          updateProgress(
            Math.round(100 * (bytesBeforeFile + loaded) / Math.max(1, totalBytes)),
            `Uploading file ${index + 1} of ${selectedFiles.length}`,
            "Sending files to the local pipeline",
            filePath(file),
          );
        });
        bytesBeforeFile += file.size;
      }

      const startResponse = await fetch(`/api/jobs/${activeJobId}/start`, { method: "POST" });
      const started = await startResponse.json();
      if (!startResponse.ok) throw new Error(started.error || "Could not start processing.");
      updateProgress(started.progress || 0, "Processing recordings", started.current_stage, started.current_file);
      setRunStatus("working", "Processing");
      await pollJob(activeJobId);
    } catch (error) {
      showUploadError(error.message || "The upload could not be processed.");
      const jobError = document.getElementById("job-error");
      jobError.textContent = error.message || "The upload could not be processed.";
      jobError.classList.remove("hidden");
      setRunStatus("failed", "Needs attention");
    } finally {
      setBusy(false);
    }
  }

  async function pollJob(jobId) {
    while (activeJobId === jobId) {
      const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "Could not read job status.");
      const uploadPercent = job.status === "uploading" ? job.upload_progress : null;
      const label = job.status === "queued" ? "Waiting for GPU" : job.status === "processing" ? `Processing ${job.current_index || 1}${job.total_calls ? ` of ${job.total_calls}` : ""}` : "Processing recordings";
      updateProgress(uploadPercent ?? job.progress, label, job.current_stage, job.current_file);

      if (job.status === "completed" || job.status === "completed_with_errors" || job.status === "failed") {
        renderCompletedJob(job);
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1300));
    }
  }

  function renderCompletedJob(job) {
    const failed = job.status === "failed" || job.status === "completed_with_errors";
    setRunStatus(failed ? "failed" : "done", job.status === "completed" ? "Complete" : job.status === "failed" ? "Failed" : "Some calls failed");
    updateProgress(job.status === "failed" ? (job.progress || 0) : 100, job.status === "failed" ? "Processing stopped" : "Analysis complete", job.current_stage, null);
    currentCalls = job.calls || [];
    renderCallList(currentCalls);
    const errors = document.getElementById("job-error");
    if (job.error) {
      errors.textContent = job.error;
      errors.classList.remove("hidden");
    } else if (job.summary && job.summary.failed_count) {
      errors.textContent = `${job.summary.failed_count} recording(s) could not be processed. Select a completed recording to view its results.`;
      errors.classList.remove("hidden");
    } else {
      errors.classList.add("hidden");
    }
    renderBatchSummary(job.summary);
    if (currentCalls.length) loadCallDetails(currentCalls[0].id);
    setBusy(false);
  }

  function renderBatchSummary(summary) {
    const node = document.getElementById("batch-summary");
    if (!summary || summary.total_files === undefined) {
      node.classList.add("hidden");
      node.replaceChildren();
      return;
    }
    node.classList.remove("hidden");
    node.replaceChildren();
    const successful = summary.successful ?? (summary.failed_count ? summary.total_files - summary.failed_count : summary.total_files);
    const failed = summary.failed_count ?? 0;
    const skipped = summary.skipped_resumed ?? 0;
    [["Processed", successful], ["Failed", failed], ["Resumed", skipped]].forEach(([label, value]) => {
      const stat = el("span", "", `${label} `);
      stat.append(el("b", "", value));
      node.append(stat);
    });
  }

  function renderCallList(calls) {
    callList.replaceChildren();
    document.getElementById("result-count").textContent = String(calls.length);
    if (!calls.length) {
      callList.append(el("p", "no-calls", "No completed call results were found."));
      return;
    }
    calls.forEach((call) => {
      const button = el("button", `call-item${call.status === "failed" ? " error" : ""}${selectedCall === call.id ? " selected" : ""}`);
      button.type = "button";
      button.append(el("span", "file-mini-icon", call.status === "failed" ? "!" : "♫"));
      const info = el("span", "call-info");
      info.append(el("span", "call-name", call.name));
      info.append(el("span", "call-state", call.status === "failed" ? (call.error || "Processing failed") : "Transcript · Intent · Evaluation"));
      button.append(info, el("span", "call-arrow", call.status === "failed" ? "" : "›"));
      button.disabled = call.status === "failed";
      button.addEventListener("click", () => loadCallDetails(call.id));
      callList.append(button);
    });
  }

  async function loadCallDetails(callId) {
    if (!activeJobId) return;
    try {
      const response = await fetch(`/api/jobs/${activeJobId}/calls/${encodeURIComponent(callId)}`, { cache: "no-store" });
      const details = await response.json();
      if (!response.ok) throw new Error(details.error || "Could not load recording details.");
      selectedCall = callId;
      document.getElementById("details-heading").textContent = details.name;
      detailsPanel.classList.remove("hidden");
      renderCallList(currentCalls);
      renderDetail(details, selectedDetail);
      if (window.innerWidth < 960) detailsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      const node = document.getElementById("job-error");
      node.textContent = error.message;
      node.classList.remove("hidden");
    }
  }

  function renderDetail(details, tab) {
    selectedDetail = tab;
    document.querySelectorAll(".detail-tab").forEach((button) => {
      const active = button.dataset.detail === tab;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    detailContent.replaceChildren();
    if (!details[tab]) {
      detailContent.append(el("p", "muted-copy", "This result is not available for this recording."));
      return;
    }
    if (tab === "transcript") renderTranscript(detailContent, details.transcript);
    if (tab === "intent") renderIntent(detailContent, details.intent);
    if (tab === "evaluation") renderEvaluation(detailContent, details.evaluation);
  }

  function renderTranscript(container, transcript) {
    const segments = Array.isArray(transcript) ? transcript : [];
    const toolbar = el("div", "transcript-toolbar");
    toolbar.append(el("span", "", `${segments.length} timestamped segments`));
    const speakers = new Set(segments.map((segment) => segment.speaker_id).filter(Boolean));
    toolbar.append(el("span", "", `${speakers.size} speaker${speakers.size === 1 ? "" : "s"}`));
    container.append(toolbar);
    if (!segments.length) {
      container.append(el("p", "muted-copy", "No transcript segments were returned."));
      return;
    }
    const list = el("div", "transcript-list");
    segments.forEach((segment) => {
      const row = el("div", "transcript-turn");
      row.append(el("span", "turn-time", `${formatTime(segment.start)} – ${formatTime(segment.end)}`));
      const speaker = el("span", `speaker-chip${isAgentLabel(segment.speaker_id) ? " agent" : ""}`, segment.speaker_id || "Unknown");
      row.append(speaker);
      const textWrap = el("div", "turn-text", segment.text || segment.raw_text || "[No recognized speech]");
      if (segment.speaker_match_confidence !== undefined) {
        textWrap.append(el("span", "confidence", `Speaker match ${Math.round(Number(segment.speaker_match_confidence) * 100)}%`));
      }
      row.append(textWrap);
      list.append(row);
    });
    container.append(list);
  }

  function renderIntent(container, intent) {
    const roles = intent.speaker_roles || {};
    const agent = roles.agent || {};
    const client = roles.client || {};
    const grid = el("div", "detail-summary-grid");
    grid.append(summaryCard("Primary customer intent", intent.primary_client_intent || "Unclear", "Main reason for the call"));
    grid.append(summaryCard("Customer speaker", client.speaker_id || "Unconfirmed", client.evidence || "Role identified from call context"));
    grid.append(summaryCard("Agent speaker", agent.speaker_id || "Unconfirmed", agent.evidence || "Role identified from call context"));
    container.append(grid);
    appendListSection(container, "Secondary intents", intent.secondary_intents);
    appendListSection(container, "Customer needs and requests", intent.client_needs_and_requests);
    appendListSection(container, "Key events", intent.key_events);
    appendListSection(container, "Uncertainties", intent.uncertainties);
  }

  function renderEvaluation(container, evaluation) {
    const resolution = evaluation.resolution || {};
    const scorecard = evaluation.scorecard || {};
    const grid = el("div", "score-grid");
    [
      ["Professionalism", scorecard.professionalism_score],
      ["Respectfulness", scorecard.respectfulness_score],
      ["Politeness", scorecard.politeness_score],
      ["Total score", scorecard.total_score],
    ].forEach(([label, score]) => {
      const card = el("div", "score-card");
      card.append(el("span", "score-label", label));
      const value = score === null || score === undefined ? "—" : `${Number(score).toFixed(Number(score) % 1 ? 1 : 0)}`;
      card.append(el("strong", `score-value${value === "—" ? " unavailable" : ""}`, value));
      if (value !== "—") card.lastChild.append(document.createTextNode(" / 100"));
      grid.append(card);
    });
    container.append(grid);

    const resolutionSection = el("section", "detail-section");
    resolutionSection.append(el("h3", "", "Resolution"));
    const resolutionRow = el("div", "tag-list");
    resolutionRow.append(el("span", "resolution-pill", (resolution.status || "unclear").replaceAll("_", " ")));
    if (resolution.explanation) resolutionRow.append(el("span", "detail-tag", resolution.explanation));
    resolutionSection.append(resolutionRow);
    container.append(resolutionSection);

    appendListSection(container, "Client needs", evaluation.client_needs);
    appendListSection(container, "Agent actions", evaluation.agent_actions);
    if (scorecard.evidence && scorecard.evidence.length) {
      const section = el("section", "detail-section");
      section.append(el("h3", "", "Evidence from the call"));
      scorecard.evidence.forEach((item) => {
        const card = el("div", "evidence-card");
        card.append(el("blockquote", "", `“${item.quote || ""}”`));
        card.append(el("div", "evidence-meta", [item.timestamp, item.speaker_id, item.category, item.interpretation].filter(Boolean).join(" · ")));
        section.append(card);
      });
      container.append(section);
    }
    appendListSection(container, "Uncertainties", evaluation.uncertainties);
  }

  function summaryCard(label, value, detail) {
    const card = el("div", "summary-card");
    card.append(el("label", "", label));
    card.append(el("strong", "", value));
    if (detail) card.append(el("p", "", detail));
    return card;
  }

  function appendListSection(container, title, items) {
    if (!Array.isArray(items) || !items.length) return;
    const section = el("section", "detail-section");
    section.append(el("h3", "", title));
    if (items.every((item) => typeof item === "string")) {
      const tags = el("div", "tag-list");
      items.forEach((item) => tags.append(el("span", "detail-tag", item)));
      section.append(tags);
    } else {
      const list = el("ul", "bullet-list");
      items.forEach((item) => list.append(el("li", "", typeof item === "string" ? item : JSON.stringify(item))));
      section.append(list);
    }
    container.append(section);
  }

  function formatTime(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value)) return "--:--";
    const mins = Math.floor(value / 60);
    const secs = Math.floor(value % 60).toString().padStart(2, "0");
    const hours = Math.floor(mins / 60);
    const remainingMins = (mins % 60).toString().padStart(2, "0");
    return hours ? `${hours}:${remainingMins}:${secs}` : `${mins}:${secs}`;
  }

  function isAgentLabel(value) {
    return typeof value === "string" && value.toLowerCase().includes("agent");
  }

  document.querySelectorAll(".mode-tab").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  Object.entries(modeInputs).forEach(([inputMode, input]) => {
    input.addEventListener("change", () => setFiles(input.files, inputMode));
  });
  dropZone.addEventListener("click", () => modeInputs[mode].click());
  dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    setFiles(event.dataTransfer.files, mode);
  });
  startButton.addEventListener("click", beginAnalysis);
  document.getElementById("close-details").addEventListener("click", () => detailsPanel.classList.add("hidden"));
  document.querySelectorAll(".detail-tab").forEach((button) => {
    button.addEventListener("click", () => {
      selectedDetail = button.dataset.detail;
      if (selectedCall && activeJobId) loadCallDetails(selectedCall);
    });
  });
  window.addEventListener("beforeunload", () => {
    if (activeJobId) sessionStorage.setItem("callwise-last-job", activeJobId);
  });

  // Keep the most recent result visible if the page is refreshed during a run.
  const rememberedJob = sessionStorage.getItem("callwise-last-job");
  if (rememberedJob) {
    activeJobId = rememberedJob;
    fetch(`/api/jobs/${rememberedJob}`, { cache: "no-store" }).then(async (response) => {
      if (!response.ok) return;
      const job = await response.json();
      showRun();
      if (["completed", "completed_with_errors", "failed"].includes(job.status)) renderCompletedJob(job);
      else {
        setBusy(true);
        setRunStatus("working", job.status === "uploading" ? "Uploading" : "Processing");
        pollJob(rememberedJob).finally(() => setBusy(false));
      }
    }).catch(() => sessionStorage.removeItem("callwise-last-job"));
  }
})();
