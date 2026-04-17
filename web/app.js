(() => {
  const INPUT_STATE_KEY = "autofigure_input_state_v1";

  const page = document.body.dataset.page;
  if (page === "input") {
    initInputPage();
  } else if (page === "canvas") {
    initCanvasPage();
  }

  function $(id) {
    return document.getElementById(id);
  }

  function initInputPage() {
    const confirmBtn = $("confirmBtn");
    const errorMsg = $("errorMsg");
    const uploadZone = $("uploadZone");
    const referenceFile = $("referenceFile");
    const referencePreview = $("referencePreview");
    const referenceStatus = $("referenceStatus");
    const imageSizeGroup = $("imageSizeGroup");
    const imageSizeInput = $("imageSize");
    const samBackend = $("samBackend");
    const samPrompt = $("samPrompt");
    const samApiKeyGroup = $("samApiKeyGroup");
    const samApiKeyInput = $("samApiKey");
    const pipelineSelect = $("pipelineSelect");
    const methodTextSection = $("methodTextSection");
    const inputImageSection = $("inputImageSection");
    const inputImageFile = $("inputImageFile");
    const inputImageZone = $("inputImageZone");
    const inputImagePreview = $("inputImagePreview");
    const inputImageStatus = $("inputImageStatus");
    let uploadedReferencePath = null;
    let uploadedInputImagePath = null;

    function syncPipelineVisibility() {
      const pipeline = pipelineSelect ? pipelineSelect.value : "svg";
      if (methodTextSection) methodTextSection.hidden = (pipeline === "drawio");
      if (inputImageSection) inputImageSection.hidden = (pipeline !== "drawio");
      saveInputState();
    }

    if (pipelineSelect) {
      pipelineSelect.addEventListener("change", syncPipelineVisibility);
      syncPipelineVisibility();
    }

    if (inputImageZone && inputImageFile) {
      inputImageZone.addEventListener("click", () => inputImageFile.click());
      inputImageZone.addEventListener("dragover", (e) => { e.preventDefault(); inputImageZone.classList.add("dragging"); });
      inputImageZone.addEventListener("dragleave", () => inputImageZone.classList.remove("dragging"));
      inputImageZone.addEventListener("drop", async (e) => {
        e.preventDefault();
        inputImageZone.classList.remove("dragging");
        const file = e.dataTransfer.files[0];
        if (file) {
          const uploaded = await uploadReference(file, confirmBtn, inputImagePreview, inputImageStatus);
          if (uploaded) { uploadedInputImagePath = uploaded.path; saveInputState(); }
        }
      });
      inputImageFile.addEventListener("change", async () => {
        const file = inputImageFile.files[0];
        if (file) {
          const uploaded = await uploadReference(file, confirmBtn, inputImagePreview, inputImageStatus);
          if (uploaded) { uploadedInputImagePath = uploaded.path; saveInputState(); }
        }
      });
    }

    function loadInputState() {
      try {
        const raw = window.sessionStorage.getItem(INPUT_STATE_KEY);
        if (!raw) {
          return null;
        }
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : null;
      } catch (_err) {
        return null;
      }
    }

    function saveInputState() {
      const state = {
        methodText: $("methodText")?.value ?? "",
        provider: $("provider")?.value ?? "gemini",
        apiKey: $("apiKey")?.value ?? "",
        optimizeIterations: $("optimizeIterations")?.value ?? "0",
        imageSize: imageSizeInput?.value ?? "4K",
        samBackend: samBackend?.value ?? "roboflow",
        samPrompt: samPrompt?.value ?? "icon,person,robot,animal",
        samApiKey: samApiKeyInput?.value ?? "",
        referencePath: uploadedReferencePath,
        referenceUrl: referencePreview?.src ?? "",
        referenceStatus: referenceStatus?.textContent ?? "",
      };
      try {
        window.sessionStorage.setItem(INPUT_STATE_KEY, JSON.stringify(state));
      } catch (_err) {
        // Ignore storage failures (e.g. private mode / quota)
      }
    }

    function applyInputState() {
      const state = loadInputState();
      if (!state) {
        return;
      }
      if (typeof state.methodText === "string") {
        $("methodText").value = state.methodText;
      }
      if (typeof state.provider === "string" && $("provider")) {
        $("provider").value = state.provider;
      }
      if (typeof state.apiKey === "string") {
        $("apiKey").value = state.apiKey;
      }
      if (typeof state.optimizeIterations === "string" && $("optimizeIterations")) {
        $("optimizeIterations").value = state.optimizeIterations;
      }
      if (typeof state.imageSize === "string" && imageSizeInput) {
        imageSizeInput.value = state.imageSize;
      }
      if (typeof state.samBackend === "string" && samBackend) {
        samBackend.value = state.samBackend;
      }
      if (typeof state.samPrompt === "string" && samPrompt) {
        samPrompt.value = state.samPrompt;
      }
      if (typeof state.samApiKey === "string" && samApiKeyInput) {
        samApiKeyInput.value = state.samApiKey;
      }
      if (typeof state.referencePath === "string" && state.referencePath) {
        uploadedReferencePath = state.referencePath;
      }
      if (
        referencePreview &&
        typeof state.referenceUrl === "string" &&
        state.referenceUrl
      ) {
        referencePreview.src = state.referenceUrl;
        referencePreview.classList.add("visible");
      }
      if (
        referenceStatus &&
        typeof state.referenceStatus === "string" &&
        state.referenceStatus
      ) {
        referenceStatus.textContent = state.referenceStatus;
      }
    }

    function syncImageSizeVisibility() {
      const provider = $("provider")?.value ?? "gemini";
      const show = provider === "gemini";
      if (imageSizeGroup) {
        imageSizeGroup.hidden = !show;
      }
      saveInputState();
    }

    function syncSamApiKeyVisibility() {
      const shouldShow =
        samBackend &&
        (samBackend.value === "fal" || samBackend.value === "roboflow");
      if (samApiKeyGroup) {
        samApiKeyGroup.hidden = !shouldShow;
      }
      if (!shouldShow && samApiKeyInput) {
        samApiKeyInput.value = "";
      }
      saveInputState();
    }

    applyInputState();

    if (samBackend) {
      samBackend.addEventListener("change", syncSamApiKeyVisibility);
      syncSamApiKeyVisibility();
    }
    if ($("provider")) {
      $("provider").addEventListener("change", syncImageSizeVisibility);
      syncImageSizeVisibility();
    }

    if (uploadZone && referenceFile) {
      uploadZone.addEventListener("click", () => referenceFile.click());
      uploadZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadZone.classList.add("dragging");
      });
      uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragging");
      });
      uploadZone.addEventListener("drop", async (event) => {
        event.preventDefault();
        uploadZone.classList.remove("dragging");
        const file = event.dataTransfer.files[0];
        if (file) {
          const uploadedRef = await uploadReference(file, confirmBtn, referencePreview, referenceStatus);
          if (uploadedRef) {
            uploadedReferencePath = uploadedRef.path;
            saveInputState();
          }
        }
      });
      referenceFile.addEventListener("change", async () => {
        const file = referenceFile.files[0];
        if (file) {
          const uploadedRef = await uploadReference(file, confirmBtn, referencePreview, referenceStatus);
          if (uploadedRef) {
            uploadedReferencePath = uploadedRef.path;
            saveInputState();
          }
        }
      });
    }

    const autoSaveFields = [
      $("methodText"),
      $("provider"),
      $("apiKey"),
      $("optimizeIterations"),
      $("imageSize"),
      samPrompt,
      samApiKeyInput,
    ];
    for (const field of autoSaveFields) {
      if (!field) {
        continue;
      }
      field.addEventListener("input", saveInputState);
      field.addEventListener("change", saveInputState);
    }

    confirmBtn.addEventListener("click", async () => {
      errorMsg.textContent = "";
      const pipeline = pipelineSelect ? pipelineSelect.value : "svg";

      if (pipeline === "drawio") {
        if (!uploadedInputImagePath) {
          errorMsg.textContent = "Please upload an input image for the draw.io pipeline.";
          return;
        }
      } else {
        const methodText = $("methodText").value.trim();
        if (!methodText) {
          errorMsg.textContent = "Please provide method text.";
          return;
        }
      }

      confirmBtn.disabled = true;
      confirmBtn.textContent = "Starting...";

      let payload;
      if (pipeline === "drawio") {
        payload = {
          pipeline: "drawio",
          input_image_path: uploadedInputImagePath,
          grid: $("gridSelect") ? $("gridSelect").value : "auto",
          target_ssim: parseFloat($("targetSsim") ? $("targetSsim").value : "0.90"),
          provider: $("provider").value,
          api_key: $("apiKey").value.trim() || null,
          optimize_iterations: parseInt($("optimizeIterations").value, 10),
          sam_backend: $("samBackend").value,
          sam_prompt: $("samPrompt").value.trim() || null,
          sam_api_key: $("samApiKey").value.trim() || null,
        };
      } else {
        payload = {
          pipeline: "svg",
          method_text: $("methodText").value.trim(),
          provider: $("provider").value,
          api_key: $("apiKey").value.trim() || null,
          optimize_iterations: parseInt($("optimizeIterations").value, 10),
          reference_image_path: uploadedReferencePath,
          sam_backend: $("samBackend").value,
          sam_prompt: $("samPrompt").value.trim() || null,
          sam_api_key: $("samApiKey").value.trim() || null,
        };
        if ($("provider").value === "gemini") {
          payload.image_size = imageSizeInput?.value || "4K";
        }
      }
      if (payload.sam_backend === "local") {
        payload.sam_api_key = null;
      }
      saveInputState();

      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || "Request failed");
        }

        const data = await response.json();
        window.location.href = `/canvas.html?job=${encodeURIComponent(data.job_id)}`;
      } catch (err) {
        errorMsg.textContent = err.message || "Failed to start job";
        confirmBtn.disabled = false;
        confirmBtn.textContent = "Confirm -> Canvas";
      }
    });
  }

  async function uploadReference(file, confirmBtn, previewEl, statusEl) {
    if (!file.type.startsWith("image/")) {
      statusEl.textContent = "Only image files are supported.";
      return null;
    }

    confirmBtn.disabled = true;
    statusEl.textContent = "Uploading reference...";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Upload failed");
      }

      const data = await response.json();
      statusEl.textContent = `Using uploaded reference: ${data.name}`;
      if (previewEl) {
        previewEl.src = data.url || "";
        previewEl.classList.add("visible");
      }
      return {
        path: data.path || null,
        url: data.url || "",
        name: data.name || "",
      };
    } catch (err) {
      statusEl.textContent = err.message || "Upload failed";
      return null;
    } finally {
      confirmBtn.disabled = false;
    }
  }

  async function initCanvasPage() {
    const params = new URLSearchParams(window.location.search);
    const jobId = params.get("job");
    const statusText = $("statusText");
    const jobIdEl = $("jobId");
    const artifactPanel = $("artifactPanel");
    const artifactList = $("artifactList");
    const toggle = $("artifactToggle");
    const logToggle = $("logToggle");
    const backToConfigBtn = $("backToConfigBtn");
    const logPanel = $("logPanel");
    const logBody = $("logBody");
    const iframe = $("svgEditorFrame");
    const fallback = $("svgFallback");
    const fallbackObject = $("fallbackObject");

    if (!jobId) {
      statusText.textContent = "Missing job id";
      return;
    }

    jobIdEl.textContent = jobId;

    toggle.addEventListener("click", () => {
      artifactPanel.classList.toggle("open");
    });

    logToggle.addEventListener("click", () => {
      logPanel.classList.toggle("open");
    });
    if (backToConfigBtn) {
      backToConfigBtn.addEventListener("click", () => {
        window.location.href = "/";
      });
    }

    let svgEditAvailable = false;
    let svgEditPath = null;
    try {
      const configRes = await fetch("/api/config");
      if (configRes.ok) {
        const config = await configRes.json();
        svgEditAvailable = Boolean(config.svgEditAvailable);
        svgEditPath = config.svgEditPath || null;
      }
    } catch (err) {
      svgEditAvailable = false;
    }

    if (svgEditAvailable && svgEditPath) {
      iframe.src = svgEditPath;
    } else {
      fallback.classList.add("active");
      iframe.style.display = "none";
    }

    let svgReady = false;
    let pendingSvgText = null;

    iframe.addEventListener("load", () => {
      svgReady = true;
      if (pendingSvgText) {
        tryLoadSvg(pendingSvgText);
        pendingSvgText = null;
      }
    });

    const stepMap = {
      figure: { step: 1, label: "Figure generated" },
      samed: { step: 2, label: "SAM3 segmentation" },
      icon_raw: { step: 3, label: "Icons extracted" },
      icon_nobg: { step: 3, label: "Icons refined" },
      icon_sheet: { step: 3, label: "Icon sheet generated" },
      template_svg: { step: 4, label: "Template SVG ready" },
      final_svg: { step: 5, label: "Final SVG ready" },
      template_drawio: { step: 4, label: "draw.io template ready" },
      optimized_drawio: { step: 5, label: "draw.io optimized" },
      final_drawio: { step: 6, label: "Final draw.io ready" },
      rendered: { step: 6, label: "Rendered preview" },
    };

    let currentStep = 0;

    const artifacts = new Set();
    const eventSource = new EventSource(`/api/events/${jobId}`);
    let isFinished = false;

    eventSource.addEventListener("artifact", async (event) => {
      const data = JSON.parse(event.data);
      if (!artifacts.has(data.path)) {
        artifacts.add(data.path);
        addArtifactCard(artifactList, data);
      }

      if (data.kind === "template_svg" || data.kind === "final_svg") {
        await loadSvgAsset(data.url);
      }

      if (data.kind === "final_drawio" || data.kind === "template_drawio") {
        // Show download link for .drawio files
        const dlLink = document.createElement("a");
        dlLink.href = data.url;
        dlLink.download = data.name;
        dlLink.className = "artifact-card";
        dlLink.innerHTML = `<div class="artifact-meta"><div class="artifact-name">${data.name}</div><div class="artifact-badge">Download .drawio</div></div>`;
        artifactList.prepend(dlLink);
      }

      if (stepMap[data.kind] && stepMap[data.kind].step > currentStep) {
        currentStep = stepMap[data.kind].step;
        const totalSteps = (data.kind.includes("drawio") || data.kind === "rendered" || data.kind === "icon_sheet") ? 7 : 5;
        statusText.textContent = `Step ${currentStep}/${totalSteps} - ${stepMap[data.kind].label}`;
      }
    });

    eventSource.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.state === "started") {
        statusText.textContent = "Running";
      } else if (data.state === "finished") {
        isFinished = true;
        if (typeof data.code === "number" && data.code !== 0) {
          statusText.textContent = `Failed (code ${data.code})`;
        } else {
          statusText.textContent = "Done";
        }
      }
    });

    eventSource.addEventListener("log", (event) => {
      const data = JSON.parse(event.data);
      appendLogLine(logBody, data);
    });

    eventSource.onerror = () => {
      if (isFinished) {
        eventSource.close();
        return;
      }
      statusText.textContent = "Disconnected";
    };

    async function loadSvgAsset(url) {
      let svgText = "";
      try {
        const response = await fetch(url);
        svgText = await response.text();
      } catch (err) {
        return;
      }

      if (svgEditAvailable) {
        if (!svgEditPath) {
          return;
        }
        if (!svgReady) {
          pendingSvgText = svgText;
          return;
        }

        const loaded = tryLoadSvg(svgText);
        if (!loaded) {
          iframe.src = `${svgEditPath}?url=${encodeURIComponent(url)}`;
        }
      } else {
        fallbackObject.data = url;
      }
    }

    function tryLoadSvg(svgText) {
      if (!iframe.contentWindow) {
        return false;
      }

      const win = iframe.contentWindow;
      if (win.svgEditor && typeof win.svgEditor.loadFromString === "function") {
        win.svgEditor.loadFromString(svgText);
        return true;
      }
      if (win.svgCanvas && typeof win.svgCanvas.setSvgString === "function") {
        win.svgCanvas.setSvgString(svgText);
        return true;
      }
      return false;
    }
  }

  function appendLogLine(container, data) {
    const line = `[${data.stream}] ${data.line}`;
    const lines = container.textContent.split("\n").filter(Boolean);
    lines.push(line);
    if (lines.length > 200) {
      lines.splice(0, lines.length - 200);
    }
    container.textContent = lines.join("\n");
    container.scrollTop = container.scrollHeight;
  }

  function addArtifactCard(container, data) {
    const card = document.createElement("a");
    card.className = "artifact-card";
    card.href = data.url;
    card.target = "_blank";
    card.rel = "noreferrer";

    const img = document.createElement("img");
    img.src = data.url;
    img.alt = data.name;
    img.loading = "lazy";

    const meta = document.createElement("div");
    meta.className = "artifact-meta";

    const name = document.createElement("div");
    name.className = "artifact-name";
    name.textContent = data.name;

    const badge = document.createElement("div");
    badge.className = "artifact-badge";
    badge.textContent = formatKind(data.kind);

    meta.appendChild(name);
    meta.appendChild(badge);
    card.appendChild(img);
    card.appendChild(meta);
    container.prepend(card);
  }

  function formatKind(kind) {
    switch (kind) {
      case "figure":
        return "figure";
      case "samed":
        return "samed";
      case "icon_raw":
        return "icon raw";
      case "icon_nobg":
        return "icon no-bg";
      case "icon_sheet":
        return "icon sheet";
      case "template_svg":
        return "template SVG";
      case "final_svg":
        return "final SVG";
      case "template_drawio":
        return "template drawio";
      case "optimized_drawio":
        return "optimized drawio";
      case "final_drawio":
        return "final drawio";
      case "rendered":
        return "rendered";
      default:
        return "artifact";
    }
  }
})();
