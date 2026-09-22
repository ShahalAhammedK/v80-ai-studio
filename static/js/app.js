import { getHealth, segmentCampaign, replacePerson, detectGender } from "./api.js";
import { showToast } from "./toast.js";
import { MaskEditor } from "./maskEditor.js";
import { downloadImage, dataUrlToBlob, copyImage, shareToPlatform } from "./share.js";

const MAX_FILE_MB = 12;
const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];

const state = {
  campaignFile: null,
  campaignImg: null, // HTMLImageElement, natural size
  personFile: null,
  personImg: null,
  maskDataUrl: null, // grayscale PNG data URL, white = person
  settings: {
    identityPreservation: "high",
    scenePreservation: "maximum",
    keepPose: true,
    matchLighting: true,
  },
  generations: [], // { dataUrl }
  activeGeneration: -1,
  campaignSource: null, // null | "upload" | "default-woman" | "default-man"
  suggestedGender: null, // "male" | "female" | "unknown" | null
};

const el = (id) => document.getElementById(id);
const maskEditor = new MaskEditor();

function fileSizeLabel(bytes) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("corrupt"));
    };
    img.src = url;
  });
}

function validateClientFile(file) {
  if (!file) return "Please choose an image file to upload.";
  if (!ALLOWED_TYPES.includes(file.type)) return "Please upload a JPG, PNG or WebP image.";
  if (file.size > MAX_FILE_MB * 1024 * 1024) return `That image is too large. Please upload a file under ${MAX_FILE_MB}MB.`;
  return null;
}

/* ---------------- Health / status pill ---------------- */

async function refreshStatus() {
  const health = await getHealth();
  const dot = el("statusDot");
  const text = el("statusText");
  if (health.aiReady) {
    dot.classList.remove("offline");
    text.textContent = "AI Ready";
  } else {
    dot.classList.add("offline");
    text.textContent = "AI Not Configured";
  }
}

/* ---------------- Generic uploader wiring ---------------- */

function wireUploader({ dropZoneId, inputId, onFile }) {
  const zone = el(dropZoneId);
  const input = el(inputId);

  const openPicker = () => input.click();
  zone.addEventListener("click", openPicker);
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); }
  });

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) onFile(input.files[0]);
    input.value = "";
  });

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove("dragover"); })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) onFile(file);
  });
}

/* ---------------- Campaign upload ---------------- */

async function handleCampaignFile(file, source = "upload") {
  const err = validateClientFile(file);
  if (err) return showToast(err);

  let loaded;
  try {
    loaded = await loadImageFromFile(file);
  } catch {
    return showToast("We couldn't read that image. Please try a different file.");
  }

  state.campaignFile = file;
  state.campaignImg = loaded.img;
  state.campaignSource = source;
  state.maskDataUrl = null;
  state.generations = [];
  el("resultSection").classList.add("hidden");
  updateDefaultCampaignActiveState();

  el("campaignPreviewImg").src = loaded.url;
  el("campaignFileName").textContent = file.name;
  el("campaignFileSize").textContent = fileSizeLabel(file.size);
  el("campaignPreview").classList.remove("hidden");

  el("maskBaseImg").src = loaded.url;
  el("maskThumbCanvas").getContext("2d").clearRect(0, 0, 9999, 9999);
  el("maskStatus").textContent = "Detecting the person automatically…";
  el("editMaskBtn").disabled = true;
  updateReplaceButtonState();

  // The mask panel is hidden now, so a failure here would otherwise leave the
  // Replace button disabled with nothing on screen explaining why — surface it.
  try {
    const result = await segmentCampaign(file);
    if (result.available && result.mask) {
      state.maskDataUrl = `data:image/png;base64,${result.mask}`;
      el("maskStatus").textContent = "Person detected automatically. Review and adjust if needed.";
    } else {
      el("maskStatus").textContent = "Automatic detection isn't available — please paint the mask manually.";
      showToast("We couldn't prepare the campaign image. Please reload the page and try again.");
    }
  } catch (e) {
    el("maskStatus").textContent = "Automatic detection failed — please paint the mask manually.";
    showToast("We couldn't prepare the campaign image. Please reload the page and try again.");
  }

  el("editMaskBtn").disabled = false;
  drawMaskThumbnail();
  updateReplaceButtonState();
}

function drawMaskThumbnail() {
  const canvas = el("maskThumbCanvas");
  if (!state.campaignImg) return;
  canvas.width = state.campaignImg.naturalWidth;
  canvas.height = state.campaignImg.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.maskDataUrl) return;

  const maskImg = new Image();
  maskImg.onload = () => {
    const off = document.createElement("canvas");
    off.width = canvas.width; off.height = canvas.height;
    const offCtx = off.getContext("2d");
    offCtx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
    const data = offCtx.getImageData(0, 0, canvas.width, canvas.height);
    const out = ctx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < data.data.length; i += 4) {
      const v = data.data[i];
      const on = v > 100;
      out.data[i] = 0; out.data[i + 1] = 87; out.data[i + 2] = 255;
      out.data[i + 3] = on ? 140 : 0;
    }
    ctx.putImageData(out, 0, 0);
  };
  maskImg.src = state.maskDataUrl;
}

el("campaignRemoveBtn").addEventListener("click", () => {
  state.campaignFile = null;
  state.campaignImg = null;
  state.campaignSource = null;
  state.maskDataUrl = null;
  el("campaignPreview").classList.add("hidden");
  el("maskThumbCanvas").getContext("2d").clearRect(0, 0, 9999, 9999);
  el("maskBaseImg").src = "";
  el("maskStatus").textContent = "Choose a V80 version to generate a mask.";
  el("editMaskBtn").disabled = true;
  updateDefaultCampaignActiveState();
  updateReplaceButtonState();
});

/* ---------------- Default campaign picker ---------------- */

async function urlToFile(url, filename) {
  const response = await fetch(url);
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type });
}

function updateDefaultCampaignActiveState() {
  document.querySelectorAll(".default-campaign-card").forEach((card) => {
    const isActive = state.campaignSource === `default-${card.dataset.gender}`;
    card.classList.toggle("active", isActive);
  });
}

async function selectDefaultCampaign(gender, src) {
  try {
    const ext = src.split(".").pop();
    const file = await urlToFile(src, `v80-${gender}.${ext}`);
    await handleCampaignFile(file, `default-${gender}`);
  } catch {
    showToast("We couldn't load that default campaign. Please try uploading your own image.");
  }
}

document.querySelectorAll(".default-campaign-card").forEach((card) => {
  card.addEventListener("click", () => selectDefaultCampaign(card.dataset.gender, card.dataset.src));
});

/**
 * There is one fixed campaign image now, so the detected gender no longer
 * picks between campaigns. It's passed to the backend instead, so the prompt
 * can dress the replacement person in a black suit cut for their own build —
 * the campaign shot stays the same either way.
 */
function applyGenderSuggestion(gender) {
  state.suggestedGender = gender;
  state.settings.personGender = gender;
}

/* ---------------- Person upload ---------------- */

async function handlePersonFile(file) {
  const err = validateClientFile(file);
  if (err) return showToast(err);

  let loaded;
  try {
    loaded = await loadImageFromFile(file);
  } catch {
    return showToast("We couldn't read that image. Please try a different file.");
  }

  state.personFile = file;
  state.personImg = loaded.img;

  el("personPreviewImg").src = loaded.url;
  el("personFileName").textContent = file.name;
  el("personFileSize").textContent = fileSizeLabel(file.size);
  el("personPreview").classList.remove("hidden");
  el("personUploader").classList.add("hidden");
  updateReplaceButtonState();

  detectGender(file).then((result) => {
    if (result.gender === "male" || result.gender === "female") {
      applyGenderSuggestion(result.gender);
    }
  });
}

el("personRemoveBtn").addEventListener("click", () => {
  state.personFile = null;
  state.personImg = null;
  state.suggestedGender = null;
  delete state.settings.personGender;
  el("personPreview").classList.add("hidden");
  el("personUploader").classList.remove("hidden");
  updateReplaceButtonState();
});
el("personReplaceBtn").addEventListener("click", () => el("personInput").click());

wireUploader({ dropZoneId: "personUploader", inputId: "personInput", onFile: handlePersonFile });

/* ---------------- Settings ---------------- */

el("settingsToggle").addEventListener("click", () => {
  const body = el("settingsBody");
  const collapsed = body.classList.toggle("collapsed");
  el("settingsChevron").style.transform = collapsed ? "" : "rotate(180deg)";
});

document.querySelectorAll(".segmented").forEach((group) => {
  const key = group.dataset.group;
  group.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      group.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.settings[key] = btn.dataset.value;
    });
  });
});

/* ---------------- Mask editor ---------------- */

async function openMaskEditor() {
  if (!state.campaignImg) return;
  const result = await maskEditor.open(state.campaignImg, state.maskDataUrl);
  if (!result) return;
  state.maskDataUrl = result.previewDataUrl;
  drawMaskThumbnail();
  el("maskStatus").textContent = "Mask updated manually.";
  updateReplaceButtonState();
}

el("editMaskBtn").addEventListener("click", openMaskEditor);

/* ---------------- Replace button state ---------------- */

function updateReplaceButtonState() {
  el("replaceBtn").disabled = !(state.campaignFile && state.personFile && state.maskDataUrl);
}

/* ---------------- Processing + generation ---------------- */

const STAGES = [
  "Preparing your images…",
  "Analyzing the subject…",
  "Creating the replacement mask…",
  "Preserving the campaign design…",
  "Replacing the person…",
  "Matching lighting and details…",
  "Finalizing your image…",
  "Almost done…",
];

function runStageAnimation() {
  let i = 0;
  el("processingStage").textContent = STAGES[0];
  const timer = setInterval(() => {
    i = Math.min(i + 1, STAGES.length - 1);
    el("processingStage").textContent = STAGES[i];
  }, 2200);
  return () => clearInterval(timer);
}

async function maskDataUrlToBlob(dataUrl) {
  return dataUrlToBlob(dataUrl);
}

async function performReplace() {
  if (!state.campaignFile || !state.personFile || !state.maskDataUrl) return;

  el("processingModal").classList.remove("hidden");
  const stopStages = runStageAnimation();

  try {
    const maskBlob = await maskDataUrlToBlob(state.maskDataUrl);
    const response = await replacePerson({
      campaignFile: state.campaignFile,
      personFile: state.personFile,
      maskBlob,
      settings: state.settings,
    });

    const dataUrl = `data:image/png;base64,${response.image}`;
    state.generations.push({ dataUrl });
    state.activeGeneration = state.generations.length - 1;
    showResult();
  } catch (e) {
    showToast(e.message || "We couldn't process this image. Try a clearer reference photo or adjust the mask.");
  } finally {
    stopStages();
    el("processingModal").classList.add("hidden");
  }
}

el("replaceBtn").addEventListener("click", performReplace);
el("regenerateBtn").addEventListener("click", performReplace);

function showResult() {
  const gen = state.generations[state.activeGeneration];
  if (!gen) return;

  el("resultImg").src = gen.dataUrl;
  el("resultSection").classList.remove("hidden");
  el("resultSection").scrollIntoView({ behavior: "smooth", block: "start" });

  renderGenerationsRow();
}

function renderGenerationsRow() {
  const row = el("generationsRow");
  row.innerHTML = "";
  if (state.generations.length < 2) {
    row.classList.add("hidden");
    return;
  }
  row.classList.remove("hidden");
  state.generations.forEach((gen, idx) => {
    const thumb = document.createElement("div");
    thumb.className = `generation-thumb${idx === state.activeGeneration ? " active" : ""}`;
    const img = document.createElement("img");
    img.src = gen.dataUrl;
    img.alt = `Generation ${idx + 1}`;
    thumb.appendChild(img);
    thumb.addEventListener("click", () => {
      state.activeGeneration = idx;
      el("resultImg").src = gen.dataUrl;
      renderGenerationsRow();
    });
    row.appendChild(thumb);
  });
}

/* ---------------- Try another person ---------------- */

el("tryAnotherPersonBtn").addEventListener("click", () => {
  el("personRemoveBtn").click();
  document.getElementById("personUploader").scrollIntoView({ behavior: "smooth", block: "center" });
});

/* ---------------- Fullscreen ---------------- */

el("fullscreenBtn").addEventListener("click", () => {
  const gen = state.generations[state.activeGeneration];
  if (!gen) return;
  const win = window.open("", "_blank");
  if (!win) return showToast("Please allow pop-ups to view fullscreen.", "info");
  win.document.write(
    `<title>V80 AI Studio Result</title><body style="margin:0;background:#0b1533;display:flex;align-items:center;justify-content:center;min-height:100vh;"><img src="${gen.dataUrl}" style="max-width:100%;max-height:100vh;"/></body>`
  );
});

/* ---------------- Download ---------------- */

el("downloadBtn").addEventListener("click", () => {
  const gen = state.generations[state.activeGeneration];
  if (!gen) return;
  downloadImage(gen.dataUrl, el("formatSelect").value);
});

el("downloadOriginalBtn").addEventListener("click", () => {
  if (!state.campaignImg) return;
  downloadImage(state.campaignImg.src.startsWith("data:") ? state.campaignImg.src : toDataUrl(state.campaignImg), "png");
});

function toDataUrl(imgEl) {
  const c = document.createElement("canvas");
  c.width = imgEl.naturalWidth;
  c.height = imgEl.naturalHeight;
  c.getContext("2d").drawImage(imgEl, 0, 0);
  return c.toDataURL("image/png");
}

/* ---------------- Share ---------------- */

document.querySelectorAll(".share-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const gen = state.generations[state.activeGeneration];
    if (!gen) return;
    const platform = btn.dataset.share;

    if (typeof navigator.share === "function") {
      await shareToPlatform(platform, gen.dataUrl);
    } else {
      openShareModal(platform, gen.dataUrl);
    }
  });
});

function openShareModal(platform, dataUrl) {
  el("shareModalThumb").src = dataUrl;
  el("shareModalNote").textContent = "This platform requires you to upload the downloaded image manually.";
  const options = el("shareModalOptions");
  options.innerHTML = "";

  const addOption = (label, handler) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.addEventListener("click", handler);
    options.appendChild(b);
  };

  addOption("Download Image", () => downloadImage(dataUrl, "png"));
  addOption("Copy Image", () => copyImage(dataUrl));
  if (platform === "x") {
    addOption("Open X to Post", () => shareToPlatform("x", dataUrl));
  }

  el("shareModal").classList.remove("hidden");
}

el("shareModalClose").addEventListener("click", () => el("shareModal").classList.add("hidden"));
el("shareModal").addEventListener("click", (e) => {
  if (e.target.id === "shareModal") el("shareModal").classList.add("hidden");
});

/* ---------------- Init ---------------- */

refreshStatus();

// One fixed campaign image, applied on load — the user only uploads a person.
selectDefaultCampaign("man", "/static/assets/campaigns/man.png");
