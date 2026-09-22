import { showToast } from "./toast.js";

const FILENAME_BASE = "v80-ai-person-replacement";

export function dataUrlToBlob(dataUrl) {
  const [meta, b64] = dataUrl.split(",");
  const mime = meta.match(/:(.*?);/)[1];
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

export async function toFormatBlob(pngDataUrl, format) {
  if (format === "png") return dataUrlToBlob(pngDataUrl);

  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = pngDataUrl;
  });
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
}

export function downloadBlob(blob, filename) {
  try {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  } catch {
    showToast("We couldn't start the download. Please try again.");
  }
}

export async function downloadImage(pngDataUrl, format) {
  try {
    const blob = await toFormatBlob(pngDataUrl, format);
    downloadBlob(blob, `${FILENAME_BASE}.${format === "jpg" ? "jpg" : "png"}`);
  } catch {
    showToast("We couldn't prepare that download. Please try again.");
  }
}

export function canShareFiles(file) {
  return typeof navigator.share === "function" &&
    typeof navigator.canShare === "function" &&
    navigator.canShare({ files: [file] });
}

async function blobToFile(blob) {
  return new File([blob], `${FILENAME_BASE}.png`, { type: "image/png" });
}

export async function nativeShare(pngDataUrl, text) {
  const blob = dataUrlToBlob(pngDataUrl);
  const file = await blobToFile(blob);
  if (!canShareFiles(file)) return false;
  try {
    await navigator.share({ files: [file], text, title: "V80 AI Studio" });
    return true;
  } catch (err) {
    if (err && err.name === "AbortError") return true; // user cancelled, not an error
    return false;
  }
}

export async function copyImage(pngDataUrl) {
  if (!navigator.clipboard || typeof window.ClipboardItem === "undefined") {
    showToast("Image copying isn't supported in this browser. Download the image instead.", "info");
    return;
  }
  try {
    const blob = dataUrlToBlob(pngDataUrl);
    await navigator.clipboard.write([new window.ClipboardItem({ [blob.type]: blob })]);
    showToast("Image copied to clipboard.", "success");
  } catch {
    showToast("Image copying isn't supported in this browser. Download the image instead.", "info");
  }
}

export async function shareToPlatform(platform, pngDataUrl) {
  const text = "Created with V80 AI Studio ✨";

  if (platform === "x") {
    const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`;
    window.open(url, "_blank", "noopener,noreferrer");
    showToast("Download your image to attach it — X's share link can't attach images directly.", "info", 6000);
    return;
  }

  const shared = await nativeShare(pngDataUrl, text);
  if (shared) return;

  const messages = {
    instagram: "Your image is ready. Download it and upload it to Instagram.",
    tiktok: "Your image is ready. Download it and upload it to TikTok.",
    snapchat: "Your image is ready. Download it and share it through Snapchat.",
  };
  showToast(messages[platform] || "Your image is ready. Download it to share.", "info", 6000);
}
