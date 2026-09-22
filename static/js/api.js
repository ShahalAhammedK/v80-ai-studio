async function parseJsonSafe(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

async function friendlyFetch(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("We couldn't reach the server. Check your connection and try again.");
  }
  const body = await parseJsonSafe(response);
  if (!response.ok) {
    throw new Error((body && body.error) || "Something went wrong. Please try again.");
  }
  return body;
}

export async function getHealth() {
  try {
    return await friendlyFetch("/api/health");
  } catch {
    return { aiReady: false, autoMaskAvailable: false };
  }
}

export async function segmentCampaign(file) {
  const form = new FormData();
  form.append("campaign", file);
  return friendlyFetch("/api/segment", { method: "POST", body: form });
}

export async function detectGender(file) {
  const form = new FormData();
  form.append("person", file);
  try {
    return await friendlyFetch("/api/detect-gender", { method: "POST", body: form });
  } catch {
    return { gender: "unknown" };
  }
}

export async function replacePerson({ campaignFile, personFile, maskBlob, settings }) {
  const form = new FormData();
  form.append("campaign", campaignFile);
  form.append("person", personFile);
  form.append("mask", maskBlob, "mask.png");
  form.append("settings", JSON.stringify(settings));
  return friendlyFetch("/api/replace-person", { method: "POST", body: form });
}
