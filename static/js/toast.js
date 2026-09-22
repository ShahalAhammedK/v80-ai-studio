const stack = () => document.getElementById("toastStack");

export function showToast(message, type = "error", timeout = 5000) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  stack().appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity .2s ease";
    setTimeout(() => el.remove(), 200);
  }, timeout);
}
