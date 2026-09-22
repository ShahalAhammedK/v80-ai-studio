export function initBeforeAfterSlider() {
  const wrap = document.getElementById("sliderWrap");
  const clip = document.getElementById("sliderAfterClip");
  const handle = document.getElementById("sliderHandle");
  let dragging = false;

  function setPosition(clientX) {
    const rect = wrap.getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.min(100, Math.max(0, pct));
    clip.style.width = `${pct}%`;
    handle.style.left = `${pct}%`;
  }

  function start(e) {
    dragging = true;
    setPosition(e.touches ? e.touches[0].clientX : e.clientX);
  }
  function move(e) {
    if (!dragging) return;
    setPosition(e.touches ? e.touches[0].clientX : e.clientX);
  }
  function end() { dragging = false; }

  wrap.addEventListener("mousedown", start);
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", end);
  wrap.addEventListener("touchstart", start, { passive: true });
  window.addEventListener("touchmove", move, { passive: true });
  window.addEventListener("touchend", end);

  setPosition(wrap.getBoundingClientRect().left + wrap.getBoundingClientRect().width / 2);
}
