(function () {
  const slides = Array.from(document.querySelectorAll(".slide"));
  if (!slides.length) return;

  const total = slides.length;
  const progress = document.createElement("div");
  const progressBar = document.createElement("div");
  const presentButton = document.createElement("button");
  progress.className = "progress";
  progressBar.className = "progress-bar";
  progress.appendChild(progressBar);
  document.body.appendChild(progress);

  presentButton.className = "present-button";
  presentButton.type = "button";
  presentButton.title = "进入播放模式 (F)";
  presentButton.setAttribute("aria-label", "进入播放模式");
  presentButton.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M16 3h3a2 2 0 0 1 2 2v3"/><path d="M8 21H5a2 2 0 0 1-2-2v-3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>';
  document.body.appendChild(presentButton);

  let current = initialIndex();

  function initialIndex() {
    const hash = window.location.hash.replace(/^#\/?/, "");
    const parsed = Number.parseInt(hash, 10);
    if (Number.isFinite(parsed) && parsed >= 1 && parsed <= total) {
      return parsed - 1;
    }
    return 0;
  }

  function show(index, updateHash = true) {
    current = Math.max(0, Math.min(total - 1, index));

    slides.forEach((slide, slideIndex) => {
      const active = slideIndex === current;
      slide.classList.toggle("active", active);
      slide.setAttribute("aria-hidden", active ? "false" : "true");

      if (active) {
        requestAnimationFrame(() => {
          slide.querySelectorAll("[data-anim]").forEach((el, animIndex) => {
            el.style.transitionDelay = `${animIndex * 80}ms`;
            el.classList.add("is-visible");
          });
        });
      } else {
        slide.querySelectorAll("[data-anim]").forEach((el) => {
          el.classList.remove("is-visible");
          el.style.transitionDelay = "";
        });
      }
    });

    document.querySelectorAll(".slide-number").forEach((el) => {
      el.dataset.current = String(current + 1);
      el.dataset.total = String(total);
    });

    progressBar.style.width = `${((current + 1) / total) * 100}%`;

    if (updateHash) {
      window.history.replaceState(null, "", `#${current + 1}`);
    }
  }

  function next() {
    show(current + 1);
  }

  function previous() {
    show(current - 1);
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) {
      document.exitFullscreen();
      return;
    }

    const target = document.documentElement;
    if (target.requestFullscreen) {
      target.requestFullscreen();
    }
  }

  presentButton.addEventListener("click", toggleFullscreen);

  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const isTyping = target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
    if (isTyping) return;

    if (["ArrowRight", "ArrowDown", "PageDown", " ", "Enter"].includes(event.key)) {
      event.preventDefault();
      next();
    } else if (["ArrowLeft", "ArrowUp", "PageUp", "Backspace"].includes(event.key)) {
      event.preventDefault();
      previous();
    } else if (event.key === "Home") {
      event.preventDefault();
      show(0);
    } else if (event.key === "End") {
      event.preventDefault();
      show(total - 1);
    } else if (event.key.toLowerCase() === "f") {
      event.preventDefault();
      toggleFullscreen();
    }
  });

  let touchStartX = 0;
  document.addEventListener("touchstart", (event) => {
    touchStartX = event.changedTouches[0].clientX;
  }, { passive: true });

  document.addEventListener("touchend", (event) => {
    const dx = event.changedTouches[0].clientX - touchStartX;
    if (Math.abs(dx) < 48) return;
    if (dx < 0) next();
    else previous();
  }, { passive: true });

  window.addEventListener("hashchange", () => {
    show(initialIndex(), false);
  });

  show(current, false);
})();
