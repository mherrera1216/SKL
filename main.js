// main.js (SkyLens)
// - Animaciones (hero + stats)
// - Upload + llamada al backend FastAPI (/predict)
// - Toggle: Overlay / Máscara / Original + intensidad + resaltar

(() => {
  "use strict";

  const API_URL = "http://127.0.0.1:8000/predict";

  // ---------- Helpers ----------
  const $  = (sel, root=document) => root.querySelector(sel);
  const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function setText(el, txt){
    if (!el) return;
    el.textContent = txt;
  }

  // ---------- Navbar ----------
  function setupNavbar(){
    const nav = document.getElementById("navbar") || $(".navbar");
    if (!nav) return;

    const onScroll = () => {
      if (window.scrollY > 8) nav.classList.add("scrolled");
      else nav.classList.remove("scrolled");
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    // Botón "Comenzar análisis"
    const cta = $(".nav-cta");
    cta?.addEventListener("click", (e) => {
      e.preventDefault();
      document.getElementById("analizar")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  // Expuesto para los botones del HTML
  window.scrollToAnalyze = () => {
    document.getElementById("analizar")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  // ---------- Hero: barras (demo) ----------
  function setupHeroBars(){
    const g = document.getElementById("pb-green");
    const a = document.getElementById("pb-amber");
    const s = document.getElementById("pb-slate");
    if (!g || !a || !s) return;

    g.style.width = "0%";
    a.style.width = "0%";
    s.style.width = "0%";

    setTimeout(() => {
      g.style.width = "23%";
      a.style.width = "41%";
      s.style.width = "36%";
    }, 650);
  }

  // ---------- Stats banner: contador al hacer scroll ----------
  function animateNumber(el, to, suffix=""){
    const start = 0;
    const duration = 900;
    const t0 = performance.now();

    const isFloat = String(to).includes(".");
    const fmt = (v) => isFloat ? v.toFixed(1) : Math.round(v).toString();

    const tick = (t) => {
      const p = clamp((t - t0) / duration, 0, 1);
      const v = start + (to - start) * (1 - Math.pow(1 - p, 3));
      el.textContent = fmt(v) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function setupStatsCounters(){
    const banner = $(".stats-banner");
    if (!banner) return;

    const counters = $$("[data-counter]", banner);

    counters.forEach(el => {
      const suffix = el.getAttribute("data-suffix") || "";
      el.textContent = "0" + suffix;
      el.dataset.animated = "0";
    });

    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;

        counters.forEach((el) => {
          if (el.dataset.animated === "1") return;
          el.dataset.animated = "1";
          const target = parseFloat(el.getAttribute("data-counter") || "0");
          const suffix = el.getAttribute("data-suffix") || "";
          animateNumber(el, target, suffix);
        });

        io.disconnect();
      });
    }, { threshold: 0.35 });

    io.observe(banner);
  }

  // ---------- Scroll reveal ----------
  function setupReveal(){
    const revealTargets = $$(".step-card, .feat-card, .r-card, .stat-item");
    if (!revealTargets.length) return;

    revealTargets.forEach(el => el.classList.add("reveal"));

    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          en.target.classList.add("in");
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.14 });

    revealTargets.forEach(el => io.observe(el));
  }

  // ---------- Upload + IA ----------
  function setupUploader(){
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    const dzFileName = document.getElementById("dzFileName");
    const dzStatus = document.getElementById("dzStatus");
    const previewWrap = document.getElementById("previewWrap");
    const previewStack = document.getElementById("previewStack");

    const imgBase = document.getElementById("imgBase");
    const imgOverlay = document.getElementById("imgOverlay");
    const imgMask = document.getElementById("imgMask");

    const viewSwitcher = document.getElementById("viewSwitcher");
    const btnOverlay = document.getElementById("btnOverlay");
    const btnMask = document.getElementById("btnMask");
    const btnOriginal = document.getElementById("btnOriginal");

    const overlayOpacity = document.getElementById("overlayOpacity");
    const overlayOpacityValue = document.getElementById("overlayOpacityValue");
    const highlightToggle = document.getElementById("highlightToggle");

    const donutChart = document.getElementById("donutChart");
    const donutPct = document.getElementById("donutPct");
    const pctVeg = document.getElementById("pctVeg");
    const pctPav = document.getElementById("pctPav");
    const pctStr = document.getElementById("pctStr");

    const zoneBadge = document.getElementById("zoneBadge");
    const recoCount = document.getElementById("recoCount");
    const recoList = document.getElementById("recoList");

    if (!dropZone || !fileInput) return;

    let baseObjectUrl = null;
    let currentView = "overlay";

    function setBadge(text, kind){
      if (!zoneBadge) return;
      zoneBadge.classList.remove("badge-red", "badge-amber", "badge-green");
      if (kind === "green") zoneBadge.classList.add("badge-green");
      else if (kind === "amber") zoneBadge.classList.add("badge-amber");
      else zoneBadge.classList.add("badge-red");
      zoneBadge.textContent = text;
    }

    function setView(view){
      currentView = view;

      [btnOverlay, btnMask, btnOriginal].forEach(b => b?.classList.remove("active"));
      if (view === "overlay") btnOverlay?.classList.add("active");
      if (view === "mask") btnMask?.classList.add("active");
      if (view === "original") btnOriginal?.classList.add("active");

      previewStack?.classList.remove("mask-mode", "original-mode");
      if (view === "mask") previewStack?.classList.add("mask-mode");
      if (view === "original") previewStack?.classList.add("original-mode");

      if (imgOverlay){
        imgOverlay.style.opacity = view === "overlay"
          ? (parseInt(overlayOpacity?.value || "75", 10) / 100).toString()
          : "1";
      }
    }

    function setHighlight(on){
      if (!previewStack) return;
      if (on) previewStack.classList.add("highlight");
      else previewStack.classList.remove("highlight");
    }

    function updateOverlayOpacity(){
      if (!overlayOpacity || !overlayOpacityValue) return;
      const v = parseInt(overlayOpacity.value, 10);
      overlayOpacityValue.textContent = `${v}%`;
      if (imgOverlay && currentView === "overlay"){
        imgOverlay.style.opacity = (v / 100).toString();
      }
    }

    function updateDonut(veg, pav, str){
      const p1 = clamp(veg, 0, 100);
      const p2 = clamp(pav, 0, 100);
      const p3 = clamp(str, 0, 100);

      const a1 = p1;
      const a2 = p1 + p2;

      if (donutChart){
        donutChart.style.background =
          `conic-gradient(var(--leaf-600) 0 ${a1}%, var(--amber-500) ${a1}% ${a2}%, var(--slate-400) ${a2}% 100%)`;
      }
      setText(donutPct, `${p1.toFixed(1)}%`);
      setText(pctVeg, `${p1.toFixed(1)}%`);
      setText(pctPav, `${p2.toFixed(1)}%`);
      setText(pctStr, `${p3.toFixed(1)}%`);

      if (p1 < 15 && (p2 + p3) > 85) setBadge("Zona crítica", "red");
      else if (p1 < 25) setBadge("Zona deficiente", "amber");
      else setBadge("Zona equilibrada", "green");
    }

    function buildRecoItem(text, idx, severity){
      const icon = severity === "high" ? "🌿" : (severity === "med" ? "🧠" : "💧");
      const prioClass = severity === "high" ? "prio-high" : (severity === "med" ? "prio-med" : "prio-low");
      const prioLabel = severity === "high" ? "ALTA" : (severity === "med" ? "MEDIA" : "BAJA");

      let title = `Recomendación ${idx+1}`;
      let desc = text;
      const parts = text.split(":");
      if (parts.length >= 2){
        title = parts[0].trim();
        desc = parts.slice(1).join(":").trim();
      }

      const div = document.createElement("div");
      div.className = "reco-item";
      div.innerHTML = `
        <div class="reco-icon">${icon}</div>
        <div class="reco-body">
          <div class="reco-title">${title}</div>
          <div class="reco-text">${desc}</div>
        </div>
        <span class="prio ${prioClass}">${prioLabel}</span>
      `;
      return div;
    }

    function updateRecommendations(recos, veg, pav, str){
      if (!recoList || !recoCount) return;

      recoList.innerHTML = "";

      let list = Array.isArray(recos) ? recos.filter(Boolean) : [];
      if (!list.length){
        const auto = [];
        if (veg < 25) auto.push("Aumentar cobertura vegetal: arborización, corredores verdes y zonas de sombra.");
        if (pav > 20) auto.push("Reducir impermeabilización: pavimento permeable y jardines de lluvia.");
        if (str > 60) auto.push("Mejorar confort térmico: techos fríos/techos verdes y sombreamiento.");
        list = auto.length ? auto : ["Zona equilibrada: conservar cobertura verde y mejorar conectividad ecológica."];
      }

      const items = list.slice(0, 3).map((t, i) => {
        let sev = "low";
        if (i === 0) sev = "high";
        else if (i === 1) sev = "med";
        return buildRecoItem(t, i, sev);
      });

      items.forEach(n => recoList.appendChild(n));
      recoCount.textContent = `${items.length} acciones`;
    }

    function parsePctByClass(pct){
      const get = (k) => {
        if (!pct) return 0;
        const v = pct[k] ?? pct[String(k)];
        const f = parseFloat(v);
        return Number.isFinite(f) ? f : 0;
      };
      const str = get(1);
      const veg = get(2);
      const pav = get(4);
      const denom = str + veg + pav;
      if (denom <= 0) return { veg: 0, pav: 0, str: 0 };
      return { veg: (veg/denom)*100, pav: (pav/denom)*100, str: (str/denom)*100 };
    }

    function resetUI(){
      setText(dzFileName, "Arrastra tu imagen aquí");
      setText(dzStatus, "");
      dzStatus?.classList.remove("ok", "err");
      previewWrap?.setAttribute("hidden", "hidden");
      viewSwitcher?.setAttribute("hidden", "hidden");
      updateDonut(0,0,0);
      recoList && (recoList.innerHTML = "");
      recoCount && (recoCount.textContent = "0 acciones");

      if (baseObjectUrl){
        URL.revokeObjectURL(baseObjectUrl);
        baseObjectUrl = null;
      }
      if (imgBase) imgBase.src = "";
      if (imgOverlay) imgOverlay.src = "";
      if (imgMask) imgMask.src = "";
    }

    async function analyzeFile(file){
      if (!file) return;

      const name = file.name || "";
      const ext = name.split(".").pop()?.toLowerCase() || "";
      const okExt = ["jpg","jpeg","png","webp","tif","tiff"];
      if (!okExt.includes(ext)){
        setText(dzStatus, "✖ Formato no soportado. Usa JPG/PNG/TIFF/GeoTIFF/WebP.");
        dzStatus?.classList.add("err");
        return;
      }

      dzStatus?.classList.remove("ok", "err");
      setText(dzFileName, name);
      setText(dzStatus, "Procesando… (enviando a la IA)");

      if (baseObjectUrl) URL.revokeObjectURL(baseObjectUrl);
      baseObjectUrl = URL.createObjectURL(file);

      previewWrap?.removeAttribute("hidden");
      if (imgBase) imgBase.src = baseObjectUrl;

      if (imgMask) imgMask.src = "";
      if (imgOverlay) imgOverlay.src = "";

      dropZone.classList.add("loading");

      try{
        const fd = new FormData();
        fd.append("file", file);

        const res = await fetch(API_URL, { method:"POST", body: fd });
        if (!res.ok){
          const txt = await res.text();
          throw new Error(txt || `HTTP ${res.status}`);
        }
        const data = await res.json();

        if (imgOverlay && data.overlay_png) imgOverlay.src = data.overlay_png;
        if (imgMask && data.mask_png) imgMask.src = data.mask_png;

        viewSwitcher?.removeAttribute("hidden");
        updateOverlayOpacity();
        setHighlight(!!highlightToggle?.checked);
        setView("overlay");

        const { veg, pav, str } = parsePctByClass(data.pct_by_class);
        updateDonut(veg, pav, str);
        updateRecommendations(data.recommendations, veg, pav, str);

        setText(dzStatus, "✅ ¡Análisis completo!");
        dzStatus?.classList.add("ok");
      }catch(err){
        console.error(err);
        setText(dzStatus, "✖ Error: no pude analizar la imagen. Revisa el backend (http://127.0.0.1:8000/health).");
        dzStatus?.classList.add("err");
      }finally{
        dropZone.classList.remove("loading");
      }
    }

    // File picker (para botones del HTML)
    function openFilePicker(){
      fileInput.value = "";
      fileInput.click();
    }
    window.openFilePicker = openFilePicker;

    // Click en dropzone
    dropZone.addEventListener("click", () => openFilePicker());
    dropZone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " "){
        e.preventDefault();
        openFilePicker();
      }
    });

    // Evitar que clicks en controles abran selector
    viewSwitcher?.addEventListener("click", (e) => e.stopPropagation());

    // input change
    fileInput.addEventListener("change", (e) => {
      const f = e.target.files?.[0];
      if (f) analyzeFile(f);
    });

    // Drag&Drop
    ["dragenter","dragover"].forEach(evt => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropZone.classList.add("drag");
      });
    });
    ["dragleave","drop"].forEach(evt => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropZone.classList.remove("drag");
      });
    });
    dropZone.addEventListener("drop", (e) => {
      const f = e.dataTransfer?.files?.[0];
      if (f) analyzeFile(f);
    });

    // Controles
    btnOverlay?.addEventListener("click", () => setView("overlay"));
    btnMask?.addEventListener("click", () => setView("mask"));
    btnOriginal?.addEventListener("click", () => setView("original"));

    overlayOpacity?.addEventListener("input", updateOverlayOpacity);
    highlightToggle?.addEventListener("change", (e) => setHighlight(e.target.checked));

    resetUI();
  }

  // ---------- Init ----------
  document.addEventListener("DOMContentLoaded", () => {
    setupNavbar();
    setupHeroBars();
    setupStatsCounters();
    setupReveal();
    setupUploader();
  });

})();