// Mirrors src/data/features.py MANUFACTURERS / BRAND_MODELS so the dropdowns
// only offer values the backend's regex extraction can actually recognise
// from a constructed title -- picking a brand+model here always lands the
// prediction in the "full_info" segment.
const BRAND_MODELS = {
  "Volkswagen": ["Golf", "Polo", "Tiguan", "Passat", "Touran", "Touareg", "Jetta", "Caddy", "Up", "Amarok", "T-Roc", "Sharan"],
  "Renault": ["Clio", "Megane", "Captur", "Symbol", "Kadjar", "Talisman", "Kangoo", "Twingo", "Scenic", "Fluence", "Koleos"],
  "Peugeot": ["308", "208", "3008", "2008", "508", "Partner", "5008", "206", "207", "306", "406", "Boxer", "301", "205"],
  "Mercedes-Benz": ["Classe A", "Classe B", "Classe C", "Classe E", "Classe S", "GLC", "GLA", "GLE", "ML", "Vito", "Sprinter", "CLA", "C220", "C180", "C200", "E220", "E200"],
  "Dacia": ["Logan", "Duster", "Sandero", "Lodgy", "Dokker", "Stepway"],
  "Hyundai": ["Tucson", "Santa Fe", "Accent", "i10", "i20", "i30", "ix35", "Elantra", "Creta", "Getz", "H100"],
  "Ford": ["Fiesta", "Focus", "Kuga", "EcoSport", "Mondeo", "Ranger", "Transit", "Puma", "Fusion", "Everest"],
  "Fiat": ["Doblo", "Punto", "500", "Tipo", "Panda", "Uno", "Palio"],
  "Audi": ["A3", "A4", "A5", "A6", "Q3", "Q5", "Q7", "A1", "TT"],
  "BMW": ["Serie 1", "Serie 2", "Serie 3", "Serie 4", "Serie 5", "Serie 7", "X1", "X3", "X5", "X6"],
  "Kia": ["Picanto", "Rio", "Sportage", "Sorento", "Cerato", "Ceed", "Stonic"],
  "Citroen": ["C3", "C4", "C5", "Berlingo", "Xsara", "Jumpy", "Nemo", "C-Elysee", "C1", "Picasso"],
  "Opel": ["Corsa", "Astra", "Insignia", "Mokka", "Vectra", "Zafira", "Crossland"],
  "Toyota": ["Corolla", "Yaris", "RAV4", "Land Cruiser", "Hilux", "Camry", "Auris", "Prado"],
  "Land Rover": ["Range Rover", "Discovery", "Defender", "Evoque", "Freelander", "Velar"],
  "Nissan": ["Qashqai", "Micra", "Juke", "Note", "X-Trail", "Navara"],
  "Seat": ["Ibiza", "Leon", "Arona", "Ateca"],
  "Jeep": ["Renegade", "Compass", "Grand Cherokee", "Cherokee", "Wrangler"],
  "Skoda": ["Fabia", "Octavia", "Superb", "Rapid", "Kodiaq"],
  "Honda": ["Civic", "CR-V", "Accord", "Jazz"],
  "Suzuki": ["Alto", "Celerio", "Swift", "Vitara", "Jimny", "Baleno"],
  "Volvo": ["V40", "XC90", "XC60", "S60", "V60"],
  "Porsche": ["Cayenne", "Macan", "Panamera", "911"],
  "Mini": ["Cooper", "Countryman"],
};

// Other manufacturers seen in the market without a curated model list yet --
// still valid for brand-level detection.
const OTHER_BRANDS = [
  "Alfa Romeo", "Great Wall", "MG", "Chevrolet", "Chrysler", "Dodge",
  "Jaguar", "Lexus", "Infiniti", "Acura", "Buick", "Cadillac", "GMC",
  "Lincoln", "Maserati", "Ferrari", "Lamborghini", "Bentley", "Rolls-Royce",
  "Aston Martin", "Bugatti", "Tesla", "BYD", "Geely", "Chery", "DFSK",
  "Changan", "JAC", "Haval", "Isuzu", "Daihatsu", "SsangYong", "Lada",
  "Iveco", "Smart", "DS", "Abarth", "Lancia", "Subaru", "Mazda", "Mitsubishi",
];

const CITIES = [
  "Casablanca", "Marrakech", "Tanger", "Rabat", "Agadir", "Fès", "Salé",
  "Meknès", "Kénitra", "Temara", "Tétouan", "El Jadida", "Mohammedia",
  "Oujda", "Safi", "Nador", "Béni Mellal", "Laâyoune", "Berrechid",
  "Settat", "Khouribga", "Bouskoura", "Taza", "Larache", "Bouznika",
];

const ALL_BRANDS = [...Object.keys(BRAND_MODELS), ...OTHER_BRANDS].sort((a, b) => a.localeCompare(b));

// ---------------------------------------------------------------------------
// Hero video: skip/stop autoplay for users who asked for reduced motion
// ---------------------------------------------------------------------------
const heroVideo = document.querySelector(".hero-video");
if (heroVideo && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  heroVideo.removeAttribute("autoplay");
  heroVideo.pause();
}

// ---------------------------------------------------------------------------
// Rise section: the car climbs up through the giant numeral as you scroll
// ---------------------------------------------------------------------------
const riseWrap = document.querySelector(".rise-wrap");
const riseCar = document.querySelector(".rise-car");
if (riseWrap && riseCar) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    riseCar.style.transform = "translate(-50%, -50%)";
  } else {
    const updateRise = () => {
      const rect = riseWrap.getBoundingClientRect();
      const total = rect.height + window.innerHeight;
      const progress = Math.min(1, Math.max(0, (window.innerHeight - rect.top) / total));
      const translateY = 65 - progress * 95; // rises from below into view, past center
      riseCar.style.transform = `translate(-50%, -50%) translateY(${translateY}%)`;
      requestAnimationFrame(updateRise);
    };
    requestAnimationFrame(updateRise);
  }
}

// ---------------------------------------------------------------------------
// Smooth scrolling (Lenis)
// ---------------------------------------------------------------------------
if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches && window.Lenis) {
  const lenis = new Lenis({
    duration: 1.15,
    easing: (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t)),
    smoothWheel: true,
    // Let scrolling inside a native <select> dropdown (its options list)
    // scroll the dropdown itself instead of hijacking the whole page.
    prevent: (node) => node.tagName === "OPTION" || !!node.closest("select"),
  });
  function raf(time) {
    lenis.raf(time);
    requestAnimationFrame(raf);
  }
  requestAnimationFrame(raf);

  // Keep in-page anchor links (e.g. the "Estimer" nav CTA) working smoothly.
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      const target = document.querySelector(link.getAttribute("href"));
      if (!target) return;
      e.preventDefault();
      lenis.scrollTo(target, { offset: -20 });
    });
  });
}

// ---------------------------------------------------------------------------
// Scroll reveal: fade/slide-up on intersect, word-mask split for headlines
// ---------------------------------------------------------------------------
document.querySelectorAll("[data-reveal-words]").forEach((el) => {
  const words = el.textContent.trim().split(/\s+/);
  el.textContent = "";
  words.forEach((word, i) => {
    const mask = document.createElement("span");
    mask.className = "word-mask";
    const inner = document.createElement("span");
    inner.className = "word-inner";
    inner.textContent = word;
    inner.style.transitionDelay = `${i * 0.06}s`;
    mask.appendChild(inner);
    el.appendChild(mask);
    if (i < words.length - 1) el.appendChild(document.createTextNode(" "));
  });
});

function isAboveFold(el) {
  const rect = el.getBoundingClientRect();
  return rect.top < window.innerHeight * 0.92 && rect.bottom > 0;
}

function setupReveal() {
  const revealTargets = document.querySelectorAll("[data-reveal], [data-reveal-words]");

  if (!("IntersectionObserver" in window)) {
    revealTargets.forEach((el) => el.classList.add("is-visible"));
    return;
  }

  // Chromium's very first IntersectionObserver callback after a fresh
  // navigation can be unreliable in either direction (reports elements
  // on-screen as not intersecting, or off-screen ones as intersecting), so
  // the geometry is re-checked directly rather than trusting isIntersecting.
  const revealObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (isAboveFold(entry.target)) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.15, rootMargin: "0px 0px -8% 0px" }
  );
  revealTargets.forEach((el) => {
    if (isAboveFold(el)) {
      el.classList.add("is-visible");
    } else {
      revealObserver.observe(el);
    }
  });
}

// Fonts load asynchronously (display=swap) and can reflow the layout after
// the script's first pass runs, so wait for them plus two animation frames
// (layout + paint settled) before measuring positions for the reveal setup.
const fontsReady = document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve();
fontsReady.then(() => {
  requestAnimationFrame(() => requestAnimationFrame(setupReveal));
});

const fmtMAD = (n) => `${Math.round(n).toLocaleString("fr-FR")} DH`;

// ---------------------------------------------------------------------------
// Populate static selects
// ---------------------------------------------------------------------------
const brandSelect = document.getElementById("brand");
const modelSelect = document.getElementById("model");
const citySelect = document.getElementById("city");

for (const brand of ALL_BRANDS) {
  const opt = document.createElement("option");
  opt.value = brand;
  opt.textContent = brand;
  brandSelect.appendChild(opt);
}

for (const city of CITIES) {
  const opt = document.createElement("option");
  opt.value = city;
  opt.textContent = city;
  citySelect.appendChild(opt);
}
citySelect.value = "Casablanca";

function refreshModelOptions() {
  const brand = brandSelect.value;
  modelSelect.innerHTML = "";
  const models = BRAND_MODELS[brand];

  if (!models) {
    modelSelect.disabled = true;
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = brand ? "— Aucun modèle répertorié —" : "— Sélectionnez une marque —";
    modelSelect.appendChild(opt);
    return;
  }

  modelSelect.disabled = false;
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "— Non renseigné —";
  modelSelect.appendChild(blank);

  for (const model of models) {
    const opt = document.createElement("option");
    opt.value = model;
    opt.textContent = model;
    modelSelect.appendChild(opt);
  }
}

brandSelect.addEventListener("change", refreshModelOptions);
refreshModelOptions();

// ---------------------------------------------------------------------------
// Dealer segmented toggle
// ---------------------------------------------------------------------------
const dealerToggle = document.getElementById("dealerToggle");
const isDealerInput = document.getElementById("isDealer");

dealerToggle.addEventListener("click", (e) => {
  const btn = e.target.closest(".seg-btn");
  if (!btn) return;
  for (const b of dealerToggle.querySelectorAll(".seg-btn")) {
    b.classList.toggle("active", b === btn);
    b.setAttribute("aria-checked", b === btn ? "true" : "false");
  }
  isDealerInput.value = btn.dataset.value;
});

// ---------------------------------------------------------------------------
// Theme toggle (persisted per-browser only; harmless if storage is blocked)
// ---------------------------------------------------------------------------
const themeToggle = document.getElementById("themeToggle");
const iconSun = document.getElementById("iconSun");
const iconMoon = document.getElementById("iconMoon");

// Dark is the site's default identity (no attribute = dark); "light" is the
// only explicit override the toggle can set.
function applyTheme(theme) {
  if (theme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  const isDark = theme !== "light";
  iconSun.style.display = isDark ? "none" : "block";
  iconMoon.style.display = isDark ? "block" : "none";
}

function currentTheme() {
  try { return localStorage.getItem("theme") || "dark"; }
  catch { return "dark"; }
}

applyTheme(currentTheme());

themeToggle.addEventListener("click", () => {
  const isDark = document.documentElement.getAttribute("data-theme") !== "light";
  const next = isDark ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch {}
});

// ---------------------------------------------------------------------------
// Form submission
// ---------------------------------------------------------------------------
const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const btnLabel = submitBtn.querySelector(".btn-label");
const btnSpinner = submitBtn.querySelector(".btn-spinner");
const formError = document.getElementById("formError");

const resultEmpty = document.getElementById("resultEmpty");
const resultContent = document.getElementById("resultContent");
const confidenceBadge = document.getElementById("confidenceBadge");
const confidenceLabel = document.getElementById("confidenceLabel");
const typicalError = document.getElementById("typicalError");
const rangeLow = document.getElementById("rangeLow");
const rangeHigh = document.getElementById("rangeHigh");
const estimateMain = document.getElementById("estimateMain");
const detectedBrand = document.getElementById("detectedBrand");
const detectedModel = document.getElementById("detectedModel");

const CONFIDENCE_LABELS = {
  high: "Confiance élevée",
  medium: "Confiance moyenne",
  low: "Confiance faible",
};

function buildTitle({ brand, model, fuel, transmission, year, city }) {
  const parts = [];
  if (brand) parts.push(brand);
  if (model) parts.push(model);
  if (fuel) parts.push(fuel);
  if (transmission) parts.push(transmission);
  parts.push(String(year));
  if (city) parts.push(`à ${city}`);
  return parts.join(" ");
}

function setLoading(loading) {
  submitBtn.disabled = loading;
  btnLabel.textContent = loading ? "Estimation…" : "Estimer le prix";
  btnSpinner.style.display = loading ? "inline-block" : "none";
}

function showError(message) {
  formError.textContent = message;
  formError.classList.add("show");
}

function clearError() {
  formError.textContent = "";
  formError.classList.remove("show");
}

function renderResult(data) {
  resultEmpty.hidden = true;
  resultContent.hidden = false;

  confidenceBadge.className = `confidence-badge ${data.confidence}`;
  confidenceLabel.textContent = CONFIDENCE_LABELS[data.confidence] || data.confidence;
  typicalError.textContent = `erreur typique ±${data.typical_error_pct}%`;

  rangeLow.textContent = fmtMAD(data.range_low_mad);
  rangeHigh.textContent = fmtMAD(data.range_high_mad);
  estimateMain.textContent = fmtMAD(data.estimate_mad);

  if (data.detected_brand) {
    detectedBrand.textContent = data.detected_brand;
    detectedBrand.classList.remove("muted");
  } else {
    detectedBrand.textContent = "Non détecté";
    detectedBrand.classList.add("muted");
  }

  if (data.detected_model) {
    detectedModel.textContent = data.detected_model;
    detectedModel.classList.remove("muted");
  } else {
    detectedModel.textContent = "Non détecté";
    detectedModel.classList.add("muted");
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();

  const brand = brandSelect.value || null;
  const model = modelSelect.value || null;
  const year = parseInt(document.getElementById("year").value, 10);
  const mileageRaw = document.getElementById("mileage").value;
  const fuel = document.getElementById("fuel").value || null;
  const transmission = document.getElementById("transmission").value || null;
  const city = document.getElementById("city").value.trim() || null;
  const nPhotos = parseInt(document.getElementById("photos").value, 10) || 0;
  const isDealer = parseInt(isDealerInput.value, 10);

  if (!year || year < 1970 || year > 2027) {
    showError("Merci d'indiquer une année valide (1970–2027).");
    return;
  }

  const title = buildTitle({ brand, model, fuel, transmission, year, city });

  const payload = {
    year,
    mileage_km: mileageRaw ? Number(mileageRaw) : null,
    fuel,
    transmission,
    city,
    title,
    is_dealer: isDealer,
    n_photos: nPhotos,
  };

  setLoading(true);
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Erreur serveur (${res.status})`);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    showError(err.message || "Impossible de contacter l'API. Vérifiez qu'elle est bien démarrée.");
  } finally {
    setLoading(false);
  }
});
