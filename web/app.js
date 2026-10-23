"use strict";
// In-browser inference for scikit-learn GradientBoostingRegressor models exported to JSON (B feature excluded).
const INFO = {
  RM: ["Average rooms per dwelling", 0.01], LSTAT: ["% lower-status population", 0.01], PTRATIO: ["Pupil–teacher ratio", 0.1],
  NOX: ["Nitric-oxide concentration (pp10m)", 0.001], CRIM: ["Per-capita crime rate", 0.01], DIS: ["Distance to employment centres", 0.01],
  AGE: ["% homes built before 1940", 0.1], TAX: ["Property tax per $10,000", 1],
  ZN: ["% land zoned for large lots", 0.5], INDUS: ["% non-retail business acres", 0.01], RAD: ["Highway access index", 1], CHAS: ["Bounds the Charles River (0/1)", 1],
};
const MAIN = ["RM", "LSTAT", "PTRATIO", "NOX", "CRIM", "DIS", "AGE", "TAX"], MORE = ["ZN", "INDUS", "RAD", "CHAS"];
const COVERAGE = window.COVERAGE || null;
let model, meta;
const f32 = Math.fround, $ = id => document.getElementById(id);
function tree(t, x) { let n = 0; while (t.l[n] !== -1) n = f32(x[t.f[n]]) <= t.t[n] ? t.l[n] : t.r[n]; return t.v[n]; }
function gbr(m, x) { let s = 0; for (const t of m.trees) s += tree(t, x); return m.base + m.lr * s; }
function predict(raw) { const x = model.features.map(f => raw[f]); return {point: Math.exp(gbr(model.point, x)), lo: Math.exp(gbr(model.lo, x)), hi: Math.exp(gbr(model.hi, x))}; }
window.predictRaw = predict;

const usd = v => "$" + Math.round(v * 1000).toLocaleString("en-US");
function values() { const o = {}; model.features.forEach(f => o[f] = parseFloat($(f).value)); return o; }
function slider(f, box) {
  const [label, step] = INFO[f], d = document.createElement("div"); d.className = "field";
  d.innerHTML = `<label for="${f}">${label} <output id="o-${f}" for="${f}"></output></label><small>${f}</small>
    <input type="range" id="${f}" min="${meta.min[f]}" max="${meta.max[f]}" step="${step}" value="${meta.median[f]}">`;
  box.appendChild(d); d.querySelector("input").addEventListener("input", render);
}
function render() {
  const r = values(), p = predict(r);
  model.features.forEach(f => { $("o-" + f).textContent = f === "CHAS" ? (r[f] ? "yes" : "no") : r[f]; });
  $("price").textContent = usd(p.point);
  $("range").textContent = `${usd(p.lo)} – ${usd(p.hi)}`;
  const sc = v => Math.min(100, Math.max(0, (v - 5) / 45 * 100));
  $("band").style.left = sc(p.lo) + "%"; $("band").style.width = (sc(p.hi) - sc(p.lo)) + "%"; $("pin").style.left = sc(p.point) + "%";
  const below = meta.medv.filter(v => v < p.point).length;
  $("pct").textContent = `Higher than ${Math.round(below / meta.medv.length * 100)}% of the 506 recorded tracts.`;
  drawHist(p.point);
}
function drawHist(v) {
  const c = $("hist"), x = c.getContext("2d"), W = c.width, H = c.height, bins = new Array(23).fill(0);
  meta.medv.forEach(m => bins[Math.min(22, Math.floor((m - 5) / 2))]++);
  const mx = Math.max(...bins), bw = (W - 20) / bins.length;
  x.clearRect(0, 0, W, H);
  bins.forEach((b, i) => { x.fillStyle = i % 2 ? "#1f2f4a" : "#8b1e1e"; const h = b / mx * (H - 30); x.fillRect(10 + i * bw + 1, H - 18 - h, bw - 2, h); });
  x.fillStyle = "#6b5442"; x.font = "12px Inter, sans-serif"; ["$5k", "$27k", "$50k"].forEach((t, i) => x.fillText(t, 10 + i * (W - 50) / 2, H - 4));
  const px = 10 + (v - 5) / 46 * (W - 20); x.fillStyle = "#e8a33d"; x.fillRect(px - 2, 4, 4, H - 22);
  x.font = "16px serif"; x.fillText("★", px - 6, 16);
}
Promise.all([fetch("model.json").then(r => r.json()), fetch("data.json").then(r => r.json())]).then(([m, d]) => {
  model = m; meta = d;
  MAIN.forEach(f => slider(f, $("main-sliders"))); MORE.forEach(f => slider(f, $("more-sliders")));
  const pre = document.querySelector(".presets");
  Object.entries(meta.examples).forEach(([name, vals]) => {
    const b = document.createElement("button"); b.textContent = name;
    b.addEventListener("click", () => { model.features.forEach(f => { $(f).value = vals[f]; }); render(); }); pre.appendChild(b);
  });
  if (meta.coverage) $("cov").textContent = `(covered ${Math.round(meta.coverage * 100)}% of tracts in cross-validation)`;
  render(); document.body.dataset.ready = "1";
});
