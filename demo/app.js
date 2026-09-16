/* Quality-controlled visual search — front-end simulation.
   The LLM and the CLIP index are mocked: every completion, image and score below is the
   cached value reported in the paper's qualitative table (see data.js). */

const $ = id => document.getElementById(id);
const SPEED = parseFloat(new URLSearchParams(location.search).get("speed") || "1");
const sleep = ms => new Promise(r => setTimeout(r, ms / SPEED));

const state = { aes: null, rel: null, prefix: "", run: 0 };

/* ---------------- lookup ---------------- */

const normalize = s => s.trim().toLowerCase().replace(/\s+/g, " ");
const findEntry = q => DB.find(e => e.prefix === normalize(q));

/* ---------------- query box ---------------- */

function updateGhost() {
  const v = $("query").value;
  const hit = v ? DB.find(e => e.prefix.startsWith(normalize(v))) : null;
  $("ghost").textContent = hit && hit.prefix !== normalize(v) ? " ".repeat(v.length) + hit.prefix.slice(v.length) : "";
}
$("query").addEventListener("input", updateGhost);
$("query").addEventListener("keydown", e => { if (e.key === "Enter") search(); });
$("go").onclick = () => search();

/* ---------------- quality buttons ---------------- */

document.querySelectorAll(".qbtn").forEach(btn => {
  btn.onclick = () => {
    const { dim, level } = btn.dataset;
    state[dim] = level;
    document.querySelectorAll(`.qbtn[data-dim="${dim}"]`).forEach(b => b.classList.toggle("on", b === btn));
    btn.classList.remove("pulse"); void btn.offsetWidth; btn.classList.add("pulse");
    syncTokens();
    if ($("query").value.trim()) search();
  };
});

function setLevel(dim, level) {
  document.querySelector(`.qbtn[data-dim="${dim}"][data-level="${level}"]`).click();
}

function syncTokens() {
  for (const [dim, id] of [["aes", "tok-aes"], ["rel", "tok-rel"]]) {
    const el = $(id), lv = state[dim];
    el.className = "tok" + (lv ? " " + lv : "");
    el.textContent = `<${dim}=${lv ? lv.toUpperCase() : "?"}>`;
  }
}

/* ---------------- compare mode ---------------- */

$("compare").onclick = () => {
  const on = $("compare").classList.toggle("on");
  if (!on) { $("compare-grid").classList.remove("show"); return; }
  const entry = findEntry($("query").value);
  if (!entry) { $("compare").classList.remove("on"); return; }
  $("card").classList.remove("show");
  $("result-empty").style.display = "none";
  $("notice").classList.remove("show");
  $("compare-grid").innerHTML = LEVELS.map(lv => {
    const p = entry.profiles[lv];
    return `<div class="cmp ${lv}">
      <img src="${p.image}" alt="">
      <div class="cmp-body">
        <div class="cmp-tag">Rel: ${LEVEL_LABEL[lv]} · Aes: ${LEVEL_LABEL[lv]}</div>
        <div class="cmp-q"><b>${entry.prefix}</b> <span style="color:var(--${lv === "medium" ? "med" : lv})">${p.completion}</span></div>
        <div class="cmp-s">Aes ${p.aes.toFixed(3)} · Rel ${p.rel.toFixed(3)}</div>
      </div></div>`;
  }).join("");
  $("compare-grid").classList.add("show");
};

/* ---------------- the search pipeline ---------------- */

const DEBOUNCE = 420;   // ms: lets you set Aes and Rel before anything is dispatched
const GRACE = 900;      // ms: extra patience before calling a pair "not cached" — you are
                        // probably still on your way to the second button

async function search() {
  const run = ++state.run;                       // cancels any in-flight animation
  const alive = () => run === state.run;
  await sleep(DEBOUNCE);
  if (!alive()) return;

  const entry = findEntry($("query").value);
  $("compare").classList.remove("on");
  $("compare-grid").classList.remove("show");

  if (!state.aes || !state.rel) return showNotice("Choose an <b>Aesthetic</b> and a <b>Relevance</b> level to run the quality-controlled completion.");
  if (!entry) return showNotice(`No cached demo data for “${$("query").value.trim()}”.<br>Try one of: ${DB.map(e => "<b>" + e.prefix + "</b>").join(", ")}.`);
  if (state.aes !== state.rel) {
    await sleep(GRACE);
    if (!alive()) return;
    const lv = state.aes;
    return showNotice(
      `The paper’s table reports <b>matched</b> control levels only (Rel and Aes set to the same level).<br>` +
      `<i>Rel: ${LEVEL_LABEL[state.rel]} · Aes: ${LEVEL_LABEL[state.aes]}</i> is not in the cached index.`,
      `Align both to ${LEVEL_LABEL[lv]}`, () => setLevel("rel", lv));
  }

  const level = state.aes, p = entry.profiles[level], base = entry.profiles.low;
  hideNotice();
  $("result-empty").style.display = "none";
  $("card").classList.remove("show");
  $("ghost").textContent = "";

  /* step 1 — LLM completes the query under the control tokens */
  $("step-llm").classList.add("active");
  $("step-ret").classList.remove("active");
  $("prompt-prefix").textContent = entry.prefix;
  $("ret-status").textContent = "waiting for completion…";
  $("ret-bar").style.width = "0%";
  $("latency").textContent = "";
  $("c-prefix").textContent = entry.prefix + " ";
  $("c-rest").textContent = "";
  $("c-rest").className = "c-rest " + level;
  $("llm-dots").classList.add("on");
  renderTable(entry, level);
  $("caret").classList.add("on");
  await sleep(520);
  if (!alive()) return;
  $("llm-dots").classList.remove("on");

  for (const ch of p.completion) {                // token-by-token typing
    if (!alive()) return;
    $("c-rest").textContent += ch;
    await sleep(ch === " " ? 34 : 19);
  }
  $("caret").classList.remove("on");
  await sleep(240);
  if (!alive()) return;

  /* step 2 — retrieval over the index */
  $("step-ret").classList.add("active");
  $("ret-status").textContent = `scoring 123,287 images against the completed query…`;
  for (const w of [18, 46, 74, 100]) {
    if (!alive()) return;
    $("ret-bar").style.width = w + "%";
    await sleep(150);
  }
  const ms = (38 + Math.random() * 22).toFixed(1);
  $("ret-status").textContent = "top-1 retrieved";
  $("latency").textContent = ms + " ms";
  await sleep(160);
  if (!alive()) return;

  /* step 3 — the result card */
  const img = $("res-img");
  img.classList.remove("in");
  img.src = p.image;
  $("profile-tag").className = "profile-tag " + level;
  $("profile-tag").textContent = `Rel: ${LEVEL_LABEL[level]} · Aes: ${LEVEL_LABEL[level]}`;
  $("card-query").innerHTML = `<b>${entry.prefix}</b> <span class="c-rest ${level}">${p.completion}</span>`;
  $("card").classList.add("show");
  requestAnimationFrame(() => img.classList.add("in"));

  $("aes-bar").className = "sbar-fill " + level;
  $("rel-bar").className = "sbar-fill " + level;
  $("aes-bar").style.width = pct(p.aes, 1, 10) + "%";
  $("rel-bar").style.width = pct(p.rel, 0.30, 0.45) + "%";
  countUp($("aes-val"), p.aes, 3, alive);
  countUp($("rel-val"), p.rel, 3, alive);
  $("aes-delta").innerHTML = delta(p.aes - base.aes, 3);
  $("rel-delta").innerHTML = delta(p.rel - base.rel, 3);
}

function renderTable(entry, current) {
  const rows = [`<tr class="hdr"><td></td><td class="num-c">Aes</td><td class="num-c">Rel</td><td class="num-c">Δ Aes</td></tr>`];
  const base = entry.profiles.low;
  for (const lv of LEVELS) {
    const p = entry.profiles[lv], d = p.aes - base.aes;
    rows.push(`<tr class="${lv}${lv === current ? " cur" : ""}">
      <td class="lv">${LEVEL_LABEL[lv]}</td>
      <td class="num-c">${p.aes.toFixed(3)}</td>
      <td class="num-c">${p.rel.toFixed(3)}</td>
      <td class="num-c">${d ? (d > 0 ? "+" : "") + d.toFixed(3) : "—"}</td></tr>`);
  }
  $("ptable").querySelector("tbody").innerHTML = rows.join("");
  $("step-tbl").classList.add("active");
}

const pct = (v, lo, hi) => Math.max(2, Math.min(100, (v - lo) / (hi - lo) * 100));

function delta(d, n) {
  if (Math.abs(d) < 1e-9) return `<span style="color:#9aa1ad">baseline (Low)</span>`;
  const s = (d > 0 ? "+" : "") + d.toFixed(n);
  return `<span class="${d > 0 ? "up" : "down"}">${s} vs Low</span>`;
}

async function countUp(el, target, digits, alive) {
  const steps = 22;
  for (let i = 1; i <= steps; i++) {
    if (alive && !alive()) return;
    el.textContent = (target * (1 - Math.pow(1 - i / steps, 3))).toFixed(digits);
    await sleep(16);
  }
  el.textContent = target.toFixed(digits);
}

function showNotice(html, btnLabel, onClick) {
  $("card").classList.remove("show");
  $("result-empty").style.display = "none";
  const n = $("notice");
  n.innerHTML = html;
  if (btnLabel) {
    const b = document.createElement("button");
    b.textContent = btnLabel;
    b.onclick = onClick;
    n.appendChild(document.createElement("br"));
    n.appendChild(b);
  }
  n.classList.add("show");
}
function hideNotice() { $("notice").classList.remove("show"); }

syncTokens();
$("corpus-name").textContent = CORPUS;
