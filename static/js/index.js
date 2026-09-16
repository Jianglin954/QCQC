/* Builds the qualitative-results grid from the same data the live demo uses
   (demo/data.js), so the page and the demo can never drift apart. */
(function () {
  const grid = document.getElementById("results-grid");
  if (!grid || typeof DB === "undefined") return;

  const img = path => "static/images/qualitative/" + path.split("/").pop();

  const head = document.createElement("div");
  head.className = "col-head";
  head.innerHTML = "<div></div>" + LEVELS.map(lv =>
    `<div class="h-${lv}">Rel: ${LEVEL_LABEL[lv]} · Aes: ${LEVEL_LABEL[lv]}</div>`).join("");
  grid.appendChild(head);

  DB.forEach(entry => {
    const row = document.createElement("div");
    row.className = "result-row";
    row.innerHTML = `<div class="row-prefix">${entry.prefix}</div>` + LEVELS.map(lv => {
      const p = entry.profiles[lv];
      return `<div class="res-card ${lv}">
          <img src="${img(p.image)}" alt="image retrieved for “${entry.prefix} ${p.completion}”" loading="lazy">
          <div class="res-body">
            <div class="res-tag">Rel: ${LEVEL_LABEL[lv]} · Aes: ${LEVEL_LABEL[lv]}</div>
            <div class="res-q"><b>${entry.prefix}</b> <span class="cmp">${p.completion}</span></div>
            <div class="res-scores">
              <span>Aes <b>${p.aes.toFixed(3)}</b></span>
              <span>Rel <b>${p.rel.toFixed(3)}</b></span>
            </div>
          </div>
        </div>`;
    }).join("");
    grid.appendChild(row);
  });
})();
