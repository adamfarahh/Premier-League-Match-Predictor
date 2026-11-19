"use strict";

const byId = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("en-US");
const dates = new Intl.DateTimeFormat("en-US", {
  year: "numeric", month: "short", day: "numeric", timeZone: "UTC",
});
const formatDate = (date) => dates.format(new Date(date + "T00:00:00Z"));
const percent = (value) => (value * 100).toFixed(1) + "%";
const outcomes = ["Home win", "Draw", "Away win"];
const tones = ["home", "draw", "away"];
const calls = { H: "Home win", D: "Draw", A: "Away win" };
const pageSize = 12;

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

async function readJson(path, params) {
  const query = params ? "?" + new URLSearchParams(params) : "";
  const response = await fetch(path + query);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed.");
  return data;
}

function status(panel, message, error = false) {
  const output = byId(panel + "-status");
  output.textContent = message;
  output.classList.toggle("error", error);
}

function control(form, name) {
  return form.elements.namedItem(name);
}

function enable(form) {
  for (const input of form.elements) input.disabled = false;
}

function addOptions(select, values) {
  for (const value of values) select.add(new Option(value, value));
}

function formPanel(club) {
  const card = element("article", undefined, "card form-card");
  card.append(element("h3", club.name));
  card.append(element("p", `${number.format(club.rating)} Elo · ${number.format(club.matches_played)} prior fixtures`));
  if (!club.recent.length) {
    card.append(element("p", "No form in the previous 12 months."));
    return card;
  }
  const list = element("ul");
  list.setAttribute("aria-label", "Recent form, oldest to newest");
  const resultClasses = { W: "win", D: "tied", L: "loss" };
  const resultLabels = { W: "Win", D: "Draw", L: "Loss" };
  for (const match of club.recent) {
    const item = element("li", match.result, resultClasses[match.result]);
    item.title = `${resultLabels[match.result]} ${match.goals_for}–${match.goals_against} vs ${match.opponent}, ${formatDate(match.date)}`;
    item.setAttribute("aria-label", item.title);
    list.append(item);
  }
  card.append(list);
  return card;
}

function renderForecast(forecast) {
  byId("forecast-date").textContent = formatDate(forecast.date);
  byId("matchup").textContent = `${forecast.home.name} vs ${forecast.away.name}`;
  const favourite = { H: forecast.home.name, D: "Draw", A: forecast.away.name };
  byId("favourite").textContent = "Most likely outcome · " + favourite[forecast.prediction];

  const bar = byId("probability-bar");
  const labels = byId("probabilities");
  bar.replaceChildren();
  labels.replaceChildren();
  for (const [index, value] of forecast.probabilities.entries()) {
    const segment = element("span", undefined, tones[index]);
    segment.style.flexGrow = value;
    bar.append(segment);
    const label = element("div", undefined, tones[index]);
    label.append(element("dt", outcomes[index]), element("dd", percent(value)));
    labels.append(label);
  }
  byId("training-cutoff").textContent = `Trained on ${number.format(forecast.training_matches)} fixtures through ${formatDate(forecast.trained_through)}`;
  const actual = byId("actual-score");
  actual.hidden = !forecast.actual;
  if (forecast.actual) actual.textContent = `Actual score · ${forecast.actual.home_goals}–${forecast.actual.away_goals}`;
  const clubs = byId("club-form");
  clubs.replaceChildren(formPanel(forecast.home), formPanel(forecast.away));
  clubs.hidden = false;
  byId("forecast-output").hidden = false;
}

const forecastForm = byId("forecast-form");
async function runForecast() {
  const button = forecastForm.querySelector("button");
  button.disabled = true;
  const outlook = document.querySelector(".outlook");
  outlook.setAttribute("aria-busy", "true");
  status("forecast", "Calculating probabilities…");
  byId("forecast-output").hidden = true;
  byId("club-form").hidden = true;
  try {
    renderForecast(await readJson("/api/forecast", new FormData(forecastForm)));
    status("forecast", "");
  } catch (error) {
    status("forecast", error.message, true);
  } finally {
    outlook.setAttribute("aria-busy", "false");
    button.disabled = false;
  }
}

function pager(id, rows, draw) {
  const pagination = byId(id + "-pagination");
  let page = 1;
  const pages = Math.ceil(rows.length / pageSize);
  const previous = element("button", "Previous");
  const next = element("button", "Next");
  const info = element("span");
  info.setAttribute("aria-live", "polite");
  pagination.replaceChildren(previous, info, next);
  pagination.hidden = pages <= 1;

  function renderPage() {
    const start = (page - 1) * pageSize;
    draw(rows.slice(start, start + pageSize));
    info.textContent = `Page ${page} of ${pages}`;
    previous.disabled = page === 1;
    next.disabled = page === pages;
  }
  previous.addEventListener("click", () => {
    page -= 1;
    renderPage();
    if (previous.disabled) next.focus();
  });
  next.addEventListener("click", () => {
    page += 1;
    renderPage();
    if (next.disabled) previous.focus();
  });
  renderPage();
}

function score(match) {
  return `${match.home_goals}–${match.away_goals}`;
}

function matchDetails(match) {
  const card = element("div", undefined, "stat-detail");
  const stats = match.stats;
  const meta = [stats.kickoff, stats.referee].filter(Boolean);
  if (meta.length) card.append(element("p", meta.join(" · ")));
  const rows = [
    ["Half-time score", stats.home_ht_goals, stats.away_ht_goals],
    ["Expected goals", match.home_xg, match.away_xg],
    ["Shots", stats.home_shots, stats.away_shots],
    ["Shots on target", stats.home_shots_on_target, stats.away_shots_on_target],
    ["Corners", stats.home_corners, stats.away_corners],
    ["Fouls", stats.home_fouls, stats.away_fouls],
    ["Yellow cards", stats.home_yellows, stats.away_yellows],
    ["Red cards", stats.home_reds, stats.away_reds],
  ].filter(([, home, away]) => home !== null || away !== null);
  if (rows.length) {
    const list = element("dl");
    for (const [label, home, away] of rows) {
      const row = element("div");
      row.append(element("dt", label), element("dd", `${home ?? "—"}  /  ${away ?? "—"}`));
      list.append(row);
    }
    card.append(list);
  } else if (!meta.length) {
    card.append(element("p", "No additional match statistics are available."));
  }
  return card;
}

function renderResultsPage(matches) {
  const rows = byId("results-rows");
  rows.replaceChildren();
  for (const [index, match] of matches.entries()) {
    const row = element("tr");
    const action = element("td");
    const button = element("button", "Stats", "detail-button");
    button.setAttribute("aria-label", `Stats for ${match.home} vs ${match.away}`);
    button.setAttribute("aria-expanded", "false");
    action.append(button);
    row.append(
      element("td", formatDate(match.date), "entry-date"), element("td", match.home),
      element("td", score(match), "number"), element("td", match.away), action,
    );
    const detail = element("tr", undefined, "table-detail");
    detail.id = "fixture-details-" + index;
    detail.hidden = true;
    const cell = element("td");
    cell.colSpan = 5;
    cell.append(matchDetails(match));
    detail.append(cell);
    button.setAttribute("aria-controls", detail.id);
    button.addEventListener("click", () => {
      detail.hidden = !detail.hidden;
      button.setAttribute("aria-expanded", String(!detail.hidden));
    });
    rows.append(row, detail);
  }
}

const resultsForm = byId("results-form");
let resultsRequest = 0;
async function runResults() {
  const request = ++resultsRequest;
  status("results", "Loading results…");
  byId("results-output").hidden = true;
  try {
    const data = await readJson("/api/results", new FormData(resultsForm));
    if (request !== resultsRequest) return;
    if (!data.fixtures.length) {
      status("results", "No completed fixtures match those filters.");
      return;
    }
    status("results", "");
    byId("results-count").textContent = `${number.format(data.fixtures.length)} fixtures`;
    pager("results", data.fixtures, renderResultsPage);
    byId("results-output").hidden = false;
  } catch (error) {
    if (request === resultsRequest) status("results", error.message, true);
  }
}

function metric(name, model, baseline, instruction) {
  const card = element("dl", undefined, "card metric");
  card.append(
    element("dt", name), element("dd", model, "value"),
    element("dd", "Baseline " + baseline, "baseline"), element("dd", instruction, "baseline"),
  );
  return card;
}

function renderMatrix(confusion) {
  const labels = ["Home", "Draw", "Away"];
  const table = byId("matrix");
  const caption = element("caption", "Actual outcome by predicted outcome");
  const headings = element("tr");
  headings.append(element("th", "Actual"));
  for (const label of labels) {
    const heading = element("th", label);
    heading.scope = "col";
    headings.append(heading);
  }
  const head = element("thead");
  head.append(headings);
  const rows = element("tbody");
  for (const [actual, counts] of confusion.entries()) {
    const row = element("tr");
    const heading = element("th", labels[actual]);
    heading.scope = "row";
    row.append(heading);
    for (const [predicted, count] of counts.entries()) {
      row.append(element("td", number.format(count), actual === predicted ? "correct-count" : undefined));
    }
    rows.append(row);
  }
  table.replaceChildren(caption, head, rows);
}

function renderBacktestPage(matches) {
  const rows = byId("backtest-rows");
  rows.replaceChildren();
  for (const match of matches) {
    const row = element("tr");
    row.append(
      element("td", formatDate(match.date), "entry-date"),
      element("td", `${match.home} vs ${match.away}`),
      element("td", score(match), "number"),
      element("td", calls[match.prediction]),
      element("td", match.probabilities.map(percent).join(" / "), "number"),
    );
    rows.append(row);
  }
}

function renderBacktest(report, params) {
  const { model, baseline } = report;
  byId("metrics").replaceChildren(
    metric("Accuracy", percent(model.accuracy), percent(baseline.accuracy), "Higher is better"),
    metric("Log loss", model.log_loss.toFixed(3), baseline.log_loss.toFixed(3), "Lower is better"),
    metric("Brier score", model.brier.toFixed(3), baseline.brier.toFixed(3), "Lower is better"),
  );
  byId("matrix-detail").textContent = `${number.format(report.matches)} fixtures against a league outcome frequency baseline.`;
  renderMatrix(report.confusion);
  pager("backtest", report.predictions, renderBacktestPage);
  byId("download").href = "/api/backtest.csv?" + new URLSearchParams(params);
  byId("backtest-output").hidden = false;
}

const backtestForm = byId("backtest-form");
async function runBacktest() {
  const button = backtestForm.querySelector("button");
  button.disabled = true;
  status("backtest", "Evaluating fixtures…");
  byId("backtest-output").hidden = true;
  const params = new FormData(backtestForm);
  try {
    renderBacktest(await readJson("/api/backtest", params), params);
    status("backtest", "");
  } catch (error) {
    status("backtest", error.message, true);
  } finally {
    button.disabled = false;
  }
}

const views = { forecast: runForecast, results: runResults, backtest: runBacktest };
const visited = new Set();
function showView() {
  const hash = location.hash.slice(1);
  const active = Object.hasOwn(views, hash) ? hash : "forecast";
  for (const key of Object.keys(views)) {
    byId(key).hidden = key !== active;
    const link = document.querySelector(`[data-view="${key}"]`);
    if (key === active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (!visited.has(active)) {
    visited.add(active);
    views[active]();
  }
}

async function load() {
  try {
    const history = await readJson("/api/history");
    byId("archive-count").textContent = number.format(history.matches);
    byId("archive-date").textContent = formatDate(history.last_date);
    byId("results-coverage").textContent = `${formatDate(history.first_date)} through ${formatDate(history.last_date)}`;
    for (const name of ["home", "away"]) {
      const input = control(forecastForm, name);
      addOptions(input, history.clubs);
      input.value = history.example[name];
    }
    control(forecastForm, "date").value = history.example.date;
    if (history.first_forecast) control(forecastForm, "date").min = history.first_forecast;
    addOptions(control(resultsForm, "year"), history.years);
    addOptions(control(resultsForm, "club"), history.clubs);
    control(backtestForm, "since").value = history.backtest_start;
    control(backtestForm, "until").value = history.last_date;
    for (const name of ["since", "until"]) {
      control(backtestForm, name).max = history.last_date;
      if (history.first_forecast) control(backtestForm, name).min = history.first_forecast;
    }
    for (const form of [forecastForm, resultsForm, backtestForm]) enable(form);
    addEventListener("hashchange", showView);
    showView();
  } catch (error) {
    const message = byId("startup-error");
    message.textContent = "Could not load match history. " + error.message;
    message.hidden = false;
    status("forecast", "");
    document.querySelector(".outlook").setAttribute("aria-busy", "false");
  }
}

forecastForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runForecast();
});
resultsForm.addEventListener("submit", (event) => event.preventDefault());
resultsForm.addEventListener("change", runResults);
backtestForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runBacktest();
});
load();
