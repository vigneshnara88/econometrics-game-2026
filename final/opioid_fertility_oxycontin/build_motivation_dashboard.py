from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("opioid_fertility_oxycontin")
DATA = ROOT / "data" / "processed" / "master_state_year_2007_2013.csv"
OUT = ROOT / "paper_results_bundle" / "motivation_dashboard.html"


ABBR = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL",
    13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA",
    23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
    32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK",
    41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT",
    51: "VA", 53: "WA", 54: "WV", 55: "WI", 56: "WY",
}


def build_state_payload(df: pd.DataFrame) -> tuple[list[dict], dict]:
    df = df.copy()
    df["abbr"] = df["state_code"].map(ABBR)
    df = df[df["abbr"].notna()].copy()

    rows = []
    for (state, code, abbr), sub in df.groupby(["state", "state_code", "abbr"]):
        sub = sub.sort_values("year")
        y2009 = sub[sub["year"] == 2009].iloc[0]
        y2013 = sub[sub["year"] == 2013].iloc[0]
        rows.append(
            {
                "state": state,
                "state_code": int(code),
                "abbr": abbr,
                "initial_oxy": float(sub["initial_oxy"].iloc[0]),
                "fertility_2009": float(y2009["fertility_rate_proxy_per_1000_women_15_44"]),
                "fertility_2013": float(y2013["fertility_rate_proxy_per_1000_women_15_44"]),
                "fertility_change": float(y2013["fertility_rate_proxy_per_1000_women_15_44"] - y2009["fertility_rate_proxy_per_1000_women_15_44"]),
                "oxycodone_change_pct": float(100 * (y2013["arcos_totaloxycod"] / y2009["arcos_totaloxycod"] - 1)),
                "births_avg": float(sub["births"].mean()),
                "years": sub["year"].tolist(),
                "fertility_series": sub["fertility_rate_proxy_per_1000_women_15_44"].round(3).tolist(),
                "oxycodone_index_series": (100 * sub["arcos_totaloxycod"] / y2009["arcos_totaloxycod"]).round(1).tolist(),
            }
        )

    states = sorted(rows, key=lambda x: x["abbr"])
    metrics = {
        "initial_oxy": {
            "label": "Baseline OxyContin exposure",
            "format": "pct",
            "values": [s["initial_oxy"] for s in states],
        },
        "fertility_change": {
            "label": "Fertility change, 2009 to 2013",
            "format": "delta",
            "values": [s["fertility_change"] for s in states],
        },
    }
    return states, metrics


def build_summary(df: pd.DataFrame, states: list[dict]) -> dict:
    sub = df[df["year"] != 2010].copy()
    exposure = pd.Series([s["initial_oxy"] for s in states])
    weighted_2009 = (
        sub[sub["year"] == 2009]["fertility_rate_proxy_per_1000_women_15_44"] * sub[sub["year"] == 2009]["female_pop_15_44_proxy"]
    ).sum() / sub[sub["year"] == 2009]["female_pop_15_44_proxy"].sum()
    weighted_2013 = (
        sub[sub["year"] == 2013]["fertility_rate_proxy_per_1000_women_15_44"] * sub[sub["year"] == 2013]["female_pop_15_44_proxy"]
    ).sum() / sub[sub["year"] == 2013]["female_pop_15_44_proxy"].sum()
    return {
        "n_states": int(len(states)),
        "exposure_min": float(exposure.min()),
        "exposure_max": float(exposure.max()),
        "exposure_sd": float(exposure.std()),
        "avg_births": float(sub["births"].mean()),
        "weighted_fertility_change": float(weighted_2013 - weighted_2009),
    }


def html(states: list[dict], metrics: dict, summary: dict) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Opioid Fertility Motivation Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script src="https://cdn.jsdelivr.net/npm/topojson-client@3"></script>
  <style>
    :root {{
      --bg: #f7f4ee;
      --ink: #1f2933;
      --muted: #5b6670;
      --panel: #fffdf8;
      --line: #d8d0c4;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    .wrap {{ max-width: 1380px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 10px; font-size: 36px; line-height: 1.1; }}
    .sub {{ color: var(--muted); max-width: 920px; font-size: 17px; line-height: 1.5; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin: 24px 0; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); padding: 16px; border-radius: 14px; }}
    .card .k {{ font-size: 28px; font-weight: 700; }}
    .card .l {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 6px; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; }}
    .panel h2 {{ margin: 0 0 8px; font-size: 20px; }}
    .controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 14px; color: var(--muted); font-size: 14px; }}
    select {{ padding: 8px 10px; border-radius: 10px; border: 1px solid var(--line); background: white; }}
    .legend {{ margin-top: 14px; }}
    .legend-bar {{ height: 12px; border-radius: 999px; background: linear-gradient(90deg, #1e3a5f, #4d86a8, #f2d3a1, #c45b39, #7d1616); }}
    .legend-labels {{ display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-top: 6px; }}
    .detail-head {{ display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }}
    .detail-head .state {{ font-size: 26px; font-weight: 700; }}
    .mini {{ color: var(--muted); font-size: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; margin: 18px 0; }}
    .stat {{ padding: 10px 12px; background: #faf7f1; border: 1px solid var(--line); border-radius: 12px; }}
    .stat .v {{ font-size: 22px; font-weight: 700; }}
    .stat .t {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .foot {{ color: var(--muted); font-size: 13px; margin-top: 18px; line-height: 1.5; }}
    .rank-list {{ margin: 0; padding-left: 18px; line-height: 1.7; }}
    .state-path {{ stroke: #f8f4ec; stroke-width: 1.1; cursor: pointer; }}
    .state-path.active {{ stroke: #111827; stroke-width: 2.4; }}
    .state-label {{ font-size: 10px; fill: rgba(255,255,255,0.88); pointer-events: none; text-anchor: middle; }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Why This Question Is Plausible</h1>
    <div class="sub">
      This dashboard uses the opioid-fertility state panel to show three ingredients behind the research question:
      large cross-state variation in baseline OxyContin exposure, a meaningful post-2010 shift in local opioid supply,
      and a fertility outcome that moved differently across exposed states.
    </div>

    <div class="cards">
      <div class="card"><div class="k">{summary["n_states"]}</div><div class="l">States + DC in sample</div></div>
      <div class="card"><div class="k">{summary["exposure_min"]:.2f} to {summary["exposure_max"]:.2f}</div><div class="l">Range of baseline Oxy exposure</div></div>
      <div class="card"><div class="k">{summary["exposure_sd"]:.2f}</div><div class="l">Cross-state SD of exposure</div></div>
      <div class="card"><div class="k">{summary["weighted_fertility_change"]:.2f}</div><div class="l">Weighted fertility change, 2009 to 2013</div></div>
    </div>

    <div class="grid">
      <div class="panel">
        <h2>State Map</h2>
        <div class="controls">
          <label for="metric">Color by</label>
          <select id="metric">
            <option value="initial_oxy">Baseline OxyContin exposure</option>
            <option value="fertility_change">Fertility change, 2009 to 2013</option>
          </select>
        </div>
        <svg viewBox="0 0 975 610" id="usMap" aria-label="US choropleth map"></svg>
        <div class="legend">
          <div class="legend-bar"></div>
          <div class="legend-labels"><span id="legendMin"></span><span id="legendMid"></span><span id="legendMax"></span></div>
        </div>
      </div>

      <div class="panel">
        <div class="detail-head">
          <div>
            <div class="state" id="stateName">Select a state</div>
            <div class="mini" id="stateSub">Click a state to view opioid and fertility trends</div>
          </div>
        </div>
        <div class="stats" id="stateStats"></div>
        <svg viewBox="0 0 520 260" id="detailChart" aria-label="State trend chart"></svg>
        <div class="foot">
          Blue line: fertility rate per 1,000 women ages 15-44. Red line: oxycodone supply indexed to 2009 = 100.
          The state view is descriptive, but it helps connect the opioid disruption to the fertility question.
        </div>
      </div>
    </div>

    <div class="panel" style="margin-top: 18px;">
      <h2>Quick Motivating Facts</h2>
      <ol class="rank-list" id="factsList"></ol>
    </div>
  </div>

  <script>
    const states = {json.dumps(states)};
    const metrics = {json.dumps(metrics)};
    const stateLookup = Object.fromEntries(states.map(s => [String(s.state_code).padStart(2, "0"), s]));
    let activeState = states.find(s => s.abbr === "WV") || states[0];

    function interpColor(t) {{
      const stops = [
        [30, 58, 95], [77, 134, 168], [242, 211, 161], [196, 91, 57], [125, 22, 22]
      ];
      const scaled = Math.max(0, Math.min(1, t)) * (stops.length - 1);
      const i = Math.floor(scaled);
      const f = scaled - i;
      const a = stops[i];
      const b = stops[Math.min(i + 1, stops.length - 1)];
      const mix = a.map((v, idx) => Math.round(v + (b[idx] - v) * f));
      return `rgb(${{mix[0]}}, ${{mix[1]}}, ${{mix[2]}})`;
    }}

    function metricLabel(metric, value) {{
      if (metric === "initial_oxy") return value.toFixed(2);
      if (metric === "fertility_change") return (value >= 0 ? "+" : "") + value.toFixed(2);
      return (value >= 0 ? "+" : "") + value.toFixed(1) + "%";
    }}

    async function renderMap() {{
      const metric = document.getElementById("metric").value;
      const values = metrics[metric].values;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const mid = (min + max) / 2;
      document.getElementById("legendMin").textContent = metricLabel(metric, min);
      document.getElementById("legendMid").textContent = metricLabel(metric, mid);
      document.getElementById("legendMax").textContent = metricLabel(metric, max);

      const svg = d3.select("#usMap");
      svg.selectAll("*").remove();
      const width = 975, height = 610;

      const us = await d3.json("https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json");
      const statesGeo = topojson.feature(us, us.objects.states);
      const mesh = topojson.mesh(us, us.objects.states, (a, b) => a !== b);

      const projection = d3.geoAlbersUsa().fitSize([width, height], statesGeo);
      const path = d3.geoPath(projection);

      svg.append("g")
        .selectAll("path")
        .data(statesGeo.features)
        .join("path")
        .attr("class", d => {{
          const code = String(d.id).padStart(2, "0");
          const info = stateLookup[code];
          return "state-path" + (info && info.abbr === activeState.abbr ? " active" : "");
        }})
        .attr("fill", d => {{
          const code = String(d.id).padStart(2, "0");
          const info = stateLookup[code];
          if (!info) return "#ddd7ca";
          const t = max === min ? 0.5 : (info[metric] - min) / (max - min);
          return interpColor(t);
        }})
        .attr("d", path)
        .append("title")
        .text(d => {{
          const code = String(d.id).padStart(2, "0");
          const info = stateLookup[code];
          return info ? `${{info.state}} | ${{metrics[metric].label}}: ${{metricLabel(metric, info[metric])}}` : "No data";
        }});

      svg.append("path")
        .datum(mesh)
        .attr("fill", "none")
        .attr("stroke", "#f8f4ec")
        .attr("stroke-width", 0.9)
        .attr("d", path);

      svg.selectAll(".state-click")
        .data(statesGeo.features)
        .join("path")
        .attr("d", path)
        .attr("fill", "transparent")
        .style("cursor", "pointer")
        .on("click", (_, d) => {{
          const code = String(d.id).padStart(2, "0");
          const info = stateLookup[code];
          if (!info) return;
          activeState = info;
          renderMap();
          renderDetail();
        }});
    }}

    function renderDetail() {{
      const s = activeState;
      document.getElementById("stateName").textContent = s.state;
      document.getElementById("stateSub").textContent = `Baseline exposure ${{s.initial_oxy.toFixed(2)}} | Avg births ${{Math.round(s.births_avg).toLocaleString()}}`;
      const stats = [
        [`${{s.initial_oxy.toFixed(2)}}`, "Baseline Oxy exposure"],
        [`${{s.fertility_2009.toFixed(1)}}`, "Fertility rate in 2009"],
        [`${{(s.fertility_change >= 0 ? '+' : '') + s.fertility_change.toFixed(2)}}`, "Fertility change, 2009 to 2013"],
        [`${{Math.round(s.births_avg).toLocaleString()}}`, "Average annual births"],
      ];
      document.getElementById("stateStats").innerHTML = stats.map(([v,t]) => `<div class="stat"><div class="v">${{v}}</div><div class="t">${{t}}</div></div>`).join("");

      const svg = document.getElementById("detailChart");
      const W = 520, H = 260, ml = 42, mr = 20, mt = 16, mb = 34;
      const innerW = W - ml - mr, innerH = H - mt - mb;
      const xVals = s.years;
      const fert = s.fertility_series;
      const oxy = s.oxycodone_index_series;
      const yMin = Math.min(...fert, ...oxy) * 0.95;
      const yMax = Math.max(...fert, ...oxy) * 1.05;
      const x = i => ml + i * (innerW / (xVals.length - 1));
      const y = v => mt + innerH - ((v - yMin) / (yMax - yMin)) * innerH;
      const points = arr => arr.map((v, i) => `${{x(i)}},${{y(v)}}`).join(" ");

      let axes = "";
      for (let i = 0; i < xVals.length; i++) {{
        axes += `<text x="${{x(i)}}" y="${{H - 10}}" text-anchor="middle" font-size="11" fill="#5b6670">${{xVals[i]}}</text>`;
      }}
      for (let j = 0; j <= 4; j++) {{
        const val = yMin + (yMax - yMin) * j / 4;
        const yy = y(val);
        axes += `<line x1="${{ml}}" y1="${{yy}}" x2="${{W-mr}}" y2="${{yy}}" stroke="#e9e1d4" />`;
        axes += `<text x="8" y="${{yy + 4}}" font-size="11" fill="#5b6670">${{val.toFixed(0)}}</text>`;
      }}
      svg.innerHTML = `
        <rect x="0" y="0" width="${{W}}" height="${{H}}" fill="transparent"></rect>
        ${{axes}}
        <polyline fill="none" stroke="#145b7d" stroke-width="3" points="${{points(fert)}}" />
        <polyline fill="none" stroke="#9f2f24" stroke-width="3" points="${{points(oxy)}}" />
      `;
    }}

    function renderFacts() {{
      const byExposure = [...states].sort((a, b) => b.initial_oxy - a.initial_oxy);
      const byFertilityChange = [...states].sort((a, b) => b.fertility_change - a.fertility_change);
      const facts = [
        `Baseline OxyContin exposure ranges from ${{byExposure.at(-1).initial_oxy.toFixed(2)}} to ${{byExposure[0].initial_oxy.toFixed(2)}}, giving the design substantial cross-state treatment intensity variation.`,
        `The highest-exposure states in this panel are ${{byExposure.slice(0, 5).map(s => s.abbr).join(", ")}}.`,
        `The strongest fertility improvements between 2009 and 2013 appear in ${{byFertilityChange.slice(0, 5).map(s => `${{s.abbr}} (${{s.fertility_change >= 0 ? '+' : ''}}${{s.fertility_change.toFixed(2)}})`).join(", ")}}.`,
      ];
      document.getElementById("factsList").innerHTML = facts.map(f => `<li>${{f}}</li>`).join("");
    }}

    document.getElementById("metric").addEventListener("change", () => renderMap());
    renderMap();
    renderDetail();
    renderFacts();
  </script>
</body>
</html>"""


def main() -> None:
    df = pd.read_csv(DATA)
    states, metrics = build_state_payload(df)
    summary = build_summary(df, states)
    OUT.write_text(html(states, metrics, summary), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
