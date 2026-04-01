"""
MRI Calibration Data Plotter
Reads the GSP/PVP calibration xlsx and generates a self-contained interactive HTML file.
Usage: python build_plotter.py <input.xlsx> [output.html]
"""

import sys
import json
import pandas as pd
from pathlib import Path


def parse_calibration_xlsx(filepath: str) -> dict:
    """
    Parse the multi-formulation calibration spreadsheet.
    Expected structure (0-indexed rows):
      Row 0: column headers (Formulation Name, Temperature, D reported, D uncertainty, ...)
      Row 1: units
      Rows 2+: data (formulation name appears only on first row of each group)
    Returns dict keyed by formulation name.
    """
    xl = pd.ExcelFile(filepath)
    sheet = xl.parse(xl.sheet_names[0], header=None)

    # Locate header row — find row containing "Formulation Name"
    header_row = None
    for i, row in sheet.iterrows():
        if any("Formulation" in str(v) for v in row.values if pd.notna(v)):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find header row containing 'Formulation Name'")

    units_row = header_row + 1
    data_start = header_row + 2  # skip units row

    # Build clean column names from headers + units
    headers = sheet.iloc[header_row].tolist()
    units = sheet.iloc[units_row].tolist()

    col_map = {
        "formulation": None,
        "temperature": None,
        "D_val": None,
        "D_unc": None,
        "T1_val": None,
        "T1_unc": None,
        "T2_val": None,
        "T2_unc": None,
    }

    for idx, h in enumerate(headers):
        h_str = str(h).strip().lower()
        if "formulation" in h_str:
            col_map["formulation"] = idx
        elif "temperature" in h_str:
            col_map["temperature"] = idx
        else:
            # Use sequence: D reported, D uncertainty, T1 reported, T1 uncertainty, T2 reported, T2 uncertainty
            pass

    # Find numeric data columns by position after temperature
    temp_col = col_map["temperature"]
    numeric_cols = [i for i in range(len(headers))
                    if i > temp_col and str(headers[i]).strip() not in ("", "nan")]

    # Map pairs: (reported, uncertainty) for D, T1, T2
    pairs = ["D", "T1", "T2"]
    for pi, pair in enumerate(pairs):
        if 2 * pi < len(numeric_cols):
            col_map[f"{pair}_val"] = numeric_cols[2 * pi]
        if 2 * pi + 1 < len(numeric_cols):
            col_map[f"{pair}_unc"] = numeric_cols[2 * pi + 1]

    # Parse data rows
    data_rows = sheet.iloc[data_start:].reset_index(drop=True)

    formulations = {}
    current_name = None

    for _, row in data_rows.iterrows():
        vals = row.tolist()

        # Formulation name — may be blank (continuation of previous)
        fname = vals[col_map["formulation"]] if col_map["formulation"] is not None else None
        if pd.notna(fname) and str(fname).strip():
            current_name = str(fname).strip()

        if current_name is None:
            continue

        # Temperature
        try:
            temp = float(vals[col_map["temperature"]])
        except (ValueError, TypeError):
            continue

        record = {"temperature": temp}
        for param in ["D", "T1", "T2"]:
            try:
                record[f"{param}_val"] = float(vals[col_map[f"{param}_val"]])
            except (ValueError, TypeError, KeyError):
                record[f"{param}_val"] = None
            try:
                record[f"{param}_unc"] = float(vals[col_map[f"{param}_unc"]])
            except (ValueError, TypeError, KeyError):
                record[f"{param}_unc"] = None

        if current_name not in formulations:
            formulations[current_name] = []
        formulations[current_name].append(record)

    return formulations


def build_html(formulations: dict) -> str:
    data_json = json.dumps(formulations, indent=2)

    # Assign a distinct colour per formulation (palette that works light+dark)
    palette = [
        "#378ADD", "#D85A30", "#1D9E75", "#D4537E",
        "#7F77DD", "#BA7517", "#639922", "#888780",
        "#185FA5", "#993C1D",
    ]
    form_names = list(formulations.keys())
    colour_map = {name: palette[i % len(palette)] for i, name in enumerate(form_names)}
    colour_json = json.dumps(colour_map)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MRI Calibration Plotter</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #ffffff; --bg2: #f4f4f2; --bg3: #e9e9e6;
    --text: #1a1a18; --text2: #5f5e5a; --border: rgba(0,0,0,0.12);
    --radius: 8px; --font: system-ui, -apple-system, sans-serif;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1c1c1a; --bg2: #252523; --bg3: #2e2e2b;
      --text: #e8e8e4; --text2: #888780; --border: rgba(255,255,255,0.10);
    }}
  }}
  body {{ font-family: var(--font); background: var(--bg); color: var(--text); padding: 24px; }}
  h1 {{ font-size: 18px; font-weight: 500; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: var(--text2); margin-bottom: 20px; }}

  /* Controls */
  .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-bottom: 20px; }}
  .ctrl {{ display: flex; flex-direction: column; gap: 4px; }}
  label {{ font-size: 12px; color: var(--text2); font-weight: 500; }}
  select {{
    padding: 6px 10px; border: 0.5px solid var(--border); border-radius: var(--radius);
    background: var(--bg2); color: var(--text); font-size: 13px; cursor: pointer;
    appearance: none; padding-right: 28px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 8px center;
  }}
  .toggle-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .chip {{
    display: flex; align-items: center; gap: 6px;
    padding: 5px 10px; border-radius: 99px; border: 0.5px solid var(--border);
    cursor: pointer; font-size: 12px; background: var(--bg2);
    user-select: none; transition: opacity 0.15s;
  }}
  .chip.off {{ opacity: 0.35; }}
  .chip-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}

  /* Charts */
  .charts {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
  @media (min-width: 900px) {{ .charts {{ grid-template-columns: 1fr 1fr; }} }}
  .chart-card {{
    background: var(--bg2); border: 0.5px solid var(--border);
    border-radius: 12px; padding: 16px;
  }}
  .chart-title {{ font-size: 13px; font-weight: 500; margin-bottom: 4px; }}
  .chart-unit {{ font-size: 11px; color: var(--text2); margin-bottom: 12px; }}
  .chart-wrap {{ position: relative; width: 100%; height: 280px; }}

  /* Stats table */
  .stats-section {{ margin-top: 28px; }}
  .stats-title {{ font-size: 13px; font-weight: 500; color: var(--text2); margin-bottom: 10px; }}
  .stats-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .stats-table th {{
    text-align: left; padding: 6px 10px; background: var(--bg3);
    border-bottom: 0.5px solid var(--border); color: var(--text2); font-weight: 500;
  }}
  .stats-table td {{ padding: 5px 10px; border-bottom: 0.5px solid var(--border); }}
  .stats-table tr:last-child td {{ border-bottom: none; }}
  .color-swatch {{
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
  }}
  .temp-range {{ font-size: 11px; color: var(--text2); }}
</style>
</head>
<body>

<h1>MRI Calibration — GSP / PVP Phantom</h1>
<p class="subtitle">ADC, T1 and T2 vs temperature · error bars show reported uncertainty</p>

<div class="controls">
  <div class="ctrl">
    <label>Error bars</label>
    <select id="errMode">
      <option value="1">± 1× uncertainty</option>
      <option value="2">± 2× uncertainty</option>
      <option value="0">Hide</option>
    </select>
  </div>
  <div class="ctrl">
    <label>Show formulations</label>
    <div class="toggle-row" id="chips"></div>
  </div>
</div>

<div class="charts">
  <div class="chart-card">
    <div class="chart-title">Apparent Diffusion Coefficient (ADC)</div>
    <div class="chart-unit">10⁻³ mm²/s</div>
    <div class="chart-wrap"><canvas id="chartD"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">T1 Relaxation Time</div>
    <div class="chart-unit">ms</div>
    <div class="chart-wrap"><canvas id="chartT1"></canvas></div>
  </div>
  <div class="chart-card" style="grid-column: 1 / -1;">
    <div class="chart-title">T2 Relaxation Time</div>
    <div class="chart-unit">ms</div>
    <div class="chart-wrap" style="height: 260px;"><canvas id="chartT2"></canvas></div>
  </div>
</div>

<div class="stats-section">
  <div class="stats-title">Summary — values at 20 °C (or nearest available temperature)</div>
  <table class="stats-table" id="statsTable">
    <thead>
      <tr>
        <th>Formulation</th>
        <th>Temp (°C)</th>
        <th>ADC (10⁻³ mm²/s)</th>
        <th>T1 (ms)</th>
        <th>T2 (ms)</th>
      </tr>
    </thead>
    <tbody id="statsBody"></tbody>
  </table>
</div>

<script>
const RAW = {data_json};
const COLOURS = {colour_json};

const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
const gridColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)";
const tickColor = isDark ? "#888780" : "#888780";

const formNames = Object.keys(RAW);
let active = new Set(formNames);

// --- chip toggles ---
const chipsEl = document.getElementById("chips");
formNames.forEach(name => {{
  const chip = document.createElement("div");
  chip.className = "chip";
  chip.innerHTML = `<span class="chip-dot" style="background:${{COLOURS[name]}}"></span>${{name}}`;
  chip.addEventListener("click", () => {{
    if (active.has(name)) {{ active.delete(name); chip.classList.add("off"); }}
    else {{ active.add(name); chip.classList.remove("off"); }}
    updateAll();
  }});
  chipsEl.appendChild(chip);
}});

document.getElementById("errMode").addEventListener("change", updateAll);

// --- chart helpers ---
function makeDatasets(param, errMult) {{
  return formNames.filter(n => active.has(n)).map(name => {{
    const pts = RAW[name]
      .filter(r => r[`${{param}}_val`] !== null)
      .sort((a, b) => a.temperature - b.temperature);
    const dataset = {{
      label: name,
      data: pts.map(r => ({{ x: r.temperature, y: r[`${{param}}_val`] }})),
      borderColor: COLOURS[name],
      backgroundColor: COLOURS[name] + "22",
      pointBackgroundColor: COLOURS[name],
      pointRadius: 4,
      pointHoverRadius: 6,
      borderWidth: 1.5,
      tension: 0.3,
      fill: false,
    }};
    if (errMult > 0) {{
      dataset.errorBars = {{}};
      pts.forEach((r, i) => {{
        const unc = r[`${{param}}_unc`];
        if (unc !== null) {{
          dataset.errorBars[i] = {{ yMin: r[`${{param}}_val`] - errMult * unc, yMax: r[`${{param}}_val`] + errMult * unc }};
        }}
      }});
    }}
    return dataset;
  }});
}}

const baseOpts = (title) => ({{
  responsive: true,
  maintainAspectRatio: false,
  plugins: {{
    legend: {{ display: false }},
    tooltip: {{
      callbacks: {{
        label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(3)}}`,
      }}
    }}
  }},
  scales: {{
    x: {{
      type: "linear",
      title: {{ display: true, text: "Temperature (°C)", color: tickColor, font: {{ size: 11 }} }},
      ticks: {{ color: tickColor, font: {{ size: 11 }} }},
      grid: {{ color: gridColor }},
    }},
    y: {{
      title: {{ display: false }},
      ticks: {{ color: tickColor, font: {{ size: 11 }} }},
      grid: {{ color: gridColor }},
    }}
  }}
}});

// Error bar plugin (custom, lightweight)
const errorBarPlugin = {{
  id: "errorBar",
  afterDatasetsDraw(chart) {{
    const ctx = chart.ctx;
    chart.data.datasets.forEach((ds, di) => {{
      if (!ds.errorBars) return;
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      ctx.save();
      ctx.strokeStyle = ds.borderColor;
      ctx.lineWidth = 1.2;
      Object.entries(ds.errorBars).forEach(([i, eb]) => {{
        const el = meta.data[+i];
        if (!el) return;
        const x = el.x;
        const yMin = chart.scales.y.getPixelForValue(eb.yMin);
        const yMax = chart.scales.y.getPixelForValue(eb.yMax);
        const cap = 4;
        ctx.beginPath();
        ctx.moveTo(x, yMin); ctx.lineTo(x, yMax);
        ctx.moveTo(x - cap, yMin); ctx.lineTo(x + cap, yMin);
        ctx.moveTo(x - cap, yMax); ctx.lineTo(x + cap, yMax);
        ctx.stroke();
      }});
      ctx.restore();
    }});
  }}
}};

function makeChart(canvasId, param) {{
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {{
    type: "line",
    data: {{ datasets: [] }},
    options: baseOpts(param),
    plugins: [errorBarPlugin],
  }});
}}

const charts = {{
  D: makeChart("chartD", "D"),
  T1: makeChart("chartT1", "T1"),
  T2: makeChart("chartT2", "T2"),
}};

function updateAll() {{
  const errMult = parseFloat(document.getElementById("errMode").value);
  ["D", "T1", "T2"].forEach(param => {{
    charts[param].data.datasets = makeDatasets(param, errMult);
    charts[param].update();
  }});
  updateStats();
}}

// --- stats table ---
function updateStats() {{
  const tbody = document.getElementById("statsBody");
  tbody.innerHTML = "";
  const TARGET_TEMP = 20;
  formNames.filter(n => active.has(n)).forEach(name => {{
    const pts = RAW[name].filter(r => r.D_val !== null);
    if (!pts.length) return;
    const closest = pts.reduce((best, r) =>
      Math.abs(r.temperature - TARGET_TEMP) < Math.abs(best.temperature - TARGET_TEMP) ? r : best
    );
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="color-swatch" style="background:${{COLOURS[name]}}"></span>${{name}}</td>
      <td>${{closest.temperature.toFixed(1)}}</td>
      <td>${{closest.D_val !== null ? closest.D_val.toFixed(4) : "—"}} <span class="temp-range">± ${{closest.D_unc !== null ? closest.D_unc.toFixed(4) : "?"}}</span></td>
      <td>${{closest.T1_val !== null ? closest.T1_val.toFixed(1) : "—"}} <span class="temp-range">± ${{closest.T1_unc !== null ? closest.T1_unc.toFixed(1) : "?"}}</span></td>
      <td>${{closest.T2_val !== null ? closest.T2_val.toFixed(1) : "—"}} <span class="temp-range">± ${{closest.T2_unc !== null ? closest.T2_unc.toFixed(1) : "?"}}</span></td>
    `;
    tbody.appendChild(tr);
  }});
}}

updateAll();
</script>
</body>
</html>
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_plotter.py <input.xlsx> [output.html]")
        sys.exit(1)

    xlsx_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else Path(xlsx_path).stem + "_plotter.html"

    print(f"Parsing: {xlsx_path}")
    formulations = parse_calibration_xlsx(xlsx_path)
    print(f"Found {len(formulations)} formulations:")
    for name, rows in formulations.items():
        temps = [r['temperature'] for r in rows]
        print(f"  {name!r}: {len(rows)} rows, T = {min(temps)}–{max(temps)} °C")

    html = build_html(formulations)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()
