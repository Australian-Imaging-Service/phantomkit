"""
MRI Calibration Plotter & Temperature Estimator
================================================
Reads a GSP/PVP phantom calibration xlsx and provides two functions:

  1. build_html()  — generates a self-contained interactive HTML plotter
  2. estimate_temperature() — inverse lookup: given measured ADC/T1/T2,
     returns a temperature estimate with uncertainty for a named formulation

Usage (HTML plotter):
    python calibration_plotter.py plot <input.xlsx> [output.html]

Usage (temperature estimate):
    python calibration_plotter.py estimate <input.xlsx> <formulation> [--D val] [--T1 val] [--T2 val]

    Example:
        python calibration_plotter.py estimate calibration.xlsx "1mM Aqueous NiCl2 (0% PVP)" --D 2.017 --T1 1023.6
"""

import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParameterEstimate:
    """Temperature estimate from a single MRI parameter."""

    param: str  # "D", "T1", or "T2"
    measured: float  # measured value supplied by user
    temperature: float  # estimated temperature (°C)
    temp_unc: float  # ±1σ uncertainty on temperature (°C)
    fit_residual: float  # RMS residual of polynomial fit (same units as param)
    poly_degree: int  # degree of polynomial used
    t_range: tuple  # (T_min, T_max) of calibration data used


@dataclass
class TemperatureEstimate:
    """Combined temperature estimate from one or more MRI parameters."""

    formulation: str
    estimates: list[ParameterEstimate] = field(default_factory=list)
    combined_temperature: Optional[float] = None
    combined_unc: Optional[float] = None

    def summary(self) -> str:
        lines = [f"Formulation : {self.formulation}"]
        lines.append("-" * 55)
        for e in self.estimates:
            lines.append(
                f"  {e.param:<4}  measured = {e.measured:.4g}"
                f"  →  T = {e.temperature:.2f} ± {e.temp_unc:.2f} °C"
                f"  (fit residual: {e.fit_residual:.4g})"
            )
        if self.combined_temperature is not None:
            lines.append("-" * 55)
            lines.append(
                f"  Combined    T = {self.combined_temperature:.2f} ± {self.combined_unc:.2f} °C"
                f"  (inverse-variance weighted)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_calibration_xlsx(filepath: str) -> dict:
    """
    Parse the multi-formulation calibration spreadsheet.
    Returns dict: { formulation_name: [{"temperature": float, "D_val": float, ...}, ...] }
    """
    xl = pd.ExcelFile(filepath)
    sheet = xl.parse(xl.sheet_names[0], header=None)

    # Locate header row
    header_row = None
    for i, row in sheet.iterrows():
        if any("Formulation" in str(v) for v in row.values if pd.notna(v)):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find header row containing 'Formulation Name'")

    data_start = header_row + 2  # skip units row
    headers = sheet.iloc[header_row].tolist()

    col_map = {"formulation": None, "temperature": None}
    for idx, h in enumerate(headers):
        h_str = str(h).strip().lower()
        if "formulation" in h_str:
            col_map["formulation"] = idx
        elif "temperature" in h_str:
            col_map["temperature"] = idx

    temp_col = col_map["temperature"]
    numeric_cols = [
        i
        for i in range(len(headers))
        if i > temp_col and str(headers[i]).strip() not in ("", "nan")
    ]

    for pi, pair in enumerate(["D", "T1", "T2"]):
        col_map[f"{pair}_val"] = (
            numeric_cols[2 * pi] if 2 * pi < len(numeric_cols) else None
        )
        col_map[f"{pair}_unc"] = (
            numeric_cols[2 * pi + 1] if 2 * pi + 1 < len(numeric_cols) else None
        )

    formulations = {}
    current_name = None

    for _, row in sheet.iloc[data_start:].reset_index(drop=True).iterrows():
        vals = row.tolist()
        fname = (
            vals[col_map["formulation"]] if col_map["formulation"] is not None else None
        )
        if pd.notna(fname) and str(fname).strip():
            current_name = str(fname).strip()
        if current_name is None:
            continue

        try:
            temp = float(vals[col_map["temperature"]])
        except (ValueError, TypeError):
            continue

        record = {"temperature": temp}
        for param in ["D", "T1", "T2"]:
            for suffix in ["_val", "_unc"]:
                key = f"{param}{suffix}"
                try:
                    record[key] = (
                        float(vals[col_map[key]]) if col_map[key] is not None else None
                    )
                except (ValueError, TypeError):
                    record[key] = None

        formulations.setdefault(current_name, []).append(record)

    return formulations


def list_formulations(formulations: dict) -> None:
    print("Available formulations:")
    for i, (name, rows) in enumerate(formulations.items()):
        temps = [r["temperature"] for r in rows if np.isfinite(r["temperature"])]
        print(
            f"  [{i}] {name!r}  ({len(temps)} points, T = {min(temps):.0f}–{max(temps):.0f} °C)"
        )


def resolve_formulation(formulations: dict, query: str) -> str:
    """
    Match formulation by exact name, case-insensitive substring, or integer index.
    Raises ValueError with helpful message if not found or ambiguous.
    """
    names = list(formulations.keys())

    # Integer index
    try:
        idx = int(query)
        return names[idx]
    except (ValueError, IndexError):
        pass

    # Exact match
    if query in formulations:
        return query

    # Case-insensitive substring
    matches = [n for n in names if query.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous formulation query {query!r}. Matches:\n"
            + "\n".join(f"  {m}" for m in matches)
        )

    raise ValueError(
        f"No formulation matching {query!r}.\n"
        + "Available:\n"
        + "\n".join(f"  {n}" for n in names)
    )


# ---------------------------------------------------------------------------
# Temperature estimation (inverse lookup)
# ---------------------------------------------------------------------------


def _best_poly_degree(n_points: int) -> int:
    """Choose polynomial degree based on number of calibration points."""
    if n_points <= 3:
        return 1
    if n_points <= 6:
        return 2
    return 3


def _fit_and_invert(
    temps: np.ndarray,
    values: np.ndarray,
    uncertainties: np.ndarray,
    measured: float,
    param: str,
) -> ParameterEstimate:
    """
    Fit a polynomial to (temperature → parameter) calibration data,
    then numerically invert it to find T given a measured parameter value.
    Uncertainty is propagated via Monte Carlo (1000 samples).
    """
    # Remove any NaN rows (e.g. blank separator rows absorbed during parsing)
    mask = np.isfinite(temps) & np.isfinite(values) & np.isfinite(uncertainties)
    temps, values, uncertainties = temps[mask], values[mask], uncertainties[mask]

    n = len(temps)
    if n < 2:
        raise ValueError(
            f"{param}: insufficient valid calibration points after filtering ({n})"
        )

    degree = _best_poly_degree(n)

    # Fit polynomial: value = f(temperature)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = np.polyfit(temps, values, degree)
    poly = np.poly1d(coeffs)

    # RMS fit residual
    residual = float(np.sqrt(np.mean((poly(temps) - values) ** 2)))

    t_min, t_max = float(temps.min()), float(temps.max())

    # Check measured value is within the range spanned by the calibration curve
    cal_min, cal_max = sorted([float(poly(t_min)), float(poly(t_max))])
    margin = 0.1 * (cal_max - cal_min)
    if not (cal_min - margin <= measured <= cal_max + margin):
        raise ValueError(
            f"{param}: measured value {measured:.4g} is outside the calibration range "
            f"[{cal_min:.4g}, {cal_max:.4g}] for temperatures {t_min:.1f}–{t_max:.1f} °C"
        )

    def solve_T(val: float) -> float:
        """Find T such that poly(T) = val using Brent's method."""
        # Bracket: search for sign change
        f = lambda t: poly(t) - val
        # Try the full calibration range first
        if f(t_min) * f(t_max) < 0:
            return brentq(f, t_min, t_max)
        # Expand bracket slightly in case of near-boundary solution
        lo, hi = t_min - 2, t_max + 2
        if f(lo) * f(hi) < 0:
            return brentq(f, lo, hi)
        raise ValueError(
            f"{param}: could not bracket root for value {val:.4g} "
            f"in range [{lo:.1f}, {hi:.1f}] °C"
        )

    t_est = solve_T(measured)

    # --- Uncertainty via Monte Carlo ---
    # Perturb both calibration values (by their reported uncertainty) and
    # the polynomial coefficients (by resampling) to estimate T uncertainty.
    N_MC = 2000
    t_samples = []
    rng = np.random.default_rng(42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(N_MC):
            # Resample calibration values within their uncertainties
            v_perturbed = values + rng.normal(0, uncertainties)
            c_perturbed = np.polyfit(temps, v_perturbed, degree)
            p_perturbed = np.poly1d(c_perturbed)

            # Also perturb the measured value — assume its uncertainty equals
            # the fit residual (conservative; user can supply better if known)
            m_perturbed = measured + rng.normal(0, residual if residual > 0 else 1e-6)

            try:
                f2 = lambda t: p_perturbed(t) - m_perturbed
                lo2, hi2 = t_min - 5, t_max + 5
                if f2(lo2) * f2(hi2) < 0:
                    t_samples.append(brentq(f2, lo2, hi2))
            except Exception:
                pass

    if len(t_samples) < 10:
        raise ValueError(
            f"{param}: Monte Carlo uncertainty estimation failed (too few successful samples)"
        )

    t_unc = float(np.std(t_samples))

    return ParameterEstimate(
        param=param,
        measured=measured,
        temperature=t_est,
        temp_unc=t_unc,
        fit_residual=residual,
        poly_degree=degree,
        t_range=(t_min, t_max),
    )


def estimate_temperature(
    formulations: dict,
    formulation_query: str,
    D: Optional[float] = None,
    T1: Optional[float] = None,
    T2: Optional[float] = None,
) -> TemperatureEstimate:
    """
    Estimate temperature from one or more measured MRI parameter values.

    Parameters
    ----------
    formulations : dict
        Parsed calibration data from parse_calibration_xlsx().
    formulation_query : str
        Formulation name, case-insensitive substring, or integer index.
    D : float, optional
        Measured ADC value (10⁻³ mm²/s).
    T1 : float, optional
        Measured T1 (ms).
    T2 : float, optional
        Measured T2 (ms).

    Returns
    -------
    TemperatureEstimate
        Per-parameter estimates and an inverse-variance weighted combined estimate.
    """
    if D is None and T1 is None and T2 is None:
        raise ValueError("Supply at least one of D, T1, or T2.")

    name = resolve_formulation(formulations, formulation_query)
    rows = formulations[name]

    result = TemperatureEstimate(formulation=name)

    inputs = {"D": D, "T1": T1, "T2": T2}

    for param, measured in inputs.items():
        if measured is None:
            continue

        # Extract calibration points that have both value and uncertainty
        pts = [
            (r["temperature"], r[f"{param}_val"], r[f"{param}_unc"])
            for r in rows
            if r[f"{param}_val"] is not None and r[f"{param}_unc"] is not None
        ]

        if len(pts) < 2:
            print(f"  Warning: insufficient calibration data for {param}, skipping.")
            continue

        temps_arr = np.array([p[0] for p in pts])
        vals_arr = np.array([p[1] for p in pts])
        uncs_arr = np.array([p[2] for p in pts])

        try:
            est = _fit_and_invert(temps_arr, vals_arr, uncs_arr, measured, param)
            result.estimates.append(est)
        except ValueError as e:
            print(f"  Warning ({param}): {e}")

    # Combined estimate — inverse-variance weighted mean
    if len(result.estimates) >= 2:
        weights = [1.0 / (e.temp_unc**2) for e in result.estimates]
        w_sum = sum(weights)
        result.combined_temperature = (
            sum(w * e.temperature for w, e in zip(weights, result.estimates)) / w_sum
        )
        result.combined_unc = float(np.sqrt(1.0 / w_sum))
    elif len(result.estimates) == 1:
        result.combined_temperature = result.estimates[0].temperature
        result.combined_unc = result.estimates[0].temp_unc

    return result


# ---------------------------------------------------------------------------
# HTML plotter
# ---------------------------------------------------------------------------


def build_html(formulations: dict) -> str:
    data_json = json.dumps(formulations, indent=2)

    palette = [
        "#378ADD",
        "#D85A30",
        "#1D9E75",
        "#D4537E",
        "#7F77DD",
        "#BA7517",
        "#639922",
        "#888780",
        "#185FA5",
        "#993C1D",
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
    display: flex; align-items: center; gap: 6px; padding: 5px 10px;
    border-radius: 99px; border: 0.5px solid var(--border); cursor: pointer;
    font-size: 12px; background: var(--bg2); user-select: none; transition: opacity 0.15s;
  }}
  .chip.off {{ opacity: 0.35; }}
  .chip-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .charts {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
  @media (min-width: 900px) {{ .charts {{ grid-template-columns: 1fr 1fr; }} }}
  .chart-card {{
    background: var(--bg2); border: 0.5px solid var(--border);
    border-radius: 12px; padding: 16px;
  }}
  .chart-title {{ font-size: 13px; font-weight: 500; margin-bottom: 4px; }}
  .chart-unit {{ font-size: 11px; color: var(--text2); margin-bottom: 12px; }}
  .chart-wrap {{ position: relative; width: 100%; height: 280px; }}

  /* Inverse lookup panel */
  .lookup-section {{ margin-top: 32px; }}
  .section-title {{ font-size: 14px; font-weight: 500; margin-bottom: 4px; }}
  .section-sub {{ font-size: 12px; color: var(--text2); margin-bottom: 16px; }}
  .lookup-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 12px; }}
  .input-group {{ display: flex; flex-direction: column; gap: 4px; }}
  .input-group label {{ font-size: 12px; color: var(--text2); font-weight: 500; }}
  .input-group input {{
    padding: 7px 10px; border: 0.5px solid var(--border); border-radius: var(--radius);
    background: var(--bg2); color: var(--text); font-size: 13px;
    width: 100%; outline: none;
  }}
  .input-group input:focus {{ border-color: #378ADD; }}
  .input-hint {{ font-size: 11px; color: var(--text2); margin-top: 2px; }}
  .run-btn {{
    padding: 8px 20px; border: 0.5px solid var(--border); border-radius: var(--radius);
    background: var(--bg2); color: var(--text); font-size: 13px; cursor: pointer;
    transition: background 0.15s;
  }}
  .run-btn:hover {{ background: var(--bg3); }}
  .results-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 16px; }}
  .result-card {{
    background: var(--bg2); border: 0.5px solid var(--border);
    border-radius: var(--radius); padding: 12px 14px;
  }}
  .result-card.combined {{ border-color: #378ADD44; background: #378ADD0a; }}
  .result-label {{ font-size: 11px; color: var(--text2); margin-bottom: 6px; font-weight: 500; }}
  .result-temp {{ font-size: 22px; font-weight: 500; color: var(--text); }}
  .result-unc {{ font-size: 12px; color: var(--text2); margin-top: 2px; }}
  .result-detail {{ font-size: 11px; color: var(--text2); margin-top: 6px; border-top: 0.5px solid var(--border); padding-top: 6px; }}
  .warn-msg {{ font-size: 12px; color: #D85A30; margin-top: 8px; }}
  .empty-msg {{ font-size: 13px; color: var(--text2); margin-top: 12px; }}

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
  .color-swatch {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
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

<!-- Inverse temperature lookup -->
<div class="lookup-section">
  <div class="section-title">Temperature estimator</div>
  <p class="section-sub">Enter one or more measured values for a formulation to estimate temperature via inverse calibration curve lookup.</p>

  <div class="controls" style="margin-bottom: 12px;">
    <div class="ctrl">
      <label>Formulation</label>
      <select id="lookupForm"></select>
    </div>
  </div>

  <div class="lookup-grid">
    <div class="input-group">
      <label>ADC (10⁻³ mm²/s)</label>
      <input type="number" id="inD" placeholder="e.g. 2.017" step="0.001">
      <span class="input-hint">Leave blank to exclude</span>
    </div>
    <div class="input-group">
      <label>T1 (ms)</label>
      <input type="number" id="inT1" placeholder="e.g. 1023.6" step="0.1">
      <span class="input-hint">Leave blank to exclude</span>
    </div>
    <div class="input-group">
      <label>T2 (ms)</label>
      <input type="number" id="inT2" placeholder="e.g. 774.0" step="0.1">
      <span class="input-hint">Leave blank to exclude</span>
    </div>
  </div>

  <button class="run-btn" onclick="runLookup()">Estimate temperature</button>

  <div id="lookupResults"></div>
</div>

<!-- Summary table -->
<div class="stats-section">
  <div class="stats-title">Summary — values at 20 °C (or nearest available temperature)</div>
  <table class="stats-table">
    <thead>
      <tr>
        <th>Formulation</th><th>Temp (°C)</th>
        <th>ADC (10⁻³ mm²/s)</th><th>T1 (ms)</th><th>T2 (ms)</th>
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
const tickColor = "#888780";
const formNames = Object.keys(RAW);
let active = new Set(formNames);

// --- formulation selector for lookup ---
const lookupSel = document.getElementById("lookupForm");
formNames.forEach((name, i) => {{
  const opt = document.createElement("option");
  opt.value = name; opt.textContent = name;
  lookupSel.appendChild(opt);
}});

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

// --- polynomial helpers (mirrors Python logic) ---
function polyfit(xs, ys, degree) {{
  // Vandermonde matrix, least-squares via normal equations
  const n = xs.length;
  const A = [];
  for (let i = 0; i < n; i++) {{
    const row = [];
    for (let d = degree; d >= 0; d--) row.push(Math.pow(xs[i], d));
    A.push(row);
  }}
  // Simple least squares via Gaussian elimination (works fine for degree ≤ 3)
  return lstsq(A, ys);
}}

function lstsq(A, b) {{
  // AtA x = Atb
  const m = A.length, n = A[0].length;
  const AtA = Array.from({{length: n}}, () => new Array(n).fill(0));
  const Atb = new Array(n).fill(0);
  for (let i = 0; i < m; i++) {{
    for (let j = 0; j < n; j++) {{
      Atb[j] += A[i][j] * b[i];
      for (let k = 0; k < n; k++) AtA[j][k] += A[i][j] * A[i][k];
    }}
  }}
  return gaussElim(AtA, Atb);
}}

function gaussElim(A, b) {{
  const n = A.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let col = 0; col < n; col++) {{
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    for (let row = col + 1; row < n; row++) {{
      const f = M[row][col] / M[col][col];
      for (let k = col; k <= n; k++) M[row][k] -= f * M[col][k];
    }}
  }}
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {{
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }}
  return x;
}}

function polyval(coeffs, x) {{
  return coeffs.reduce((acc, c) => acc * x + c, 0);
}}

function brentq(f, a, b, tol=1e-8, maxIter=100) {{
  let fa = f(a), fb = f(b);
  if (fa * fb > 0) return null;
  let c = a, fc = fa, d = 0, e = 0;
  for (let i = 0; i < maxIter; i++) {{
    if (fb * fc > 0) {{ c = a; fc = fa; d = e = b - a; }}
    if (Math.abs(fc) < Math.abs(fb)) {{
      [a, b, c] = [b, c, b]; [fa, fb, fc] = [fb, fc, fb];
    }}
    const tol1 = 2e-16 * Math.abs(b) + tol / 2;
    const xm = (c - b) / 2;
    if (Math.abs(xm) <= tol1 || fb === 0) return b;
    if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {{
      let p, q, s = fb / fa;
      if (a === c) {{ p = 2 * xm * s; q = 1 - s; }}
      else {{
        q = fa / fc; const r = fb / fc;
        p = s * (2 * xm * q * (q - r) - (b - a) * (r - 1));
        q = (q - 1) * (r - 1) * (s - 1);
      }}
      if (p > 0) q = -q; else p = -p;
      if (2 * p < Math.min(3 * xm * q - Math.abs(tol1 * q), Math.abs(e * q))) {{
        e = d; d = p / q;
      }} else {{ d = xm; e = d; }}
    }} else {{ d = xm; e = d; }}
    a = b; fa = fb;
    b += Math.abs(d) > tol1 ? d : (xm > 0 ? tol1 : -tol1);
    fb = f(b);
  }}
  return b;
}}

function estimateTemp(name, param, measured) {{
  const pts = RAW[name]
    .filter(r => r[`${{param}}_val`] !== null && r[`${{param}}_unc`] !== null)
    .sort((a, b) => a.temperature - b.temperature);
  if (pts.length < 2) return null;

  const temps = pts.map(p => p.temperature);
  const vals  = pts.map(p => p[`${{param}}_val`]);
  const uncs  = pts.map(p => p[`${{param}}_unc`]);
  const degree = pts.length <= 3 ? 1 : pts.length <= 6 ? 2 : 3;
  const coeffs = polyfit(temps, vals, degree);

  const calMin = Math.min(...temps.map(t => polyval(coeffs, t)));
  const calMax = Math.max(...temps.map(t => polyval(coeffs, t)));
  const margin = 0.1 * (calMax - calMin);
  if (measured < calMin - margin || measured > calMax + margin) {{
    return {{ error: `Value ${{measured.toFixed(4)}} outside calibration range [${{calMin.toFixed(3)}}, ${{calMax.toFixed(3)}}]` }};
  }}

  const tMin = temps[0], tMax = temps[temps.length - 1];
  const T = brentq(t => polyval(coeffs, t) - measured, tMin - 2, tMax + 2);
  if (T === null) return {{ error: "Could not solve inverse" }};

  // Residual
  const residual = Math.sqrt(vals.reduce((s, v, i) => s + (polyval(coeffs, temps[i]) - v) ** 2, 0) / vals.length);

  // Monte Carlo uncertainty (500 samples for speed in browser)
  const N = 500;
  const tSamples = [];
  for (let s = 0; s < N; s++) {{
    const vPerturbed = vals.map((v, i) => v + (Math.random() * 2 - 1) * uncs[i] * 1.7449); // ~normal via Box-Muller approx
    const mPerturbed = measured + (Math.random() * 2 - 1) * residual * 1.7449;
    const cP = polyfit(temps, vPerturbed, degree);
    const t2 = brentq(t => polyval(cP, t) - mPerturbed, tMin - 5, tMax + 5);
    if (t2 !== null) tSamples.push(t2);
  }}
  const mean = tSamples.reduce((a, b) => a + b, 0) / tSamples.length;
  const unc = Math.sqrt(tSamples.reduce((s, t) => s + (t - mean) ** 2, 0) / tSamples.length);

  return {{ T, unc, residual, degree, tMin, tMax }};
}}

function runLookup() {{
  const name = lookupSel.value;
  const dVal  = parseFloat(document.getElementById("inD").value);
  const t1Val = parseFloat(document.getElementById("inT1").value);
  const t2Val = parseFloat(document.getElementById("inT2").value);

  const inputs = [
    {{ param: "D",  label: "ADC",  unit: "10⁻³ mm²/s", val: isNaN(dVal)  ? null : dVal  }},
    {{ param: "T1", label: "T1",   unit: "ms",          val: isNaN(t1Val) ? null : t1Val }},
    {{ param: "T2", label: "T2",   unit: "ms",          val: isNaN(t2Val) ? null : t2Val }},
  ].filter(i => i.val !== null);

  const el = document.getElementById("lookupResults");

  if (!inputs.length) {{
    el.innerHTML = `<p class="empty-msg">Enter at least one measured value above.</p>`;
    return;
  }}

  const results = inputs.map(inp => ({{ ...inp, res: estimateTemp(name, inp.param, inp.val) }}));
  const valid = results.filter(r => r.res && !r.res.error);

  let combined = null;
  if (valid.length >= 2) {{
    const weights = valid.map(r => 1 / (r.res.unc ** 2));
    const wSum = weights.reduce((a, b) => a + b, 0);
    const Tcomb = valid.reduce((s, r, i) => s + weights[i] * r.res.T, 0) / wSum;
    const uncComb = Math.sqrt(1 / wSum);
    combined = {{ T: Tcomb, unc: uncComb }};
  }}

  let html = `<div class="results-grid">`;
  results.forEach(r => {{
    if (r.res && r.res.error) {{
      html += `<div class="result-card">
        <div class="result-label">${{r.label}}</div>
        <div class="warn-msg">${{r.res.error}}</div>
      </div>`;
    }} else if (r.res) {{
      html += `<div class="result-card">
        <div class="result-label">From ${{r.label}} = ${{r.val}} ${{r.unit}}</div>
        <div class="result-temp">${{r.res.T.toFixed(2)}} °C</div>
        <div class="result-unc">± ${{r.res.unc.toFixed(2)}} °C</div>
        <div class="result-detail">degree-${{r.res.degree}} poly · fit residual ${{r.res.residual.toFixed(4)}}<br>
          cal. range ${{r.res.tMin.toFixed(0)}}–${{r.res.tMax.toFixed(0)}} °C</div>
      </div>`;
    }}
  }});
  if (combined) {{
    html += `<div class="result-card combined">
      <div class="result-label">Combined (inv-variance weighted)</div>
      <div class="result-temp">${{combined.T.toFixed(2)}} °C</div>
      <div class="result-unc">± ${{combined.unc.toFixed(2)}} °C</div>
    </div>`;
  }}
  html += `</div>`;
  el.innerHTML = html;
}}

// --- chart helpers ---
function makeDatasets(param, errMult) {{
  return formNames.filter(n => active.has(n)).map(name => {{
    const pts = RAW[name].filter(r => r[`${{param}}_val`] !== null).sort((a, b) => a.temperature - b.temperature);
    const ds = {{
      label: name, data: pts.map(r => ({{ x: r.temperature, y: r[`${{param}}_val`] }})),
      borderColor: COLOURS[name], backgroundColor: COLOURS[name] + "22",
      pointBackgroundColor: COLOURS[name], pointRadius: 4, pointHoverRadius: 6,
      borderWidth: 1.5, tension: 0.3, fill: false,
    }};
    if (errMult > 0) {{
      ds.errorBars = {{}};
      pts.forEach((r, i) => {{
        const u = r[`${{param}}_unc`];
        if (u !== null) ds.errorBars[i] = {{ yMin: r[`${{param}}_val`] - errMult * u, yMax: r[`${{param}}_val`] + errMult * u }};
      }});
    }}
    return ds;
  }});
}}

const baseOpts = () => ({{
  responsive: true, maintainAspectRatio: false,
  plugins: {{
    legend: {{ display: false }},
    tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(3)}}` }} }}
  }},
  scales: {{
    x: {{ type: "linear", title: {{ display: true, text: "Temperature (°C)", color: tickColor, font: {{ size: 11 }} }},
          ticks: {{ color: tickColor, font: {{ size: 11 }} }}, grid: {{ color: gridColor }} }},
    y: {{ ticks: {{ color: tickColor, font: {{ size: 11 }} }}, grid: {{ color: gridColor }} }}
  }}
}});

const errorBarPlugin = {{
  id: "errorBar",
  afterDatasetsDraw(chart) {{
    const ctx = chart.ctx;
    chart.data.datasets.forEach((ds, di) => {{
      if (!ds.errorBars) return;
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      ctx.save(); ctx.strokeStyle = ds.borderColor; ctx.lineWidth = 1.2;
      Object.entries(ds.errorBars).forEach(([i, eb]) => {{
        const el = meta.data[+i]; if (!el) return;
        const x = el.x, cap = 4;
        const yMin = chart.scales.y.getPixelForValue(eb.yMin);
        const yMax = chart.scales.y.getPixelForValue(eb.yMax);
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

function makeChart(id, param) {{
  return new Chart(document.getElementById(id).getContext("2d"), {{
    type: "line", data: {{ datasets: [] }}, options: baseOpts(), plugins: [errorBarPlugin],
  }});
}}

const charts = {{ D: makeChart("chartD"), T1: makeChart("chartT1"), T2: makeChart("chartT2") }};

function updateAll() {{
  const errMult = parseFloat(document.getElementById("errMode").value);
  ["D", "T1", "T2"].forEach(p => {{ charts[p].data.datasets = makeDatasets(p, errMult); charts[p].update(); }});
  updateStats();
}}

function updateStats() {{
  const tbody = document.getElementById("statsBody"); tbody.innerHTML = "";
  formNames.filter(n => active.has(n)).forEach(name => {{
    const pts = RAW[name].filter(r => r.D_val !== null);
    if (!pts.length) return;
    const c = pts.reduce((b, r) => Math.abs(r.temperature - 20) < Math.abs(b.temperature - 20) ? r : b);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="color-swatch" style="background:${{COLOURS[name]}}"></span>${{name}}</td>
      <td>${{c.temperature.toFixed(1)}}</td>
      <td>${{c.D_val?.toFixed(4) ?? "—"}} <span class="temp-range">± ${{c.D_unc?.toFixed(4) ?? "?"}}</span></td>
      <td>${{c.T1_val?.toFixed(1) ?? "—"}} <span class="temp-range">± ${{c.T1_unc?.toFixed(1) ?? "?"}}</span></td>
      <td>${{c.T2_val?.toFixed(1) ?? "—"}} <span class="temp-range">± ${{c.T2_unc?.toFixed(1) ?? "?"}}</span></td>`;
    tbody.appendChild(tr);
  }});
}}

updateAll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MRI Calibration Plotter & Temperature Estimator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # plot subcommand
    p_plot = sub.add_parser("plot", help="Generate interactive HTML plotter")
    p_plot.add_argument("xlsx", help="Calibration spreadsheet (.xlsx)")
    p_plot.add_argument(
        "output", nargs="?", help="Output HTML file (default: <xlsx stem>_plotter.html)"
    )

    # estimate subcommand
    p_est = sub.add_parser(
        "estimate", help="Estimate temperature from measured MRI values"
    )
    p_est.add_argument("xlsx", help="Calibration spreadsheet (.xlsx)")
    p_est.add_argument(
        "formulation",
        help="Formulation name, substring, or index (use 'list' to see options)",
    )
    p_est.add_argument(
        "--D", type=float, metavar="VAL", help="Measured ADC (10⁻³ mm²/s)"
    )
    p_est.add_argument("--T1", type=float, metavar="VAL", help="Measured T1 (ms)")
    p_est.add_argument("--T2", type=float, metavar="VAL", help="Measured T2 (ms)")

    args = parser.parse_args()

    print(f"Parsing: {args.xlsx}")
    formulations = parse_calibration_xlsx(args.xlsx)
    print(f"Found {len(formulations)} formulations.\n")

    if args.command == "plot":
        out = args.output or Path(args.xlsx).stem + "_plotter.html"
        html = build_html(formulations)
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML plotter written to: {out}")

    elif args.command == "estimate":
        if args.formulation.lower() == "list":
            list_formulations(formulations)
            return

        result = estimate_temperature(
            formulations,
            formulation_query=args.formulation,
            D=args.D,
            T1=args.T1,
            T2=args.T2,
        )
        print(result.summary())


if __name__ == "__main__":
    main()
