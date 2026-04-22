# PhantomKit — Interactive Plot Overview

## Purpose

PhantomKit QA generates a set of **self-contained interactive HTML files** for each scan session. Each file can be opened in any web browser with no server, no dependencies, and no internet connection — all data (images and charts) are embedded directly inside the HTML.

---

## Output files

| File | Contents |
|---|---|
| `{contrast_name}.html` | Per-contrast vial scatter plot (one per DWI/T1/FA/ADC contrast) |
| `{contrast_name}.png` | Per-contrast scatter plot for individual Inversion Recovery (IR) and Multi-Echo Spin Echo (MESE) timepoints |
| `T1_mapping.html` | Inversion Recovery (IR) T1 curve fits for all vials |
| `T2_mapping.html` | Multi-Echo Spin Echo (MESE) T2 decay fits for all vials |

All files live in `{session}/metrics/plots/`.

---

## Series naming conventions

PhantomKit uses **word-boundary token matching** to classify contrast series automatically. Filenames are matched case-insensitively; the token must not be directly adjacent to alphanumeric characters on either side.

| Series type | Radiology term | Filename token | Example filenames |
|---|---|---|---|
| Inversion Recovery | Inversion Recovery (IR), Magnetisation-Prepared Rapid Gradient Echo (MPRAGE) | `ir` | `se_ir_100.nii.gz`, `IR_500.nii.gz` |
| Multi-Echo Spin Echo | Multi-Echo Spin Echo (MESE), T2-weighted SE | `te` | `t2_se_TE_14.nii.gz`, `TE_80.nii.gz` |
| T1-weighted / MPRAGE | T1 MPRAGE, T1-Flash | `t1` or `mprage` | `T1.nii.gz`, `MPRAGE.nii.gz` |
| ADC map | Apparent Diffusion Coefficient | `adc` | `ADC.nii.gz` |
| FA map | Fractional Anisotropy | `fa` | `FA.nii.gz` |

The **numeric suffix** in IR and MESE filenames is interpreted as the physical parameter value:
- IR: the inversion time **TI** (ms), e.g. `se_ir_500` → TI = 500 ms
- MESE: the echo time **TE** (ms), e.g. `t2_se_TE_80` → TE = 80 ms

These values are used as the x-axis in the T1/T2 mapping plots and for fitting the relaxation models.

---

## What each plot contains

### Per-contrast scatter plots (`plot_vial_intensity`)

These cover ADC, FA, and any other single-volume contrast (e.g. T1_in_DWI_space).

- **Chart.js scatter chart** showing per-vial mean intensity ± std dev.
- **ADC mode** additionally overlays reference ADC values (from `adc_reference.json`) as red open circles; measured values are blue filled circles.
- **FA mode** fixes the y-axis to 0–1.
- For single-graph plots (ADC, FA, T1) the chart fills the full page width and every vial gets an x-axis label.
- **Embedded MRI viewer** (NiiVue) showing the contrast image with vial ROI overlays (see below). For ADC, only vials E–L (the calibration vials) are shown as toggleable overlays.

### Relaxometry maps (`plot_vial_ir_means_std`, `plot_vial_te_means_std`)

- **8-panel Chart.js grid** — one subplot per vial group — showing mean intensity vs TI (T1) or TE (T2), with fitted mono-exponential / inversion-recovery curves and 95% confidence intervals.
- **Measured vs Reference chart** — a full-width scatter showing fitted T1 or T2 (ms) per vial (blue filled circles) alongside reference values from `t1t2_reference.json` (red open circles). The vial order matches descending relaxation time.
- **Fit results table** — per-vial T1/T2, S0, R², sorted by descending relaxation time.
- **Embedded MRI viewer** showing the subject T1 anatomical (or first matching contrast as fallback), with only the vials listed in `t1t2_reference.json` shown as toggleable overlays.

---

## Embedded MRI viewer (NiiVue)

All plots contain a self-contained MRI viewer rendered by [NiiVue](https://github.com/niivue/niivue) v0.46.

### How embedding works

1. **NIfTI → base64**: Each NIfTI file is read from disk and base64-encoded in Python. Files stored as float64 are automatically repacked as float32 to halve the payload size.
2. **Blob URLs at runtime**: JavaScript decodes the base64 string into a `Uint8Array`, wraps it in a `Blob`, and creates a temporary `URL.createObjectURL()` URL. NiiVue loads volumes from these in-memory URLs — no HTTP requests are made.
3. **Canvas initialisation**: A `ResizeObserver` waits until the canvas has non-zero painted dimensions before calling `loadVolumes`, ensuring the viewer always renders at the correct size.

### Viewer configuration

- `isAntiAlias: false` — nearest-neighbour interpolation (no blurring).
- `sliceType: 0` — axial single-slice view.
- `dragMode` uses the NiiVue default (1 = drag to adjust contrast), which keeps the scroll wheel free for slice navigation.
- A `{ passive: false }` wheel listener prevents the browser from scrolling the page when the cursor is over the canvas.
- For ADC maps, `setCalMinMax(0, 0.0, 0.004)` is called after loading to window the intensity appropriately.

### Vial overlays

- All vial NIfTIs are embedded alongside the background image.
- Each vial appears as a **red semi-transparent overlay** (opacity 0.5).
- Toggle chip buttons above the viewer show/hide individual vials via `setOpacity`.
- For ADC plots, only the ADC calibration vials (E–L) are included.
- For T1/T2 mapping, only vials listed in `t1t2_reference.json` are included.

---

## Reference data

Reference values live in `template_data/{phantom}/`:

| File | Contents |
|---|---|
| `adc_reference.json` | Per-vial reference ADC values (mm²/s) for vials E–L |
| `t1t2_reference.json` | Per-vial reference T₁ and T₂ relaxation times (ms) from Inversion Recovery and Multi-Echo Spin Echo fits |

Both files share the same JSON structure:
```json
{
  "vials": ["A", "B", ...],
  "<metric>": { "A": <value>, "B": <value>, ... }
}
```

Vial label matching is case-insensitive throughout.

---

## Key libraries

| Library | Role |
|---|---|
| [Chart.js](https://www.chartjs.org/) v4.4 | Interactive scatter/line charts with zoom and pan |
| [NiiVue](https://github.com/niivue/niivue) v0.46 | WebGL MRI viewer; loaded from local bundle (`niivue.umd.min.js`) or CDN fallback |
| [chartjs-plugin-zoom](https://www.chartjs.org/chartjs-plugin-zoom/) | Wheel-zoom and pinch-zoom for Chart.js |
| scipy / numpy | Curve fitting (T1/T2 models) |
| pydra | Task orchestration for the full processing pipeline |
