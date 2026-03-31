#!/usr/bin/env python3
"""
phantom_processor.py
====================
Core phantom processing engine for the phantomkit package.

Contains ``PhantomProcessor``, a class that orchestrates:

  1. ANTs registration
  2. Vial mask inverse-transform to subject space
  3. Per-vial metric extraction from all contrast images
  4. Plot generation (per-contrast scatter plots, T1/T2 parametric maps)
  5. Forward transform of all contrasts to template space

This module is the functional equivalent of ``pydra_phantom_iterative.py``
from the personal repo, re-implemented to integrate with the ``phantomkit``
package structure.  Plotting calls use the ``phantomkit.plotting`` API
directly rather than shelling out to external scripts.

Path conventions (shared repo):
    template_data/<phantom>/ImageTemplate.nii.gz
    template_data/<phantom>/vials_labelled/*.nii.gz
    template_data/<phantom>/adc_reference.json
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required when plotting runs
# in a background thread (e.g. ThreadPoolExecutor on macOS)

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# PhantomProcessor
# =============================================================================


class PhantomProcessor:
    """
    Orchestrate phantom QC processing for a single session.

    Parameters
    ----------
    template_dir:
        Directory for this phantom type, e.g.
        ``<repo>/template_data/SPIRIT``.
        Must contain ``ImageTemplate.nii.gz`` and ``vials_labelled/``.
    output_base_dir:
        Top-level output directory.  Results are written to a
        ``<session_name>/`` subdirectory within this path.
    """

    def __init__(
        self,
        template_dir: str,
        output_base_dir: str,
    ):
        self.template_dir = Path(template_dir)
        self.output_base_dir = Path(output_base_dir)

        # Phantom name is the last component of template_dir (e.g. "SPIRIT")
        self.phantom_name = self.template_dir.name

        self.template_phantom = self.template_dir / "ImageTemplate.nii.gz"
        self.vial_dir = self.template_dir / "vials_labelled"
        self.vial_masks = sorted(self.vial_dir.glob("*.nii.gz"))

        # Load ADC vials from adc_reference.json; fall back to SPIRIT set if absent
        adc_ref_path = self.template_dir / "adc_reference.json"
        if adc_ref_path.exists():
            with open(adc_ref_path) as fh:
                adc_ref = json.load(fh)
            self.adc_vials = {v.upper() for v in adc_ref.get("vials", [])}
        else:
            self.adc_vials = {"E", "F", "G", "H", "I", "J", "K", "L"}

        if not self.template_phantom.exists():
            raise FileNotFoundError(f"Template not found: {self.template_phantom}")
        if len(self.vial_masks) == 0:
            raise FileNotFoundError(f"No vial masks found in: {self.vial_dir}")

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _run_ants_registration(
        self, input_image: str, output_prefix: str
    ) -> Tuple[str, str, str]:
        """Run ANTs rigid registration. Returns (warped, transform, inverse_warped)."""
        cmd = [
            "antsRegistrationSyN.sh",
            "-d",
            "3",
            "-f",
            str(self.template_phantom),
            "-m",
            input_image,
            "-o",
            output_prefix,
            "-t",
            "r",
            "-n",
            "8",
            "-j",
            "1",
        ]
        print("Running ANTs registration...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ANTs failed: {result.stderr}")

        warped = f"{output_prefix}Warped.nii.gz"
        transform = f"{output_prefix}0GenericAffine.mat"
        inverse_warped = f"{output_prefix}InverseWarped.nii.gz"
        return warped, transform, inverse_warped

    def _register(
        self, input_image: str, session_name: str, tmp_dir: Path
    ) -> Tuple[str, str, str]:
        """
        Register the input image to the template phantom.

        Returns
        -------
        (warped, transform, inverse_warped)
        """
        output_prefix = str(tmp_dir / f"{session_name}_Transformed_")
        warped, transform, inverse_warped = self._run_ants_registration(
            input_image, output_prefix
        )
        print(f"  ✓ Registration complete")
        return warped, transform, inverse_warped

    def _transform_vials_to_subject_space(
        self,
        reference_image: str,
        transform_matrix: str,
        output_vial_dir: Path,
    ) -> List[str]:
        """Transform vial masks from template space into subject (scanner) space."""
        output_vial_dir.mkdir(parents=True, exist_ok=True)
        tmp_vial_dir = output_vial_dir.parent / "tmp_vials"
        tmp_vial_dir.mkdir(parents=True, exist_ok=True)

        transformed_vials = []

        for vial_mask in self.vial_masks:
            vial_name = (
                Path(vial_mask)
                .name.replace(".nii.gz", "")
                .replace(".nii", "")
                .split(".")[0]
            )

            tmp_vial = str(tmp_vial_dir / f"{vial_name}.nii")
            output_vial = str(output_vial_dir / f"{vial_name}.nii.gz")

            # Inverse ANTs transform
            cmd = [
                "antsApplyTransforms",
                "-d",
                "3",
                "-i",
                str(vial_mask),
                "-r",
                reference_image,
                "-o",
                tmp_vial,
                "-t",
                f"[{transform_matrix}, 1]",
                "-n",
                "NearestNeighbor",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Transform failed for {vial_name}: {result.stderr}")

            cmd = ["mrconvert", "-quiet", tmp_vial, output_vial, "-force"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Final conversion failed for {vial_name}: {result.stderr}"
                )

            transformed_vials.append(output_vial)

        return transformed_vials

    def _transform_contrast_to_template_space(
        self,
        contrast_file: Path,
        transform_matrix: str,
        tmp_dir: Path,
    ) -> str:
        """
        Forward-transform a contrast image from subject space to template space.

        Handles 4D series and single-slice (degenerate z) images, which cause
        ``antsApplyTransforms -d 3`` to fail silently without padding.

        Returns
        -------
        str
            Path to the contrast image in template space.
        """
        tmp_dir.mkdir(parents=True, exist_ok=True)
        contrast_name = contrast_file.stem.replace(".nii", "")

        source_image = str(contrast_file)

        # Detect 4D and single-slice
        detect = subprocess.run(
            ["mrinfo", "-size", source_image], capture_output=True, text=True
        )
        size_parts = detect.stdout.strip().split()
        is_4d = len(size_parts) >= 4 and int(size_parts[3]) > 1
        is_single_slice = len(size_parts) >= 3 and int(size_parts[2]) == 1

        if is_single_slice:
            padded = str(tmp_dir / f"{contrast_name}_padded.nii.gz")
            cmd = [
                "mrgrid",
                source_image,
                "pad",
                "-axis",
                "2",
                "1,1",
                padded,
                "-force",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Padding of single-slice contrast {contrast_name} failed: "
                    f"{result.stderr}"
                )
            transform_input = padded
        else:
            transform_input = source_image

        warped_tmp = str(tmp_dir / f"{contrast_name}_template_space_tmp.nii.gz")
        warped_contrast = str(tmp_dir / f"{contrast_name}_template_space.nii.gz")

        cmd = [
            "antsApplyTransforms",
            "-d",
            "3",
            "-e",
            "3" if is_4d else "0",
            "-i",
            transform_input,
            "-r",
            str(self.template_phantom),
            "-o",
            warped_tmp,
            "-t",
            transform_matrix,
            "-n",
            "Linear",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Forward transform of contrast {contrast_name} failed: {result.stderr}"
            )

        # Validate output was actually produced
        verify = subprocess.run(
            ["mrinfo", "-size", warped_tmp], capture_output=True, text=True
        )
        if verify.returncode != 0 or not verify.stdout.strip():
            raise RuntimeError(
                f"antsApplyTransforms produced no valid output for {contrast_name}. "
                f"stderr: {result.stderr}"
            )

        shutil.move(warped_tmp, warped_contrast)
        return warped_contrast

    @staticmethod
    def _classify_contrast(contrast_file: Path) -> Optional[str]:
        """
        Classify a contrast image by its filename stem.

        Returns
        -------
        "adc"  – filename contains 'ADC' (case-insensitive)
        "fa"   – filename contains standalone 'FA' (whole-word, case-insensitive)
        None   – no special classification
        """
        stem = contrast_file.stem
        if re.search(r"ADC", stem, re.IGNORECASE):
            return "adc"
        if re.search(r"(?<![A-Za-z0-9])FA(?![A-Za-z0-9])", stem):
            return "fa"
        return None

    def _extract_metrics_from_contrast(
        self,
        contrast_file: Path,
        vial_masks: List[str],
        output_metrics_dir: Path,
        session_name: str,
    ) -> Dict:
        """Extract per-vial statistics from one contrast image across all vials."""
        # Strip both .nii.gz and .nii from contrast name
        contrast_name = contrast_file.name
        for ext in (".nii.gz", ".nii"):
            if contrast_name.endswith(ext):
                contrast_name = contrast_name[: -len(ext)]
                break

        clean_contrast_name = contrast_name

        contrast_type = self._classify_contrast(contrast_file)
        if contrast_type == "adc":
            original_count = len(vial_masks)
            vial_masks = [
                m
                for m in vial_masks
                if Path(m)
                .name.replace(".nii.gz", "")
                .replace(".nii", "")
                .split(".")[0]
                .upper()
                in self.adc_vials
            ]
            print(
                f"  [ADC mode] Restricting to {len(self.adc_vials)} ADC vials "
                f"({len(vial_masks)} of {original_count})"
            )

        # Get number of volumes
        cmd = ["mrinfo", "-size", str(contrast_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        size_info = result.stdout.strip().split()
        nvols = (
            int(size_info[3]) if len(size_info) >= 4 and int(size_info[3]) > 0 else 1
        )

        print(
            f"  Processing {clean_contrast_name} "
            f"({nvols} volume{'s' if nvols > 1 else ''})"
        )

        metrics_data: Dict[str, Dict[str, List[float]]] = {
            "mean": {},
            "median": {},
            "std": {},
            "min": {},
            "max": {},
        }

        tmp_vol_dir = output_metrics_dir.parent / "tmp_vols"
        tmp_vol_dir.mkdir(parents=True, exist_ok=True)

        for vial_mask in vial_masks:
            vial_name = (
                Path(vial_mask)
                .name.replace(".nii.gz", "")
                .replace(".nii", "")
                .split(".")[0]
            )
            for metric in metrics_data:
                metrics_data[metric][vial_name] = []

            regridded_mask = str(tmp_vol_dir / f"{contrast_name}_{vial_name}.nii")
            cmd = [
                "mrgrid",
                "-template",
                str(contrast_file),
                vial_mask,
                "regrid",
                regridded_mask,
                "-force",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"mrgrid regrid failed for vial {vial_name}: {result.stderr}"
                )

            for vol_idx in range(nvols):
                if nvols == 1:
                    vol_file = str(contrast_file)
                else:
                    vol_file = str(tmp_vol_dir / f"{contrast_name}_vol{vol_idx}.nii.gz")
                    cmd = [
                        "mrconvert",
                        str(contrast_file),
                        "-coord",
                        "3",
                        str(vol_idx),
                        vol_file,
                        "-quiet",
                        "-force",
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)

                cmd = [
                    "mrstats",
                    "-quiet",
                    vol_file,
                    "-output",
                    "mean",
                    "-output",
                    "median",
                    "-output",
                    "std",
                    "-output",
                    "min",
                    "-output",
                    "max",
                    "-mask",
                    regridded_mask,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0 or not result.stdout.strip():
                    raise RuntimeError(
                        f"mrstats failed for vial {vial_name}: {result.stderr}"
                    )
                values = result.stdout.strip().split()
                metrics_data["mean"][vial_name].append(float(values[0]))
                metrics_data["median"][vial_name].append(float(values[1]))
                metrics_data["std"][vial_name].append(float(values[2]))
                metrics_data["min"][vial_name].append(float(values[3]))
                metrics_data["max"][vial_name].append(float(values[4]))

        # Write CSV files
        for metric_name, vial_data in metrics_data.items():
            csv_file = (
                output_metrics_dir
                / f"{session_name}_{contrast_name}_{metric_name}_matrix.csv"
            )
            rows = [
                {
                    "vial": vn,
                    **{f"{clean_contrast_name}_vol{i}": v for i, v in enumerate(vals)},
                }
                for vn, vals in vial_data.items()
            ]
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            print(f"    Saved: {csv_file.name}")

        return metrics_data

    def _generate_mrview_screenshot(
        self,
        contrast_file: Path,
        roi_overlay: str,
        output_image: str,
        intensity_range: Optional[Tuple[float, float]] = None,
    ) -> Optional[str]:
        """
        Capture an mrview screenshot with a vial ROI overlay.

        Parameters
        ----------
        intensity_range:
            If provided, passes ``-intensity_range min,max`` to mrview.
            Use (0, 1) for FA maps and (0, 0.005) for ADC maps.
        """
        cmd = [
            "mrview",
            str(contrast_file),
            "-mode",
            "1",
            "-plane",
            "2",
            "-interpolation",
            "0",
            "-roi.load",
            roi_overlay,
            "-roi.colour",
            "1,0,0",
            "-roi.opacity",
            "1",
            "-comments",
            "0",
            "-noannotations",
            "-fullscreen",
        ]

        if intensity_range is not None:
            cmd.extend(
                ["-intensity_range", f"{intensity_range[0]},{intensity_range[1]}"]
            )

        cmd.extend(
            [
                "-capture.folder",
                str(Path(output_image).parent),
                "-capture.prefix",
                Path(output_image).stem,
                "-capture.grab",
                "-exit",
            ]
        )

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    ⚠ mrview screenshot failed: {result.stderr}")
            return None

        # mrview appends "0000" to the prefix
        actual_file = str(
            Path(output_image).parent / f"{Path(output_image).stem}0000.png"
        )
        return actual_file if Path(actual_file).exists() else None

    def _build_roi_overlay(
        self,
        contrast_file: Path,
        vial_masks_list: List[Path],
        prefix: str,
        tmp_vial_dir: Path,
    ) -> Optional[str]:
        """Regrid each vial mask to contrast space and combine into one overlay NIfTI."""
        roi_overlay = str(tmp_vial_dir / f"{prefix}_VialsCombined.nii.gz")
        regridded_vials = []

        for vial_mask in vial_masks_list:
            vial_name = vial_mask.name.replace(".nii.gz", "").replace(".nii", "")
            regridded = str(tmp_vial_dir / f"{prefix}_{vial_name}.nii")
            cmd = [
                "mrgrid",
                "-template",
                str(contrast_file),
                str(vial_mask),
                "regrid",
                regridded,
                "-force",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            regridded_vials.append(regridded)

        if not regridded_vials:
            return None

        cmd_cat = ["mrcat"] + regridded_vials + ["-", "-axis", "3"]
        cmd_math = ["mrmath", "-", "max", roi_overlay, "-axis", "3", "-force"]

        proc_cat = subprocess.Popen(
            cmd_cat, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        proc_math = subprocess.Popen(
            cmd_math,
            stdin=proc_cat.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc_cat.stdout.close()
        proc_math.communicate()

        return roi_overlay if Path(roi_overlay).exists() else None

    def _generate_plots(
        self,
        contrast_files: List[Path],
        metrics_dir: Path,
        vial_dir: Path,
        session_name: str,
    ):
        """
        Generate per-contrast scatter plots and parametric map plots (IR / TE).

        Plotting functions are imported directly from ``phantomkit.plotting``
        rather than shelled out to external scripts.
        """
        from phantomkit.plotting.vial_intensity import plot_vial_intensity
        from phantomkit.plotting.maps_ir import plot_vial_ir_means_std
        from phantomkit.plotting.maps_te import plot_vial_te_means_std

        tmp_vial_dir = vial_dir / "tmp"
        tmp_vial_dir.mkdir(exist_ok=True)
        vial_masks_list = list(vial_dir.glob("*.nii.gz"))

        # ── Per-contrast scatter plots ────────────────────────────────────────
        for contrast_file in contrast_files:
            contrast_name = contrast_file.name
            for ext in (".nii.gz", ".nii"):
                if contrast_name.endswith(ext):
                    contrast_name = contrast_name[: -len(ext)]
                    break

            mean_csv = metrics_dir / f"{session_name}_{contrast_name}_mean_matrix.csv"
            std_csv = metrics_dir / f"{session_name}_{contrast_name}_std_matrix.csv"

            if not mean_csv.exists():
                print(f"  ⚠ Mean CSV not found, skipping plot for {contrast_name}")
                continue

            output_plot = str(
                metrics_dir / f"{session_name}_{contrast_name}_PLOTmeanstd.png"
            )

            # Build ROI overlay and screenshot
            roi_image = None
            if vial_masks_list:
                roi_overlay = self._build_roi_overlay(
                    contrast_file, vial_masks_list, contrast_name, tmp_vial_dir
                )
                if roi_overlay:
                    contrast_type = self._classify_contrast(contrast_file)
                    intensity_range = (
                        (0, 1)
                        if contrast_type == "fa"
                        else (0, 0.005) if contrast_type == "adc" else None
                    )
                    screenshot_base = str(
                        tmp_vial_dir / f"{contrast_name}_roi_overlay.png"
                    )
                    actual_screenshot = self._generate_mrview_screenshot(
                        contrast_file,
                        roi_overlay,
                        screenshot_base,
                        intensity_range=intensity_range,
                    )
                    if actual_screenshot and Path(actual_screenshot).exists():
                        roi_image = actual_screenshot

            try:
                plot_vial_intensity(
                    csv_file=str(mean_csv),
                    plot_type="scatter",
                    std_csv=str(std_csv) if std_csv.exists() else None,
                    roi_image=roi_image,
                    output=output_plot,
                    phantom=self.phantom_name,
                    # template_dir is the parent of the phantom-specific dir
                    template_dir=str(self.template_dir.parent),
                )
                print(f"    ✓ Generated plot: {Path(output_plot).name}")
            except Exception as e:
                print(f"    ✗ Plot generation failed for {contrast_name}: {e}")

        # ── Parametric map plots (IR and TE) ──────────────────────────────────
        def _matches(stem: str, token: str) -> bool:
            return bool(re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", stem.lower()))

        for contrast_type_key, plot_fn, suffix in [
            ("ir", plot_vial_ir_means_std, "ir_map_PLOTmeanstd_T1mapping.png"),
            ("te", plot_vial_te_means_std, "TE_map_PLOTmeanstd_TEmapping.png"),
        ]:
            matching = [
                f for f in contrast_files if _matches(f.stem, contrast_type_key)
            ]
            if not matching:
                print(
                    f"  No {contrast_type_key.upper()} contrasts found "
                    f"(searched for '{contrast_type_key}' in filenames)"
                )
                continue

            print(
                f"  Found {len(matching)} {contrast_type_key.upper()} contrasts: "
                f"{[f.name for f in matching]}"
            )

            output_plot = str(metrics_dir / f"{session_name}_{suffix}")
            first_file = matching[0]

            # Build ROI overlay for first contrast of this type
            roi_image_arg = None
            if vial_masks_list:
                overlay_file = self._build_roi_overlay(
                    first_file,
                    vial_masks_list,
                    contrast_type_key,
                    tmp_vial_dir,
                )
                if overlay_file:
                    screenshot_base = str(
                        tmp_vial_dir / f"roi_overlay_{contrast_type_key}.png"
                    )
                    actual_screenshot = self._generate_mrview_screenshot(
                        first_file, overlay_file, screenshot_base
                    )
                    if actual_screenshot and Path(actual_screenshot).exists():
                        roi_image_arg = actual_screenshot

            try:
                plot_fn(
                    contrast_files=[str(f) for f in matching],
                    metric_dir=str(metrics_dir),
                    output_file=output_plot,
                    roi_image=roi_image_arg,
                )
                print(f"    ✓ Generated {contrast_type_key.upper()} map plot")
            except Exception as e:
                print(f"    ✗ {contrast_type_key.upper()} map plot failed: {e}")

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def process_session(self, input_image: str) -> Dict:
        """
        Process a single phantom session end-to-end.

        Runs both the subject-space pipeline (vial mask transform, metric
        extraction, plot generation) and the template-space pipeline (forward
        transform of all contrasts).

        Parameters
        ----------
        input_image:
            Path to the primary input image (T1 MPRAGE or T1 in DWI space).

        Returns
        -------
        dict
            Processing results and output paths.
        """
        input_path = Path(input_image)
        session_name = input_path.parent.name

        output_dir = self.output_base_dir / session_name
        tmp_dir = output_dir / "tmp"
        vial_dir = output_dir / "vial_segmentations"
        metrics_dir = output_dir / "metrics"
        images_template_space_dir = output_dir / "images_template_space"

        for d in [tmp_dir, vial_dir, metrics_dir, images_template_space_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Processing Session: {session_name}")
        print(f"Input: {input_image}")
        print(f"Output: {output_dir}")
        print(f"{'=' * 60}\n")

        # Step 1: Registration
        print("Step 1: Registration")
        warped, transform, inverse_warped = self._register(
            str(input_image), session_name, tmp_dir
        )

        # Save template in scanner space
        template_scanner_space = str(output_dir / "TemplatePhantom_ScannerSpace.nii.gz")
        cmd = [
            "mrconvert",
            "-quiet",
            inverse_warped,
            template_scanner_space,
            "-force",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ✓ Saved template in scanner space")

        # Gather all contrast images — exclude derived template scanner space file
        contrast_files = [
            f
            for f in input_path.parent.glob("*.nii.gz")
            if f.name != "TemplatePhantom_ScannerSpace.nii.gz"
        ]

        # Step 2: Transform vials to subject space
        print("\nStep 2: Transforming vials to subject space")
        transformed_vials = self._transform_vials_to_subject_space(
            reference_image=str(input_image),
            transform_matrix=transform,
            output_vial_dir=vial_dir,
        )
        print(f"  ✓ Transformed {len(transformed_vials)} vial masks")

        # Step 3: Extract metrics from all contrasts
        print("\nStep 3: Extracting metrics from all contrasts")
        print(f"  Found {len(contrast_files)} contrast image(s)")
        all_metrics = {}
        for contrast_file in contrast_files:
            metrics_data = self._extract_metrics_from_contrast(
                contrast_file=contrast_file,
                vial_masks=transformed_vials,
                output_metrics_dir=metrics_dir,
                session_name=session_name,
            )
            all_metrics[contrast_file.name] = metrics_data

        # Step 4: Generate plots
        print("\nStep 4: Generating plots")
        self._generate_plots(
            contrast_files=contrast_files,
            metrics_dir=metrics_dir,
            vial_dir=vial_dir,
            session_name=session_name,
        )

        # Step 5: Forward-transform all contrasts to template space
        print("\nStep 5: Transforming all contrasts to template space")
        tmp_template_space_dir = tmp_dir / "template_space_contrasts"
        tmp_template_space_dir.mkdir(parents=True, exist_ok=True)

        for contrast_file in contrast_files:
            print(f"  Transforming: {contrast_file.name}")
            warped_path = self._transform_contrast_to_template_space(
                contrast_file=contrast_file,
                transform_matrix=transform,
                tmp_dir=tmp_template_space_dir,
            )
            shutil.copy2(
                warped_path,
                str(images_template_space_dir / contrast_file.name),
            )
            print(f"    ✓ {contrast_file.name} → template space")
        print(f"  ✓ All contrasts saved to: {images_template_space_dir}")

        # Step 6: Cleanup
        print("\nStep 6: Cleaning up temporary directories")
        for temp_dir in [
            tmp_dir,
            output_dir / "tmp_vials",
            output_dir / "tmp_vols",
            vial_dir / "tmp",
        ]:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    print(f"  ✓ Removed: {temp_dir.name}")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not remove {temp_dir.name}: {e}")

        print(f"\n{'=' * 60}")
        print(f"✓ Session {session_name} complete!")
        print(f"  Metrics:               {metrics_dir}")
        print(f"  Vial masks:            {vial_dir}")
        print(f"  Template-space images: {images_template_space_dir}")
        print(f"{'=' * 60}\n")

        return {
            "session": session_name,
            "output_dir": str(output_dir),
            "metrics_dir": str(metrics_dir),
            "vial_dir": str(vial_dir),
            "images_template_space_dir": str(images_template_space_dir),
            "space_image": str(output_dir / "TemplatePhantom_ScannerSpace.nii.gz"),
            "metrics": all_metrics,
        }
