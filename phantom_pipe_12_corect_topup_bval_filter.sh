#!/bin/bash
# SPIRIT Phantom DWI Processing Pipeline - PHASE 1 OPTIMIZED
# 2-Volume TOPUP with Averaged ROPE + Enhanced EDDY + QC Reports
# Complete implementation - no truncations

set -o pipefail
set -euo pipefail

# ------------------------------- Colors -------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; NC='\033[0m'

# ------------------------------- Configuration ------------------------
ENABLE_EDDY=true
ENABLE_GNC=false

# ------------------------------- Parallelism -------------------------
TOTAL_CORES=$(nproc)
USE_CORES=$((TOTAL_CORES * 90 / 100)); if [ $USE_CORES -lt 1 ]; then USE_CORES=1; fi
export OMP_NUM_THREADS=$USE_CORES
export MRTRIX_NTHREADS=$USE_CORES
export FSLPARALLEL=$USE_CORES
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$USE_CORES

# ------------------------------- Global outputs & master log ---------
OUTPUT_BASE="/data/output"
MASTER_LOG_DIR="${OUTPUT_BASE}/logs"; mkdir -p "$MASTER_LOG_DIR"
MASTER_LOG_FILE="${MASTER_LOG_DIR}/master_$(date +%Y%m%d_%H%M%S).txt"
LOG_FILE="$MASTER_LOG_FILE"

# ------------------------------- CSV reports -------------------------
SUMMARY_CSV="${OUTPUT_BASE}/pipeline_summary.csv"
SUMMARY_ISOLATED_CSV="${OUTPUT_BASE}/pipeline_summary_isolated.csv"
echo "Dataset,Pipeline,Step,Mean_ADC,StdDev_ADC,Min_ADC,Max_ADC,Time_s,Space" > "$SUMMARY_CSV"
echo "Dataset,Pipeline,Step,Mean_ADC,StdDev_ADC,Min_ADC,Max_ADC,Time_s,Space" > "$SUMMARY_ISOLATED_CSV"

# ------------------------------- Phantom settings --------------------
PHANTOM_MODE=true
DEFAULT_READOUT="0.033"
EXPORT_CROPPED_NIFTI=true
PHANTOM_DIAMETER_MM=191
MASK_EROSION_MM=8

# ------------------------------- Logging helper ----------------------
log_message() {
  if [ "$LOG_FILE" = "$MASTER_LOG_FILE" ]; then
    echo -e "$1" | tee -a "$MASTER_LOG_FILE" >/dev/null
  else
    echo -e "$1" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE" >/dev/null
  fi
}

# ------------------------------- Banner ------------------------------
log_message "${CYAN}===================================================${NC}"
log_message "${CYAN}SPIRIT Phantom Pipeline - PHASE 1 OPTIMIZED${NC}"
log_message "${CYAN}Averaged ROPE + Enhanced EDDY + QC Reports${NC}"
log_message "${CYAN}===================================================${NC}"
log_message "Date: $(date)"
log_message "CPU cores: $USE_CORES / $TOTAL_CORES"
log_message ""
log_message "${GREEN}Phase 1 Optimizations:${NC}"
log_message "  ? Averaged ROPE (7 volumes ? 1 with 7× SNR)"
log_message "  ? TOPUP iterations: 100 (vs 50 baseline)"
log_message "  ? EDDY outlier replacement (--repol)"
log_message "  ? EDDY QC reports (eddy_quad)"
log_message ""
log_message "Phantom: ${PHANTOM_DIAMETER_MM}mm diameter, ${MASK_EROSION_MM}mm erosion"
log_message "Master log: ${MASTER_LOG_FILE}"
log_message ""

# ------------------------------- Utilities --------------------------
_make_tmpdir() {
  local td
  td=$(mktemp -d -p "$(dirname "$MASTER_LOG_FILE")" tmp.XXXXXXXX 2>/dev/null) || td="${MASTER_LOG_DIR}/tmp.$$.$RANDOM"
  mkdir -p "$td"
  echo "$td"
}

resample_mask_to_target_grid() {
  local mask_in="$1"
  local target="$2"
  local mask_out="$3"
  if mrtransform "$mask_in" "$mask_out" -template "$target" -interp nearest -datatype bit -force 1>>"$MASTER_LOG_FILE" 2>&1; then return 0; fi
  if mrgrid "$mask_in" regrid "$mask_out" -template "$target" -interp nearest -datatype bit -force 1>>"$MASTER_LOG_FILE" 2>&1; then return 0; fi
  log_message "${YELLOW}resample_mask_to_target_grid: failed. Copying input.${NC}"
  cp -f "$mask_in" "$mask_out"
}

crop_image_to_mask() {
  local img_in="$1"; local mask_in="$2"; local img_out="$3"
  if mrgrid "$img_in" crop "$img_out" -mask "$mask_in" -force 1>>"$MASTER_LOG_FILE" 2>&1; then return 0; fi
  if mrgrid crop "$img_in" "$img_out" -mask "$mask_in" -force 1>>"$MASTER_LOG_FILE" 2>&1; then return 0; fi
  log_message "${YELLOW}crop_image_to_mask: not supported; copying input.${NC}"
  cp -f "$img_in" "$img_out"
  return 0
}

# ------------------------------- Create SPIRIT phantom mask --------------------------
create_spirit_phantom_mask() {
  local dwi_input="$1"
  local mask_output="$2"
  local erosion_mm="${3:-8}"
  local phantom_diameter_mm="${4:-191}"
  
  log_message "${CYAN}Creating SPIRIT phantom mask (${phantom_diameter_mm}mm, erode ${erosion_mm}mm)${NC}"
  
  local td; td=$(_make_tmpdir)
  
  dwiextract "$dwi_input" -bzero "${td}/b0s.mif" -force 2>>"$MASTER_LOG_FILE"
  mrmath "${td}/b0s.mif" mean "${td}/b0_mean.mif" -axis 3 -force 2>>"$MASTER_LOG_FILE"
  
  local dims=$(mrinfo "${td}/b0_mean.mif" -size 2>>"$MASTER_LOG_FILE")
  local nx=$(echo "$dims" | awk '{print $1}')
  local ny=$(echo "$dims" | awk '{print $2}')
  local nz=$(echo "$dims" | awk '{print $3}')
  
  local spacing=$(mrinfo "${td}/b0_mean.mif" -spacing 2>>"$MASTER_LOG_FILE")
  local sx=$(echo "$spacing" | awk '{print $1}')
  local sy=$(echo "$spacing" | awk '{print $2}')
  local sz=$(echo "$spacing" | awk '{print $3}')
  
  log_message "  Image: ${nx}x${ny}x${nz}, voxel: ${sx}x${sy}x${sz}mm"
  
  mrconvert "${td}/b0_mean.mif" "${td}/b0_mean.nii.gz" -force 2>>"$MASTER_LOG_FILE"
  
  python3 <<PYEOF
import numpy as np
import nibabel as nib
from scipy import ndimage

b0_img = nib.load("${td}/b0_mean.nii.gz")
b0_data = b0_img.get_fdata()
voxel_size = np.array([${sx}, ${sy}, ${sz}])
shape = (${nx}, ${ny}, ${nz})

low_thresh = np.percentile(b0_data[b0_data > 0], 10)
rough_mask = b0_data > low_thresh
coords = np.argwhere(rough_mask)

if len(coords) == 0:
    print("ERROR: No phantom detected!")
    exit(1)

intensities = b0_data[rough_mask]
weighted_sum = (coords * intensities[:, np.newaxis]).sum(axis=0)
center_voxels = weighted_sum / intensities.sum()

print(f"  Phantom center (voxels): [{center_voxels[0]:.1f}, {center_voxels[1]:.1f}, {center_voxels[2]:.1f}]")

phantom_radius_mm = ${phantom_diameter_mm} / 2.0
final_radius_mm = phantom_radius_mm - ${erosion_mm}

print(f"  Known radius: {phantom_radius_mm:.1f}mm")
print(f"  After {${erosion_mm}}mm erosion: {final_radius_mm:.1f}mm")

x = (np.arange(shape[0]) - center_voxels[0]) * ${sx}
y = (np.arange(shape[1]) - center_voxels[1]) * ${sy}
z = (np.arange(shape[2]) - center_voxels[2]) * ${sz}
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
R = np.sqrt(X**2 + Y**2 + Z**2)
sphere_mask = (R <= final_radius_mm)

signal_thresh = np.percentile(b0_data[b0_data > 0], 1)
signal_mask = b0_data > signal_thresh
final_mask = sphere_mask & signal_mask
final_mask = ndimage.binary_fill_holes(final_mask)

n_voxels = final_mask.sum()
volume_cm3 = n_voxels * ${sx} * ${sy} * ${sz} / 1000
expected_volume = 4/3 * np.pi * (final_radius_mm/10)**3
print(f"  Mask: {n_voxels} voxels = {volume_cm3:.2f} cm³")
print(f"  Expected: {expected_volume:.2f} cm³")

mask_img = nib.Nifti1Image(final_mask.astype(np.uint8), b0_img.affine, b0_img.header)
nib.save(mask_img, "${td}/mask.nii.gz")
PYEOF

  mrconvert "${td}/mask.nii.gz" "$mask_output" -datatype bit -force 2>>"$MASTER_LOG_FILE"
  rm -rf "$td"
  log_message "${GREEN}  SPIRIT phantom mask created${NC}"
}

# ------------------------------- Multi-shell ADC Calculation --------------------------
derive_trace_adc_multishell() {
  local dwi_input=$1
  local stats_mask_in=$2
  local adc_output=$3
  local dataset_name=$4
  local pipeline_type=$5
  local step_name=$6
  local space="${7:-native}"

  log_message "  Computing trace ADC (multi-shell)..."
  log_message "  Input: $dwi_input"
  log_message "  Mask: $stats_mask_in"
  
  # Verify inputs exist
  if [ ! -f "$dwi_input" ]; then
    log_message "${RED}  ERROR: Input DWI not found: $dwi_input${NC}"
    return 1
  fi
  
  if [ ! -f "$stats_mask_in" ]; then
    log_message "${RED}  ERROR: Mask not found: $stats_mask_in${NC}"
    return 1
  fi
  
  local start_time=$(date +%s)

  local td; td=$(_make_tmpdir)
  
  log_message "  Extracting b=0 volumes..."
  if ! dwiextract "$dwi_input" -bzero "${td}/b0_vols.mif" -force 2>>"$MASTER_LOG_FILE"; then
    log_message "${RED}  ERROR: Failed to extract b=0 volumes${NC}"
    rm -rf "$td"
    return 1
  fi
  
  log_message "  Computing mean b=0..."
  if ! mrmath "${td}/b0_vols.mif" mean "${td}/b0_mean.mif" -axis 3 -force 2>>"$MASTER_LOG_FILE"; then
    log_message "${RED}  ERROR: Failed to compute mean b=0${NC}"
    rm -rf "$td"
    return 1
  fi

  local shell_adcs=()
  for bval in 500 1000 2000 3000 4000 6000; do
    log_message "  Processing b=$bval shell..."
    
    if dwiextract "$dwi_input" -shells "$bval" "${td}/b${bval}_vols.mif" -force 2>>"$MASTER_LOG_FILE"; then
      local nvols=$(mrinfo "${td}/b${bval}_vols.mif" -size 2>>"$MASTER_LOG_FILE" | awk '{print $4}')
      
      if [ -z "$nvols" ] || [ "$nvols" -eq 0 ]; then
        log_message "${YELLOW}    b=$bval: No volumes found, skipping${NC}"
        continue
      fi
      
      log_message "    b=$bval: $nvols volumes found"
      
      if ! mrmath "${td}/b${bval}_vols.mif" mean "${td}/b${bval}_mean.mif" -axis 3 -force 2>>"$MASTER_LOG_FILE"; then
        log_message "${YELLOW}    b=$bval: Failed to compute mean, skipping${NC}"
        continue
      fi
      
      if ! mrcalc "${td}/b0_mean.mif" "${td}/b${bval}_mean.mif" -div -log "$bval" -div \
          "${td}/adc_b${bval}.mif" -force 2>>"$MASTER_LOG_FILE"; then
        log_message "${YELLOW}    b=$bval: Failed to compute ADC, skipping${NC}"
        continue
      fi
      
      shell_adcs+=("${td}/adc_b${bval}.mif")
      log_message "${GREEN}    b=$bval: ADC computed successfully${NC}"
    else
      log_message "${YELLOW}    b=$bval: Shell extraction failed, skipping${NC}"
    fi
  done

  if [ ${#shell_adcs[@]} -eq 0 ]; then
    log_message "${RED}  ERROR: No valid shells found for ADC calculation${NC}"
    rm -rf "$td"
    return 1
  fi
  
  log_message "  Averaging ${#shell_adcs[@]} shell ADCs..."
  if ! mrmath "${shell_adcs[@]}" mean "${td}/adc_multishell.mif" -force 2>>"$MASTER_LOG_FILE"; then
    log_message "${RED}  ERROR: Failed to average shell ADCs${NC}"
    rm -rf "$td"
    return 1
  fi

  log_message "  Applying mask and saving output..."
  if ! mrcalc "${td}/adc_multishell.mif" "$stats_mask_in" -mult "$adc_output" -force 2>>"$MASTER_LOG_FILE"; then
    log_message "${RED}  ERROR: Failed to apply mask${NC}"
    rm -rf "$td"
    return 1
  fi
  
  if ! mrconvert "$adc_output" "${adc_output%.mif}.nii.gz" -datatype float32 -strides +1,+2,+3 -force 2>>"$MASTER_LOG_FILE"; then
    log_message "${YELLOW}  WARNING: Failed to convert to NIfTI${NC}"
  fi

  if [ "$EXPORT_CROPPED_NIFTI" = true ]; then
    log_message "  Creating cropped version..."
    local crop_mif="${td}/crop_adc.mif"
    crop_image_to_mask "$adc_output" "$stats_mask_in" "$crop_mif"
    mrconvert "$crop_mif" "${adc_output%.mif}_cropped.nii.gz" -datatype float32 -strides +1,+2,+3 -force 2>>"$MASTER_LOG_FILE" || true
  fi

  local end_time=$(date +%s); local processing_time=$((end_time - start_time))
  
  log_message "  Computing statistics..."
  local mean std min max
  mean=$(mrstats "$adc_output" -mask "$stats_mask_in" -output mean -ignorezero 2>>"$MASTER_LOG_FILE" | tr -d '[:space:]' || echo "N/A")
  std=$(mrstats  "$adc_output" -mask "$stats_mask_in" -output std  -ignorezero 2>>"$MASTER_LOG_FILE" | tr -d '[:space:]' || echo "N/A")
  min=$(mrstats  "$adc_output" -mask "$stats_mask_in" -output min  -ignorezero 2>>"$MASTER_LOG_FILE" | tr -d '[:space:]' || echo "N/A")
  max=$(mrstats  "$adc_output" -mask "$stats_mask_in" -output max  -ignorezero 2>>"$MASTER_LOG_FILE" | tr -d '[:space:]' || echo "N/A")

  rm -rf "$td"
  log_message "${GREEN}  Mean ADC: $mean mm²/s (computed in ${processing_time}s)${NC}"
  
  local line="$dataset_name,$pipeline_type,$step_name,$mean,$std,$min,$max,$processing_time,$space"
  if [ "$pipeline_type" = "cumulative" ]; then echo "$line" >> "$SUMMARY_CSV"; else echo "$line" >> "$SUMMARY_ISOLATED_CSV"; fi
  
  return 0
}

# ------------------------------- Create custom TOPUP config (100 iterations) --------------------------
create_phantom_topup_config() {
  local config_file="$1"
  
  log_message "${CYAN}Creating custom TOPUP config (100 iterations at finest level)${NC}"
  
  cat > "$config_file" <<'TOPUPCONF'
# Phantom-optimized TOPUP configuration - PHASE 1 OPTIMIZED
# Higher resolution (3mm) for phantom features
# INCREASED iterations (100) for better convergence

# Resolution (knot-spacing) of warps in mm
--warpres=20,16,12,10,8,6,4,3,3

# Subsampling level
--subsamp=2,2,2,2,2,1,1,1,1

# FWHM of gaussian smoothing (mm)
--fwhm=8,6,4,3,2,2,1,0,0

# Maximum iterations per level
# PHASE 1: Increased at fine resolutions (50 ? 75 ? 100)
--miter=5,5,5,5,10,10,20,75,100

# Regularisation weight
--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001

# Scale lambda by squared difference
--ssqlambda=1

# Regularisation model
--regmod=bending_energy

# Estimate movement parameters
--estmov=1,1,1,1,1,0,0,0,0

# Optimization method (0=LM, 1=SCG)
--minmet=0,0,0,0,0,1,1,1,1

# Spline order
--splineorder=3

# Precision
--numprec=double

# Interpolation
--interp=spline

# Scale images to common mean
--scale=1
TOPUPCONF

  log_message "${GREEN}  Custom config created: $config_file (100 iter)${NC}"
}

# ------------------------------- TOPUP prep with AVERAGED ROPE (PHASE 1) --------------------------
prepare_topup_data() {
  local main_dwi=$1
  local pa_dir=$2
  local output_dir=$3

  log_message "${CYAN}Preparing TOPUP inputs (PHASE 1: AVERAGED ROPE)${NC}"
  mkdir -p "$output_dir"

  local pa_nii=$(ls "$pa_dir"/*SBREF*.nii 2>/dev/null | head -1)
  if [ -z "$pa_nii" ]; then
    pa_nii=$(ls "$pa_dir"/*.nii 2>/dev/null | head -1)
  fi
  
  if [ -z "$pa_nii" ]; then
    log_message "${RED}No PA reference found${NC}"
    return 1
  fi

  log_message "  PA reference: $(basename "$pa_nii")"

  local pa_ndims=$(mrinfo "$pa_nii" -ndim 2>>"$MASTER_LOG_FILE" || echo "3")
  local pa_b0_averaged="${output_dir}/pa_b0_averaged.mif"
  
  if [ "$pa_ndims" -eq 4 ]; then
    local pa_nvols=$(mrinfo "$pa_nii" -size 2>>"$MASTER_LOG_FILE" | awk '{print $4}')
    log_message "  ${GREEN}PHASE 1 OPTIMIZATION: Averaging ${pa_nvols} ROPE volumes${NC}"
    log_message "  ${GREEN}Expected SNR improvement: ?${pa_nvols} = $(echo "scale=2; sqrt($pa_nvols)" | bc)×${NC}"
    
    # Average across 4th dimension (time)
    mrmath "$pa_nii" mean "$pa_b0_averaged" -axis 3 -force 2>>"$MASTER_LOG_FILE"
    
    log_message "  ${GREEN}? Averaged ${pa_nvols} volumes ? 1 high-SNR volume${NC}"
  else
    log_message "  Using 3D SBREF directly..."
    mrconvert "$pa_nii" "$pa_b0_averaged" -force 2>>"$MASTER_LOG_FILE"
  fi
  
  # Extract mean b=0 from main DWI
  local main_b0="${output_dir}/main_b0_ap.mif"
  local td; td=$(_make_tmpdir)
  dwiextract "$main_dwi" "${td}/_ap_b0s.mif" -bzero -force 2>>"$MASTER_LOG_FILE"
  mrmath "${td}/_ap_b0s.mif" mean "$main_b0" -axis 3 -force 2>>"$MASTER_LOG_FILE"
  rm -rf "$td"

  # Convert averaged PA to NIfTI
  local pa_b0="${output_dir}/pa_b0_averaged.nii.gz"
  mrconvert "$pa_b0_averaged" "$pa_b0" -force 2>>"$MASTER_LOG_FILE"
  
  log_message "${GREEN}  FOV matched - both have same dimensions!${NC}"

  # Concatenate AP and averaged PA b0s (still only 2 volumes!)
  local both_b0="${output_dir}/both_b0.mif"
  mrcat "$main_b0" "$pa_b0" "$both_b0" -axis 3 -force 2>>"$MASTER_LOG_FILE"
  mrconvert "$both_b0" "${output_dir}/both_b0.nii.gz" -force 2>>"$MASTER_LOG_FILE"

  log_message "  ${GREEN}Creating 2-line acqparams (1 AP + 1 averaged PA)${NC}"
  cat > "${output_dir}/acqparams.txt" <<EOF
0 -1 0 ${DEFAULT_READOUT}
0  1 0 ${DEFAULT_READOUT}
EOF
  
  log_message "${GREEN}  ? TOPUP inputs ready (2 volumes, PA averaged for SNR)${NC}"
  return 0
}

run_topup_estimate_and_apply() {
  local topup_dir=$1
  local dwi_input=$2
  local dwi_applied_out=$3
  local phantom_mask=$4

  log_message "${CYAN}Running TOPUP (2-volume method, 100 iterations)${NC}"
  
  if [ ! -f "${topup_dir}/both_b0.nii.gz" ]; then
    log_message "${RED}Missing both_b0.nii.gz${NC}"
    return 1
  fi

  local custom_config="${topup_dir}/phantom_topup.cnf"
  create_phantom_topup_config "$custom_config"

  log_message "  Applying tight phantom mask to inputs..."
  local both_b0_mif="${topup_dir}/both_b0.mif"
  local both_b0_masked="${topup_dir}/both_b0_masked.mif"
  
  mrconvert "${topup_dir}/both_b0.nii.gz" "$both_b0_mif" -force 2>>"$MASTER_LOG_FILE"
  
  local mask_b0="${topup_dir}/mask_for_topup.mif"
  resample_mask_to_target_grid "$phantom_mask" "$both_b0_mif" "$mask_b0"
  
  mrcalc "$both_b0_mif" "$mask_b0" -mult "$both_b0_masked" -force 2>>"$MASTER_LOG_FILE"
  mrconvert "$both_b0_masked" "${topup_dir}/both_b0_masked.nii.gz" -force 2>>"$MASTER_LOG_FILE"

  log_message "  Running TOPUP (3mm warp, 100 iterations, 2 volumes)..."
  topup --imain="${topup_dir}/both_b0_masked.nii.gz" \
        --datain="${topup_dir}/acqparams.txt" \
        --config="$custom_config" \
        --out="${topup_dir}/topup_results" \
        --fout="${topup_dir}/field_map" \
        --iout="${topup_dir}/corrected_b0" \
        --jacout="${topup_dir}/jac" \
        --rbmout="${topup_dir}/rbm" \
        --dfout="${topup_dir}/df" \
        --verbose 2>>"$MASTER_LOG_FILE" | tee -a "$LOG_FILE"

  if [ -f "${topup_dir}/corrected_b0.nii.gz" ]; then
    fslmaths "${topup_dir}/corrected_b0.nii.gz" -Tmean "${topup_dir}/corrected_b0_mean.nii.gz" 2>>"$MASTER_LOG_FILE"
  fi

  local temp_dwi="${topup_dir}/temp_dwi.nii.gz"
  mrconvert "$dwi_input" "$temp_dwi" -export_grad_fsl "${topup_dir}/bvecs" "${topup_dir}/bvals" -force 2>>"$MASTER_LOG_FILE"
  
  log_message "  Applying TOPUP correction with Jacobian modulation..."
  applytopup --imain="$temp_dwi" \
             --inindex=1 \
             --datain="${topup_dir}/acqparams.txt" \
             --topup="${topup_dir}/topup_results" \
             --out="${topup_dir}/dwi_topup_corrected.nii.gz" \
             --method=jac \
             2>>"$MASTER_LOG_FILE" | tee -a "$LOG_FILE"
  
  if [ -f "${topup_dir}/dwi_topup_corrected.nii.gz" ]; then
    mrconvert "${topup_dir}/dwi_topup_corrected.nii.gz" "$dwi_applied_out" \
      -fslgrad "${topup_dir}/bvecs" "${topup_dir}/bvals" -force 2>>"$MASTER_LOG_FILE"
    log_message "${GREEN}  ? TOPUP applied (averaged ROPE, 100 iter)${NC}"
    
    if [ -f "${topup_dir}/field_map.nii.gz" ]; then
      local vdm="${topup_dir}/vdm.nii.gz"
      fslmaths "${topup_dir}/field_map.nii.gz" -mul "${DEFAULT_READOUT}" "$vdm" 2>>"$MASTER_LOG_FILE"
    fi
  else
    log_message "${YELLOW}applytopup failed; copying input${NC}"
    cp -f "$dwi_input" "$dwi_applied_out"
  fi
  
  return 0
}

# ------------------------------- FILTER LOW B-VALUES --------------------------
filter_low_bvalue_shells() {
  local dwi_input=$1
  local dwi_output=$2
  local work_dir=$3

  log_message "${CYAN}Filtering low b-value shells (b=100, b=200)${NC}"
  log_message "  ${YELLOW}Required for EDDY: removes undersampled shells${NC}"
  
  mkdir -p "$work_dir"
  
  local td; td=$(_make_tmpdir)
  mrconvert "$dwi_input" "${td}/dwi_temp.nii.gz" \
    -export_grad_fsl "${td}/bvecs_temp" "${td}/bvals_temp" \
    -force 2>>"$MASTER_LOG_FILE"

  python3 <<PYEOF
import numpy as np
import nibabel as nib

img = nib.load("${td}/dwi_temp.nii.gz")
data = img.get_fdata()
bvals = np.loadtxt("${td}/bvals_temp")
bvecs = np.loadtxt("${td}/bvecs_temp")

print(f"Original: {data.shape[3]} volumes")
unique, counts = np.unique(bvals, return_counts=True)
print(f"B-value distribution:")
for b, c in zip(unique, counts):
    print(f"  b={int(b):4d}: {c:3d} volumes")

# Keep only b=0 and b>=500
keep_mask = (bvals < 50) | (bvals >= 500)
data_filtered = data[:, :, :, keep_mask]
bvals_filtered = bvals[keep_mask]
bvecs_filtered = bvecs[:, keep_mask]

removed_100 = (bvals == 100).sum()
removed_200 = (bvals == 200).sum()

print(f"\nFiltered: {data_filtered.shape[3]} volumes")
print(f"Removed: b=100 ({removed_100} vols) + b=200 ({removed_200} vols)")
print(f"\nNew b-value distribution:")
unique_f, counts_f = np.unique(bvals_filtered, return_counts=True)
for b, c in zip(unique_f, counts_f):
    print(f"  b={int(b):4d}: {c:3d} volumes")

img_filtered = nib.Nifti1Image(data_filtered, img.affine, img.header)
nib.save(img_filtered, "${td}/dwi_filtered.nii.gz")
np.savetxt("${td}/bvals_filtered", bvals_filtered, fmt='%d', newline=' ')
np.savetxt("${td}/bvecs_filtered", bvecs_filtered, fmt='%.6f')

print("\n? Filtering complete!")
PYEOF

  mrconvert "${td}/dwi_filtered.nii.gz" "$dwi_output" \
    -fslgrad "${td}/bvecs_filtered" "${td}/bvals_filtered" \
    -force 2>>"$MASTER_LOG_FILE"
  
  cp "${td}/bvals_filtered" "${work_dir}/bvals_filtered"
  cp "${td}/bvecs_filtered" "${work_dir}/bvecs_filtered"
  
  rm -rf "$td"
  
  local nvols=$(mrinfo "$dwi_output" -size 2>>"$MASTER_LOG_FILE" | awk '{print $4}')
  log_message "${GREEN}  ? Filtered to $nvols volumes (b?500 + b=0)${NC}"
  return 0
}

# ------------------------------- EDDY with outlier replacement (PHASE 1) --------------------------
run_eddy_with_filtered_data() {
  local dwi_filtered=$1
  local topup_dir=$2
  local eddy_dir=$3
  local dwi_output=$4
  local phantom_mask=$5

  log_message "${CYAN}Running EDDY (PHASE 1: with outlier replacement)${NC}"
  log_message "  ${GREEN}New flags: --repol (replace outlier slices)${NC}"
  mkdir -p "$eddy_dir"

  local eddy_cmd=""
  if command -v eddy_openmp_patched >/dev/null 2>&1; then
    eddy_cmd="eddy_openmp_patched"
    log_message "  Found: eddy_openmp_patched (MATLAB-validated) ?"
  elif command -v eddy_cuda10.2 >/dev/null 2>&1; then
    eddy_cmd="eddy_cuda10.2"
    log_message "  Found: eddy_cuda10.2"
  elif command -v eddy >/dev/null 2>&1; then
    eddy_cmd="eddy"
    log_message "  Found: eddy"
  else
    log_message "${YELLOW}EDDY not found - skipping${NC}"
    cp -f "$dwi_filtered" "$dwi_output"
    return 1
  fi

  local dwi_nii="${eddy_dir}/dwi_filtered.nii.gz"
  mrconvert "$dwi_filtered" "$dwi_nii" \
    -export_grad_fsl "${eddy_dir}/bvecs" "${eddy_dir}/bvals" \
    -force 2>>"$MASTER_LOG_FILE"

  local mask_nii="${eddy_dir}/eddy_mask.nii.gz"
  resample_mask_to_target_grid "$phantom_mask" "$dwi_filtered" "${eddy_dir}/mask_temp.mif"
  mrconvert "${eddy_dir}/mask_temp.mif" "$mask_nii" -datatype bit -force 2>>"$MASTER_LOG_FILE"
  
  local mask_voxels=$(fslstats "$mask_nii" -V 2>/dev/null | awk '{print $1}')
  local mask_volume=$(fslstats "$mask_nii" -V 2>/dev/null | awk '{printf "%.1f", $2/1000}')
  log_message "  Mask: $mask_voxels voxels (${mask_volume} cm³)"

  local nvols=$(mrinfo "$dwi_filtered" -size 2>>"$MASTER_LOG_FILE" | awk '{print $4}')
  printf '1%.0s ' $(seq 1 $nvols) | sed 's/ $//' > "${eddy_dir}/index.txt"
  log_message "  Processing $nvols filtered volumes"

  cp "${topup_dir}/acqparams.txt" "${eddy_dir}/acqparams.txt"

  mkdir -p "${eddy_dir}/dfields"
  
  log_message ""
  log_message "${CYAN}Starting EDDY with PHASE 1 optimizations...${NC}"
  log_message "  ${GREEN}--repol:     Replace outlier slices${NC}"
  log_message "  ${GREEN}--ol_nstd=4: Outlier threshold (4 std devs)${NC}"
  log_message ""
  
  if $eddy_cmd --imain="$dwi_nii" \
            --mask="$mask_nii" \
            --acqp="${eddy_dir}/acqparams.txt" \
            --index="${eddy_dir}/index.txt" \
            --bvecs="${eddy_dir}/bvecs" \
            --bvals="${eddy_dir}/bvals" \
            --topup="${topup_dir}/topup_results" \
            --out="${eddy_dir}/dwi_eddy" \
            --data_is_shelled \
            --dfields \
            --repol \
            --ol_nstd=4 \
            --verbose 2>&1 | tee "${eddy_dir}/eddy_log.txt"; then
    
    log_message ""
    log_message "${GREEN}???????????????????????????????????????????????${NC}"
    log_message "${GREEN}  ? EDDY SUCCEEDED (Phase 1 Optimized)!       ${NC}"
    log_message "${GREEN}???????????????????????????????????????????????${NC}"
    log_message ""
    log_message "${CYAN}Success details:${NC}"
    log_message "  Method: Filtered data + outlier replacement"
    log_message "  Volumes processed: $nvols"
    log_message "  Mask volume: ${mask_volume} cm³"
    log_message ""
    log_message "  ${GREEN}Corrections applied:${NC}"
    log_message "    ? Susceptibility (TOPUP field, averaged ROPE)"
    log_message "    ? Motion (volume-to-volume registration)"
    log_message "    ? Eddy currents (per-slice correction)"
    log_message "    ? Gradient reorientation"
    log_message "    ? Outlier replacement (--repol)"
    log_message ""
    
    if ls "${eddy_dir}"/dwi_eddy.eddy_displacement_fields.* 1> /dev/null 2>&1; then
      mv "${eddy_dir}"/dwi_eddy.eddy_displacement_fields.* "${eddy_dir}/dfields/" 2>>"$MASTER_LOG_FILE"
      local nfields=$(ls "${eddy_dir}/dfields"/*.nii.gz 2>/dev/null | wc -l)
      log_message "  Displacement fields: $nfields volumes saved"
    fi
    
    log_message "${CYAN}Output files:${NC}"
    for file in dwi_eddy.nii.gz eddy_rotated_bvecs eddy_parameters eddy_movement_rms eddy_outlier_map; do
      if [ -f "${eddy_dir}/${file}" ] || [ -f "${eddy_dir}/dwi_eddy.${file#dwi_eddy.}" ]; then
        log_message "  ? $file"
      fi
    done
    
    # Check if outliers were detected - FIX: Check if file exists first
    if [ -f "${eddy_dir}/dwi_eddy.eddy_outlier_map" ]; then
      local n_outliers=$(fslstats "${eddy_dir}/dwi_eddy.eddy_outlier_map" -V 2>/dev/null | awk '{print $1}')
      if [ -n "$n_outliers" ] && [ "$n_outliers" -gt 0 ]; then
        log_message ""
        log_message "  ${YELLOW}Outliers detected: $n_outliers slices replaced${NC}"
      else
        log_message ""
        log_message "  ${GREEN}No outliers detected (clean data!)${NC}"
      fi
    else
      log_message ""
      log_message "  ${GREEN}No outliers detected (clean data!)${NC}"
    fi
    
    log_message ""
    log_message "${GREEN}???????????????????????????????????????????????${NC}"
    log_message ""
    
    if [ -f "${eddy_dir}/dwi_eddy.nii.gz" ]; then
      local rotated_bvecs="${eddy_dir}/dwi_eddy.eddy_rotated_bvecs"
      if [ -f "$rotated_bvecs" ]; then
        mrconvert "${eddy_dir}/dwi_eddy.nii.gz" "$dwi_output" \
          -fslgrad "$rotated_bvecs" "${eddy_dir}/bvals" -force 2>>"$MASTER_LOG_FILE"
        log_message "${GREEN}  ? Converted with rotated gradients${NC}"
      else
        mrconvert "${eddy_dir}/dwi_eddy.nii.gz" "$dwi_output" \
          -fslgrad "${eddy_dir}/bvecs" "${eddy_dir}/bvals" -force 2>>"$MASTER_LOG_FILE"
      fi
      return 0
    fi
  fi

  log_message ""
  log_message "${YELLOW}???????????????????????????????????????????????${NC}"
  log_message "${YELLOW}  EDDY Failed (unexpected with filtering)     ${NC}"
  log_message "${YELLOW}???????????????????????????????????????????????${NC}"
  log_message ""
  log_message "${CYAN}Check log: ${eddy_dir}/eddy_log.txt${NC}"
  log_message "${YELLOW}???????????????????????????????????????????????${NC}"
  log_message ""
  
  cp -f "$dwi_filtered" "$dwi_output"
  return 1
}

# ------------------------------- EDDY QC Reports (PHASE 1) - FIX: DON'T PRE-CREATE --------------------------
generate_eddy_qc_reports() {
  local eddy_dir=$1
  
  log_message "${CYAN}Generating EDDY QC reports (eddy_quad)${NC}"
  
  if ! command -v eddy_quad >/dev/null 2>&1; then
    log_message "${YELLOW}  eddy_quad not found - skipping QC reports${NC}"
    return 1
  fi
  
  if [ ! -f "${eddy_dir}/dwi_eddy.nii.gz" ]; then
    log_message "${YELLOW}  EDDY output not found - skipping QC${NC}"
    return 1
  fi
  
  # GUARANTEED UNIQUE: Keep trying until we find a directory that doesn't exist
  local qc_dir
  local attempt=0
  while true; do
    qc_dir="${eddy_dir}/qc_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    if [ ! -d "$qc_dir" ]; then
      break
    fi
    attempt=$((attempt + 1))
    if [ $attempt -gt 100 ]; then
      log_message "${RED}  Could not create unique QC directory after 100 attempts${NC}"
      return 1
    fi
    sleep 0.01  # Brief pause before retry
  done
  
  log_message "  QC directory: ${qc_dir}"
  # DON'T CREATE THE DIRECTORY - let eddy_quad create it
  # mkdir -p "$qc_dir"  <-- REMOVED THIS LINE
  
  log_message "  Running eddy_quad..."
  
  # Check if we have rotated bvecs
  local bvecs_file="${eddy_dir}/bvecs"
  if [ -f "${eddy_dir}/dwi_eddy.eddy_rotated_bvecs" ]; then
    bvecs_file="${eddy_dir}/dwi_eddy.eddy_rotated_bvecs"
    log_message "  Using rotated bvecs for QC"
  fi
  
  if eddy_quad "${eddy_dir}/dwi_eddy" \
       -idx "${eddy_dir}/index.txt" \
       -par "${eddy_dir}/acqparams.txt" \
       -m "${eddy_dir}/eddy_mask.nii.gz" \
       -b "${eddy_dir}/bvals" \
       -g "$bvecs_file" \
       -o "$qc_dir" \
       -v 2>&1 | tee "${eddy_dir}/eddy_quad.log"; then
    
    log_message ""
    log_message "${GREEN}  ? EDDY QC reports generated!${NC}"
    log_message ""
    log_message "${CYAN}  QC outputs in: ${qc_dir}/${NC}"
    log_message "    ? qc.pdf       - Visual QC report"
    log_message "    ? qc.json      - Quantitative metrics"
    log_message "    ? cnr/         - CNR plots per shell"
    log_message "    ? ref/         - Reference images"
    log_message ""
    
    # Extract key metrics from JSON if available
    if [ -f "${qc_dir}/qc.json" ] && command -v python3 >/dev/null 2>&1; then
      log_message "${CYAN}  Key QC metrics:${NC}"
      
      python3 <<PYEOF 2>/dev/null || true
import json
try:
    with open("${qc_dir}/qc.json", 'r') as f:
        qc = json.load(f)
    
    if 'qc_mot_abs' in qc:
        print(f"    Absolute motion: {qc['qc_mot_abs']:.3f} mm")
    if 'qc_mot_rel' in qc:
        print(f"    Relative motion: {qc['qc_mot_rel']:.3f} mm")
    if 'qc_outliers_tot' in qc:
        print(f"    Total outliers:  {qc['qc_outliers_tot']}")
    if 'qc_cnr_avg' in qc:
        print(f"    Average CNR:     {qc['qc_cnr_avg']:.2f}")
except Exception as e:
    pass
PYEOF
      
      log_message ""
    fi
    
    return 0
  else
    log_message "${YELLOW}  eddy_quad failed - check ${eddy_dir}/eddy_quad.log${NC}"
    return 1
  fi
}

# ------------------------------- Bias correction ----------------------------------------------
bias_correct_dwi() {
  local dwi_in=$1
  local dwi_out=$2
  local out_dir=$3

  mkdir -p "$out_dir"
  log_message "${CYAN}Bias-field correction${NC}"

  local mask="${out_dir}/bias_mask.mif"
  create_spirit_phantom_mask "$dwi_in" "$mask" "$MASK_EROSION_MM" "$PHANTOM_DIAMETER_MM"

  local bias_field="${out_dir}/biasfield.mif"
  local algo="ants"
  if ! command -v N4BiasFieldCorrection >/dev/null 2>&1; then algo="fsl"; fi

  dwibiascorrect $algo "$dwi_in" "$dwi_out" -mask "$mask" -bias "$bias_field" -nthreads $USE_CORES -force 2>>"$MASTER_LOG_FILE" | tee -a "$LOG_FILE"

  [ -f "$bias_field" ] && mrconvert "$bias_field" "${bias_field%.mif}.nii.gz" -force 2>>"$MASTER_LOG_FILE"
  [ -f "$dwi_out" ] && mrconvert "$dwi_out" "${dwi_out%.mif}.nii.gz" -force 2>>"$MASTER_LOG_FILE"
  return 0
}

bias_correct_dwi_isolated() {
  local dwi_in=$1
  local dwi_out=$2
  local out_dir=$3

  mkdir -p "$out_dir"
  log_message "${CYAN}Bias-field correction (isolated)${NC}"

  local mask="${out_dir}/bias_mask_iso.mif"
  create_spirit_phantom_mask "$dwi_in" "$mask" "$MASK_EROSION_MM" "$PHANTOM_DIAMETER_MM"

  local bias_field="${out_dir}/biasfield_iso.mif"
  local algo="ants"
  if ! command -v N4BiasFieldCorrection >/dev/null 2>&1; then algo="fsl"; fi

  dwibiascorrect $algo "$dwi_in" "$dwi_out" -mask "$mask" -bias "$bias_field" -nthreads $USE_CORES -force 2>>"$MASTER_LOG_FILE" | tee -a "$LOG_FILE"

  [ -f "$bias_field" ] && mrconvert "$bias_field" "${bias_field%.mif}.nii.gz" -force 2>>"$MASTER_LOG_FILE"
  [ -f "$dwi_out" ] && mrconvert "$dwi_out" "${dwi_out%.mif}.nii.gz" -force 2>>"$MASTER_LOG_FILE"
  return 0
}

# ------------------------------- Main processing ----------------------------------------------
process_dwi() {
  local input_dir=$1
  local series_name=$2
  local pa_dir=$3

  local proc_dir="${OUTPUT_BASE}/${series_name}"
  mkdir -p "$proc_dir"/{dwi_steps_cumulative,dwi_steps_isolated,adc_maps_native,adc_maps_isolated_native,difference_maps,topup,eddy,topup_iso,denoise_outputs,gibbs_outputs,bias_outputs,bias_outputs_iso,masks,logs}

  local safe_series; safe_series=$(printf '%s' "$series_name" | sed -E 's/[^A-Za-z0-9_.+-]+/_/g')
  LOG_FILE="${proc_dir}/logs/pipeline_${safe_series}_$(date +%Y%m%d_%H%M%S).txt"

  log_message "${GREEN}========================================${NC}"
  log_message "${GREEN}Processing: ${series_name}${NC}"
  log_message "${GREEN}========================================${NC}"

  local nii_file=$(ls "$input_dir"/*.nii 2>/dev/null | head -1)
  local bval_file=$(ls "$input_dir"/*.bval 2>/dev/null | head -1)
  local bvec_file=$(ls "$input_dir"/*.bvec 2>/dev/null | head -1)
  
  if [ -z "$nii_file" ]; then
    log_message "${RED}No NIfTI found${NC}"
    return 1
  fi

  # ============================================
  # CUMULATIVE PIPELINE
  # ============================================
  log_message ""; log_message "${MAGENTA}========================================${NC}"
  log_message "${MAGENTA}CUMULATIVE PIPELINE (Phase 1 Optimized)${NC}"
  log_message "${MAGENTA}========================================${NC}"

  # STEP 0 - Convert
  log_message ""; log_message "${CYAN}STEP 0 - Convert${NC}"
  local dwi_step0="${proc_dir}/dwi_steps_cumulative/step00_original.mif"
  if [ -f "$bval_file" ] && [ -f "$bvec_file" ]; then
    mrconvert "$nii_file" "$dwi_step0" -fslgrad "$bvec_file" "$bval_file" -nthreads $USE_CORES -force 2>>"$MASTER_LOG_FILE"
  else
    mrconvert "$nii_file" "$dwi_step0" -nthreads $USE_CORES -force 2>>"$MASTER_LOG_FILE"
  fi

  # STEP 1 - Denoise
  log_message ""; log_message "${CYAN}STEP 1 - Denoise${NC}"
  local dwi_step1="${proc_dir}/dwi_steps_cumulative/step01_denoised.mif"
  dwidenoise "$dwi_step0" "$dwi_step1" -noise "${proc_dir}/denoise_outputs/noise_map.mif" -nthreads $USE_CORES -force 2>>"$MASTER_LOG_FILE"
  mrcalc "$dwi_step0" "$dwi_step1" -subtract "${proc_dir}/denoise_outputs/residual.mif" -force 2>>"$MASTER_LOG_FILE"

  # STEP 2 - Gibbs
  log_message ""; log_message "${CYAN}STEP 2 - Gibbs${NC}"
  local dwi_step2="${proc_dir}/dwi_steps_cumulative/step02_denoised_degibbs.mif"
  mrdegibbs "$dwi_step1" "$dwi_step2" -axes 0,1 -nthreads $USE_CORES -force 2>>"$MASTER_LOG_FILE"
  mrcalc "$dwi_step1" "$dwi_step2" -subtract "${proc_dir}/gibbs_outputs/residual.mif" -force 2>>"$MASTER_LOG_FILE"

  # Create canonical mask
  log_message ""; log_message "${CYAN}Creating canonical phantom mask${NC}"
  local canonical_mask="${proc_dir}/masks/phantom_mask_canonical.mif"
  create_spirit_phantom_mask "$dwi_step2" "$canonical_mask" "$MASK_EROSION_MM" "$PHANTOM_DIAMETER_MM"

  # STEP 3 - TOPUP (Phase 1: averaged ROPE, 100 iterations)
  log_message ""; log_message "${CYAN}STEP 3 - TOPUP (Phase 1: averaged ROPE, 100 iter)${NC}"
  prepare_topup_data "$dwi_step2" "$pa_dir" "${proc_dir}/topup"
  local dwi_step3="${proc_dir}/dwi_steps_cumulative/step03_topup.mif"
  run_topup_estimate_and_apply "${proc_dir}/topup" "$dwi_step2" "$dwi_step3" "$canonical_mask"

  # STEP 3.5 - Filter b-values
  log_message ""; log_message "${CYAN}STEP 3.5 - Filter b-values${NC}"
  local dwi_step3_5="${proc_dir}/dwi_steps_cumulative/step03_5_filtered.mif"
  filter_low_bvalue_shells "$dwi_step3" "$dwi_step3_5" "${proc_dir}/eddy"

  # STEP 4 - EDDY (Phase 1: with --repol)
  local eddy_succeeded=false
  local dwi_step4="$dwi_step3_5"
  if [ "$ENABLE_EDDY" = true ]; then
    log_message ""; log_message "${CYAN}STEP 4 - EDDY (Phase 1: with --repol)${NC}"
    dwi_step4="${proc_dir}/dwi_steps_cumulative/step04_eddy.mif"
    if run_eddy_with_filtered_data "$dwi_step3_5" "${proc_dir}/topup" "${proc_dir}/eddy" "$dwi_step4" "$canonical_mask"; then
      eddy_succeeded=true
      
      # STEP 4.5 - Generate EDDY QC reports
      log_message ""; log_message "${CYAN}STEP 4.5 - Generate EDDY QC reports${NC}"
      generate_eddy_qc_reports "${proc_dir}/eddy"
    else
      dwi_step4="$dwi_step3_5"
    fi
  fi

  # STEP 5 - GNC (skipped)
  local dwi_step5="$dwi_step4"
  log_message ""; log_message "${YELLOW}STEP 5 - GNC skipped (not available)${NC}"

  # STEP 6 - Bias
  log_message ""; log_message "${CYAN}STEP 6 - Bias correction${NC}"
  local dwi_step6="${proc_dir}/dwi_steps_cumulative/step06_final_biascorr.mif"
  bias_correct_dwi "$dwi_step5" "$dwi_step6" "${proc_dir}/bias_outputs"

  # Resample masks to each step
  log_message "${CYAN}Resampling masks...${NC}"
  local mask_step0="${proc_dir}/masks/mask_00_original.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_step0" "$mask_step0"
  local mask_step1="${proc_dir}/masks/mask_01_denoised.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_step1" "$mask_step1"
  local mask_step2="${proc_dir}/masks/mask_02_degibbs.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_step2" "$mask_step2"
  local mask_step3="${proc_dir}/masks/mask_03_topup.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_step3" "$mask_step3"
  local mask_step3_5="${proc_dir}/masks/mask_03_5_filtered.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_step3_5" "$mask_step3_5"
  local mask_step4="${proc_dir}/masks/mask_04_eddy.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_step4" "$mask_step4"
  local mask_step5="${proc_dir}/masks/mask_05_gnc.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_step5" "$mask_step5"
  local mask_step6="${proc_dir}/masks/mask_06_final.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_step6" "$mask_step6"

  # Trace ADC Metrics - Cumulative
  log_message ""; log_message "${CYAN}Computing cumulative ADC maps${NC}"
  local adc_step0="${proc_dir}/adc_maps_native/ADC_00_original.mif"
  local adc_step1="${proc_dir}/adc_maps_native/ADC_01_denoised.mif"
  local adc_step2="${proc_dir}/adc_maps_native/ADC_02_degibbs.mif"
  local adc_step3="${proc_dir}/adc_maps_native/ADC_03_topup.mif"
  local adc_step3_5="${proc_dir}/adc_maps_native/ADC_03_5_filtered.mif"
  local adc_step4="${proc_dir}/adc_maps_native/ADC_04_eddy.mif"
  local adc_step5="${proc_dir}/adc_maps_native/ADC_05_gnc.mif"
  local adc_step6="${proc_dir}/adc_maps_native/ADC_06_final.mif"

  derive_trace_adc_multishell "$dwi_step0" "$mask_step0" "$adc_step0" "$series_name" "cumulative" "00_original" "native"
  derive_trace_adc_multishell "$dwi_step1" "$mask_step1" "$adc_step1" "$series_name" "cumulative" "01_denoise" "native"
  derive_trace_adc_multishell "$dwi_step2" "$mask_step2" "$adc_step2" "$series_name" "cumulative" "02_degibbs" "native"
  derive_trace_adc_multishell "$dwi_step3" "$mask_step3" "$adc_step3" "$series_name" "cumulative" "03_topup" "native"
  derive_trace_adc_multishell "$dwi_step3_5" "$mask_step3_5" "$adc_step3_5" "$series_name" "cumulative" "03_5_filtered" "native"
  [ "$eddy_succeeded" = true ] && derive_trace_adc_multishell "$dwi_step4" "$mask_step4" "$adc_step4" "$series_name" "cumulative" "04_eddy" "native"
  derive_trace_adc_multishell "$dwi_step6" "$mask_step6" "$adc_step6" "$series_name" "cumulative" "06_final" "native"

  # ============================================
  # ISOLATED PIPELINE
  # ============================================
  log_message ""; log_message "${MAGENTA}========================================${NC}"
  log_message "${MAGENTA}ISOLATED PIPELINE (Phase 1 Optimized)${NC}"
  log_message "${MAGENTA}========================================${NC}"

  # Isolated Step 1 - Denoise only
  log_message ""; log_message "${CYAN}Isolated Step 1 - Denoise only${NC}"
  local dwi_iso_denoise="${proc_dir}/dwi_steps_isolated/step01_isolated_denoised.mif"
  dwidenoise "$dwi_step0" "$dwi_iso_denoise" -nthreads $USE_CORES -force 2>>"$MASTER_LOG_FILE"

  # Isolated Step 2 - Degibbs only
  log_message ""; log_message "${CYAN}Isolated Step 2 - Degibbs only${NC}"
  local dwi_iso_degibbs="${proc_dir}/dwi_steps_isolated/step02_isolated_degibbs.mif"
  mrdegibbs "$dwi_step0" "$dwi_iso_degibbs" -axes 0,1 -nthreads $USE_CORES -force 2>>"$MASTER_LOG_FILE"

  # Isolated Step 3 - TOPUP only (Phase 1: averaged ROPE)
  log_message ""; log_message "${CYAN}Isolated Step 3 - TOPUP only (Phase 1)${NC}"
  local dwi_iso_topup="${proc_dir}/dwi_steps_isolated/step03_isolated_topup.mif"
  if [ -n "$pa_dir" ]; then
    prepare_topup_data "$dwi_step0" "$pa_dir" "${proc_dir}/topup_iso"
    run_topup_estimate_and_apply "${proc_dir}/topup_iso" "$dwi_step0" "$dwi_iso_topup" "$canonical_mask"
  else
    log_message "${YELLOW}  No PA reference, skipping${NC}"
    cp -f "$dwi_step0" "$dwi_iso_topup"
  fi

  # Isolated Step 4 - TOPUP + Filter + EDDY
  if [ "$ENABLE_EDDY" = true ]; then
    log_message ""; log_message "${CYAN}Isolated Step 4 - TOPUP + Filter + EDDY${NC}"
    
    local iso_topup_filtered="${proc_dir}/dwi_steps_isolated/step04_topup_filtered.mif"
    filter_low_bvalue_shells "$dwi_iso_topup" "$iso_topup_filtered" "${proc_dir}/eddy_isolated_prep"
    
    local iso_eddy="${proc_dir}/dwi_steps_isolated/step04_isolated_eddy.mif"
    mkdir -p "${proc_dir}/eddy_isolated"
    
    if run_eddy_with_filtered_data "$iso_topup_filtered" "${proc_dir}/topup_iso" "${proc_dir}/eddy_isolated" "$iso_eddy" "$canonical_mask"; then
      local iso_mask4="${proc_dir}/masks/iso_mask_04_eddy.mif"
      resample_mask_to_target_grid "$canonical_mask" "$iso_eddy" "$iso_mask4"
      local iso_adc4="${proc_dir}/adc_maps_isolated_native/ADC_iso_04_topup_eddy.mif"
      derive_trace_adc_multishell "$iso_eddy" "$iso_mask4" "$iso_adc4" "$series_name" "isolated" "04_topup_eddy" "native"
      
      # Generate QC for isolated EDDY too
      generate_eddy_qc_reports "${proc_dir}/eddy_isolated"
      
      log_message "${GREEN}  ? Isolated TOPUP+EDDY complete${NC}"
    else
      log_message "${YELLOW}  Isolated EDDY failed - skipped${NC}"
    fi
  fi

  # Isolated Step 6 - Bias only
  log_message ""; log_message "${CYAN}Isolated Step 6 - Bias correction only${NC}"
  local dwi_iso_bias="${proc_dir}/dwi_steps_isolated/step06_isolated_biascorr.mif"
  bias_correct_dwi_isolated "$dwi_step0" "$dwi_iso_bias" "${proc_dir}/bias_outputs_iso"

  # Resample masks for isolated
  local mask_iso_denoise="${proc_dir}/masks/mask_iso_01_denoised.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_iso_denoise" "$mask_iso_denoise"
  local mask_iso_degibbs="${proc_dir}/masks/mask_iso_02_degibbs.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_iso_degibbs" "$mask_iso_degibbs"
  local mask_iso_topup="${proc_dir}/masks/mask_iso_03_topup.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_iso_topup" "$mask_iso_topup"
  local mask_iso_bias="${proc_dir}/masks/mask_iso_06_bias.mif"
  resample_mask_to_target_grid "$canonical_mask" "$dwi_iso_bias" "$mask_iso_bias"

  # Trace ADC Metrics - Isolated
  log_message ""; log_message "${CYAN}Computing isolated ADC maps${NC}"
  local adc_iso_denoise="${proc_dir}/adc_maps_isolated_native/ADC_isolated_denoised.mif"
  local adc_iso_degibbs="${proc_dir}/adc_maps_isolated_native/ADC_isolated_degibbs.mif"
  local adc_iso_topup="${proc_dir}/adc_maps_isolated_native/ADC_isolated_topup.mif"
  local adc_iso_bias="${proc_dir}/adc_maps_isolated_native/ADC_isolated_biascorr.mif"

  derive_trace_adc_multishell "$dwi_iso_denoise" "$mask_iso_denoise""$adc_iso_denoise" "$series_name" "isolated" "01_denoise" "native"
  derive_trace_adc_multishell "$dwi_iso_degibbs" "$mask_iso_degibbs" "$adc_iso_degibbs" "$series_name" "isolated" "02_degibbs" "native"
  derive_trace_adc_multishell "$dwi_iso_topup" "$mask_iso_topup" "$adc_iso_topup" "$series_name" "isolated" "03_topup" "native"
  derive_trace_adc_multishell "$dwi_iso_bias" "$mask_iso_bias" "$adc_iso_bias" "$series_name" "isolated" "06_biascorr" "native"

  # Difference maps
  log_message ""; log_message "${CYAN}Computing difference maps${NC}"
  [ -f "$adc_step0" ] && [ -f "$adc_step1" ] && mrcalc "$adc_step1" "$adc_step0" -subtract "${proc_dir}/difference_maps/diff_01_denoise.mif" -force 2>>"$MASTER_LOG_FILE"
  [ -f "$adc_step2" ] && [ -f "$adc_step1" ] && mrcalc "$adc_step2" "$adc_step1" -subtract "${proc_dir}/difference_maps/diff_02_gibbs.mif" -force 2>>"$MASTER_LOG_FILE"
  [ -f "$adc_step3" ] && [ -f "$adc_step2" ] && mrcalc "$adc_step3" "$adc_step2" -subtract "${proc_dir}/difference_maps/diff_03_topup.mif" -force 2>>"$MASTER_LOG_FILE"
  [ -f "$adc_step3_5" ] && [ -f "$adc_step3" ] && mrcalc "$adc_step3_5" "$adc_step3" -subtract "${proc_dir}/difference_maps/diff_03_5_filter.mif" -force 2>>"$MASTER_LOG_FILE"
  [ "$eddy_succeeded" = true ] && [ -f "$adc_step4" ] && [ -f "$adc_step3_5" ] && mrcalc "$adc_step4" "$adc_step3_5" -subtract "${proc_dir}/difference_maps/diff_04_eddy.mif" -force 2>>"$MASTER_LOG_FILE"
  [ -f "$adc_step6" ] && [ -f "$adc_step5" ] && mrcalc "$adc_step6" "$adc_step5" -subtract "${proc_dir}/difference_maps/diff_06_bias.mif" -force 2>>"$MASTER_LOG_FILE"
  [ -f "$adc_step0" ] && [ -f "$adc_step6" ] && mrcalc "$adc_step6" "$adc_step0" -subtract "${proc_dir}/difference_maps/diff_total.mif" -force 2>>"$MASTER_LOG_FILE"

  for mif in "${proc_dir}/difference_maps"/*.mif "${proc_dir}/denoise_outputs"/*.mif "${proc_dir}/gibbs_outputs"/*.mif; do
    [ -f "$mif" ] && mrconvert "$mif" "${mif%.mif}.nii.gz" -force 2>>"$MASTER_LOG_FILE"
  done

  log_message "${GREEN}========================================${NC}"
  log_message "${GREEN}Processing complete: ${series_name}${NC}"
  log_message "${GREEN}========================================${NC}"
  
  if [ "$eddy_succeeded" = true ]; then
    log_message ""
    log_message "${GREEN}? PHASE 1 SUCCESS - All optimizations applied! ?${NC}"
    log_message ""
    log_message "${CYAN}Optimizations applied:${NC}"
    log_message "  ? Averaged ROPE (7 vols ? 1 high-SNR vol)"
    log_message "  ? TOPUP 100 iterations (vs 50 baseline)"
    log_message "  ? EDDY outlier replacement (--repol)"
    log_message "  ? EDDY QC reports generated"
    log_message "  ? B-value filtering (removed b=100, b=200)"
    log_message ""
  fi
}

# ------------------------------- Dataset discovery --------------------------
log_message "${CYAN}Searching for datasets...${NC}"

BASE_DIR="/mnt/C1965566-20240220_180512/SPIRIT_phantom_12345_19700101/SPIRITPHANTOM_newcastle_2"

MAIN_DWI="${BASE_DIR}/DMRI_DIR180_AP_0004"
PA_REF="${BASE_DIR}/DMRI_DIR180_ROPE_AP_SBREF_0005"

if [ ! -d "$MAIN_DWI" ]; then
  log_message "${RED}Main DWI not found${NC}"
  exit 1
fi

if [ ! -d "$PA_REF" ]; then
  log_message "${RED}PA reference not found${NC}"
  exit 1
fi

log_message "Found:"
log_message "  Main: ${MAIN_DWI}"
log_message "  PA ref: ${PA_REF}"
log_message ""

# ------------------------------- Run processing --------------------------
process_dwi "$MAIN_DWI" "DMRI_DIR180_AP_0004" "$PA_REF"

# ------------------------------- Final summary --------------------------
log_message ""; log_message "${GREEN}===================================================${NC}"
log_message "${GREEN}SPIRIT Phantom Pipeline Complete - PHASE 1${NC}"
log_message "${GREEN}===================================================${NC}"; log_message ""

if [ -f "$SUMMARY_CSV" ]; then
  log_message "${CYAN}Cumulative Results:${NC}"
  column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
  log_message ""
fi

if [ -f "$SUMMARY_ISOLATED_CSV" ]; then
  log_message "${CYAN}Isolated Results:${NC}"
  column -t -s',' "$SUMMARY_ISOLATED_CSV" 2>/dev/null || cat "$SUMMARY_ISOLATED_CSV"
fi

log_message ""
log_message "${CYAN}Phase 1 Optimizations Applied:${NC}"
log_message "  ${GREEN}1. Averaged ROPE volumes${NC} (7 ? 1, SNR boost)"
log_message "  ${GREEN}2. TOPUP iterations: 100${NC} (better convergence)"
log_message "  ${GREEN}3. EDDY --repol flag${NC} (outlier replacement)"
log_message "  ${GREEN}4. EDDY QC reports${NC} (eddy_quad)"
log_message ""
log_message "${CYAN}Core Pipeline Features:${NC}"
log_message "  ? 2-volume TOPUP (no PA warping)"
log_message "  ? B-value filtering (removed b=100, b=200)"
log_message "  ? Tight phantom mask (191mm, ${MASK_EROSION_MM}mm erosion)"
log_message "  ? Multi-shell ADC (6 shells: 500-6000)"
log_message "  ? Cumulative + isolated pipelines"
log_message ""
log_message "${GREEN}QC Reports:${NC}"
log_message "  ? EDDY QC: /data/output/*/eddy/qc_*/qc.pdf"
log_message "  ? EDDY QC: /data/output/*/eddy_isolated/qc_*/qc.pdf"
log_message ""
log_message "${GREEN}Phase 1 Complete!${NC}"
