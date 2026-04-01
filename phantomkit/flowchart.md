flowchart TD
    INPUT[/"--input-dir  --phantom  --output-dir"/]

    INPUT --> SCAN[Scan input directory]
    SCAN --> S1
    SCAN --> S3

    subgraph S1["Stage 1 — DWI Processing (dwi_processing.py)"]
        direction TB
        DWI_IN[DWI DICOM series] --> PREPROC[dwifslpreproc / eddy]
        T1_IN[T1 DICOM] --> DCM2NIIX1[dcm2niix]
        PREPROC --> TENSOR[dwi2tensor → tensor2metric\nADC.nii.gz  FA.nii.gz]
        TENSOR --> FLIRT[flirt\nb0 → T1 co-registration]
        DCM2NIIX1 --> FLIRT
        FLIRT --> T1DWI[T1_in_DWI_space.nii.gz]
    end

    subgraph S3["Stage 3 — Native Contrast QC (phantom_processor.py)"]
        direction TB
        DCM2NIIX3[dcm2niix\nT1 + IR + TE DICOMs → NIfTI]
        DCM2NIIX3 --> WF3[PhantomSessionWf\n— see workflow detail —]
    end

    T1DWI --> S2

    subgraph S2["Stage 2 — DWI-space Phantom QC (phantom_processor.py)"]
        direction TB
        WF2[PhantomSessionWf\n— see workflow detail —]
    end

    subgraph WF["PhantomSessionWf — task dependency graph"]
        direction TB
        REG["①  antsRegistrationSyN.sh\nT1 → template  ►  0GenericAffine.mat\nInverseWarped.nii.gz"]

        REG -->|inverse_warped| SAVE["①b  mrconvert\nTemplatePhantom_ScannerSpace.nii.gz"]
        REG -->|transform_matrix| VIALS["②  antsApplyTransforms\nvial masks → subject space"]
        REG -->|transform_matrix| FWDXFM["⑤  antsApplyTransforms\nall contrasts → template space"]

        VIALS -->|vial_paths| METRICS["③  mrgrid + mrstats\nper-vial mean / median / std / min / max\n→ CSV files"]
        METRICS -->|sentinel| PLOTS["④  plot_vial_intensity\nplot_vial_ir_means_std\nplot_vial_te_means_std\n→ PNG files"]

        PLOTS -->|sentinel| CLEANUP["⑥  shutil.rmtree\ntmp  tmp_vials  tmp_vols  vial_dir/tmp"]
        FWDXFM -->|sentinel| CLEANUP
    end

    S1 -.->|"runs in parallel"| S3
    S2 -.->|"runs after Stage 1"| DONE
    S3 -.-> DONE

    DONE(["outputs/\n  {session}/metrics/\n  {session}/vial_segmentations/\n  {session}/images_template_space/\n  {session}/TemplatePhantom_ScannerSpace.nii.gz"])
