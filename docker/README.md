# phantomkit Docker Image

A self-contained image for running the phantomkit MRI phantom QA pipeline, bundling all required neuroimaging tools.

## What's included

| Tool | Version | Purpose |
|------|---------|---------|
| Ubuntu | 24.04 | Base OS |
| Python | 3.12 | Runtime |
| MRtrix3 | system apt | DWI preprocessing, format conversion |
| FSL | latest (conda installer) | `eddy`, `topup`, `flirt` |
| ANTs | 2.5.3 | Registration (`antsRegistrationSyN.sh`, `antsApplyTransforms`) |
| dcm2niix | latest | DICOM → NIfTI conversion |
| phantomkit | repo source | This package |

> **Note:** The FSL layer alone is ~7 GB. Expect a final image size of ~10–12 GB and a build time of 30–60 minutes depending on network speed.

---

## Building the image

Run from the **repository root**:

```bash
docker build -f docker/Dockerfile -t phantomkit:latest .
```

To pin a specific tag:

```bash
docker build -f docker/Dockerfile -t phantomkit:0.1.8 .
```

---

## Running the container

### Full pipeline

```bash
docker run --rm \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  phantomkit:latest \
  pipeline \
    --input-dir /data/input \
    --output-dir /data/output \
    --phantom SPIRIT
```

### Plotting (compare two HTML reports)

```bash
docker run --rm \
  -v /path/to/plots:/data/plots \
  phantomkit:latest \
  plot compare-plots \
    /data/plots/T1_mapping.html \
    /data/plots/T1_mapping_reduced.html \
    -o /data/plots/comparison.html
```

### Interactive shell (debugging)

```bash
docker run --rm -it \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  --entrypoint bash \
  phantomkit:latest
```

### List available commands

```bash
docker run --rm phantomkit:latest --help
```

---

## Volume mounts

| Container path | Purpose |
|----------------|---------|
| `/data/input`  | Input data (DICOM, NIfTI, or MIF) |
| `/data/output` | Processing outputs |

Any additional paths can be mounted with `-v`.

---

## Pushing to Docker Hub

1. **Log in:**

   ```bash
   docker login
   ```

2. **Tag the image** (replace `yourusername` with your Docker Hub username):

   ```bash
   docker tag phantomkit:latest yourusername/phantomkit:latest
   docker tag phantomkit:latest yourusername/phantomkit:0.1.8
   ```

3. **Push:**

   ```bash
   docker push yourusername/phantomkit:latest
   docker push yourusername/phantomkit:0.1.8
   ```

4. **Pull on another machine:**

   ```bash
   docker pull yourusername/phantomkit:latest
   ```

### Optional: multi-platform build for sharing (Apple Silicon + x86)

If you build on an Apple Silicon Mac and want the image to run on Linux servers:

```bash
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile \
  -t yourusername/phantomkit:latest \
  --push .
```

---

## Troubleshooting

**`template_data` not found:** The image uses an editable install so that `template_data/` is resolved relative to `/opt/phantomkit/`. If you override `WORKDIR` or move files, this may break.

**ANTs not found on PATH:** Confirm the symlink exists: `docker run --rm --entrypoint which phantomkit:latest antsRegistration`

**FSL eddy not found:** Verify `FSLDIR`: `docker run --rm --entrypoint bash phantomkit:latest -c 'echo $FSLDIR && ls $FSLDIR/bin/eddy*'`
