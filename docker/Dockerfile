FROM ubuntu:24.04

LABEL org.opencontainers.image.title="phantomkit" \
      org.opencontainers.image.description="MRI phantom QA processing pipeline" \
      org.opencontainers.image.authors="Arkiev D'Souza <arkiev.dsouza@sydney.edu.au>" \
      org.opencontainers.image.source="https://github.com/Australian-Imaging-Service/phantomkit"

ARG DEBIAN_FRONTEND=noninteractive
ARG ANTS_VERSION=2.5.3

# ---------- Base system packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget unzip \
        python3 python3-pip python3-venv \
        libgomp1 \
        dc bc \
        mrtrix3 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python virtual environment ----------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ---------- FSL via official conda-based installer ----------
# Downloads ~7 GB; includes eddy, topup, flirt, and related tools.
RUN wget -qO /tmp/fslinstaller.py \
        https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py \
    && python3 /tmp/fslinstaller.py -d /opt/fsl --skip_registration \
    && rm /tmp/fslinstaller.py

ENV FSLDIR=/opt/fsl \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE
ENV PATH="${FSLDIR}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${FSLDIR}/lib:${LD_LIBRARY_PATH:-}"

# ---------- ANTs pre-compiled binary ----------
RUN curl -fsSL \
        "https://github.com/ANTsX/ANTs/releases/download/v${ANTS_VERSION}/ants-${ANTS_VERSION}-ubuntu-22.04-X64-gcc.zip" \
        -o /tmp/ants.zip \
    && unzip -q /tmp/ants.zip -d /opt/ \
    && find /opt -maxdepth 3 -name "antsRegistration" 2>/dev/null \
         | head -1 | xargs dirname | xargs -I{} ln -s {} /opt/ants_bin \
    && rm /tmp/ants.zip

ENV PATH="/opt/ants_bin:${PATH}"

# ---------- dcm2niix pre-compiled binary ----------
RUN curl -fsSL \
        "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip" \
        -o /tmp/dcm2niix.zip \
    && unzip -q /tmp/dcm2niix.zip -d /tmp/dcm2niix \
    && install -m 755 /tmp/dcm2niix/dcm2niix /usr/local/bin/dcm2niix \
    && rm -rf /tmp/dcm2niix.zip /tmp/dcm2niix

# ---------- phantomkit (editable install preserves template_data lookup) ----------
WORKDIR /opt/phantomkit
COPY . .
RUN pip install --no-cache-dir -e .

# ---------- Runtime ----------
WORKDIR /data
VOLUME ["/data/input", "/data/output"]

ENTRYPOINT ["phantomkit"]
CMD ["--help"]
