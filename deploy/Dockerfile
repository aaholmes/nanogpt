# Reproducible env for the modded-nanogpt relu2_s1 A/B on 8xH100.
#   docker build -f deploy/Dockerfile -t nanogpt-relu2s1 .
#   docker run -it --rm --gpus all -v $(pwd):/workspace nanogpt-relu2s1 bash
#
# The CE kernel JIT-compiles CUDA at runtime (torch.cuda._compile_kernel, includes from
# /usr/local/cuda/include), so we need the full CUDA toolkit -- the *devel* image, not runtime.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_ALLOC_CONF=expandable_segments:True
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /workspace

# torch 2.10 nightly (cu126) ships a TMA-capable Triton. For an OFFICIAL record attempt,
# pin the EXACT nightly from your target record's log (see README record table) so the
# baseline wallclock matches; the A/B *delta* is robust to the exact build (both arms share it).
RUN pip install --break-system-packages --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126

# Install the rest of requirements.txt but let the nightly torch above stand (don't let the
# pinned `torch==2.10` line pull a different build).
COPY requirements.txt /workspace/requirements.txt
RUN grep -ivE '^torch([=<>!~ ]|$)' /workspace/requirements.txt > /workspace/reqs.notorch.txt && \
    pip install --break-system-packages -r /workspace/reqs.notorch.txt

# Code + data are bind-mounted at run time (-v $(pwd):/workspace); nothing else to COPY.
CMD ["bash"]
