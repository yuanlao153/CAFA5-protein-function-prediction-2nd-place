#!/bin/bash
# Create rapids-gcn-env for postproc (cudf-based collect_ttas + step + make_submission)
# Usage: ./create-rapids-gcn-env.sh <base_path>
# Example: ./create-rapids-gcn-env.sh .

conda --version

# RAPIDS 26.06 (cudf 26.06.00) with Python 3.12, CUDA 12
# This handles the 64M-row TSV merges that GPU-only
conda create --solver=libmamba -p $1/rapids-gcn-env -c rapidsai -c conda-forge -c nvidia \
    cudf=26.06 python=3.12 cuda-version=12 -y

conda activate $1/rapids-gcn-env
which python

# Additional deps for postproc scripts
pip install cupy-cuda12x pyyaml joblib tqdm
