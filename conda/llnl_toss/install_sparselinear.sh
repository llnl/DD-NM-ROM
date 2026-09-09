#!/usr/bin/env bash

# Helper script to install SparseLinear package
# This is done directly from Github because the PyPi package is old
# and requires torch_scatter and torch_sparse packages to be built
# The Github version removes these dependencies

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/hyeon95y/SparseLinear}"
INSTALL_ROOT="${INSTALL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/third_party}"
CLONE_DIR="${CLONE_DIR:-${INSTALL_ROOT}/SparseLinear}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: bash install_sparselinear.sh

Environment variables:
  REPO_URL       Git URL to clone. Default: https://github.com/hyeon95y/SparseLinear
  INSTALL_ROOT   Directory where the repo is cloned. Default: ./third_party
  CLONE_DIR      Full clone directory. Default: $INSTALL_ROOT/SparseLinear
  PYTHON_BIN     Python interpreter to use for installation. Default: python3
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not available on PATH." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} is not available on PATH." >&2
  exit 1
fi

mkdir -p "${INSTALL_ROOT}"

if [[ -d "${CLONE_DIR}/.git" ]]; then
  echo "Updating existing clone in ${CLONE_DIR}"
  git -C "${CLONE_DIR}" pull --ff-only --depth 1 origin
else
  if [[ -e "${CLONE_DIR}" && ! -d "${CLONE_DIR}" ]]; then
    echo "Error: ${CLONE_DIR} exists and is not a directory." >&2
    exit 1
  fi

  if [[ -d "${CLONE_DIR}" && -n "$(ls -A "${CLONE_DIR}")" ]]; then
    echo "Error: ${CLONE_DIR} exists and is not empty." >&2
    echo "Remove it or set CLONE_DIR to an empty directory." >&2
    exit 1
  fi

  echo "Cloning ${REPO_URL} into ${CLONE_DIR}"
  git clone --depth 1 "${REPO_URL}" "${CLONE_DIR}"
fi

echo "Installing SparseLinear into ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --editable "${CLONE_DIR}"

echo "SparseLinear installed from ${CLONE_DIR}"
