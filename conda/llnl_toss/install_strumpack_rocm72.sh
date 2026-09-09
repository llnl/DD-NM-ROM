#!/usr/bin/env bash

# Build and install torch-strumpack for the Tuolumne ROCm 7.2 environment.
#
# This is a source build because the torch-strumpack release referenced by
# strumpack_rocm.txt is a ROCm 6 / CPython 3.11 wheel and cannot be used with
# the current PyTorch ROCm 7.2 environment.  It builds GKlib, METIS,
# STRUMPACK, and then the matching torch-strumpack wheel in one install root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-${SCRIPT_DIR}/third_party/strumpack-rocm72}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${JOBS:-16}"
UPDATE_SOURCES="${UPDATE_SOURCES:-0}"

# Resolve a caller-supplied relative venv path before changing into a source
# directory later in the script.  torch-strumpack's wheel script accepts the
# containing directory through PYBIN.
if [[ "${PYTHON_BIN}" == */* ]]; then
  PYTHON_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)/$(basename "${PYTHON_BIN}")"
fi

ROCM_MODULE="${ROCM_MODULE:-rocm/7.2}"
GCC_MODULE="${GCC_MODULE:-gcc/13.3.1-magic}"
NINJA_MODULE="${NINJA_MODULE:-ninja}"
LOAD_MODULES="${LOAD_MODULES:-1}"
HIP_ARCHS="${HIP_ARCHS:-gfx942}"
WHEEL_TAG="${WHEEL_TAG:-rocm7x}"
ZFP_PREFIX="${ZFP_PREFIX:-/usr/tce/packages/python/python-3.13.2}"

GKLIB_REPO="${GKLIB_REPO:-https://github.com/KarypisLab/GKlib.git}"
METIS_REPO="${METIS_REPO:-https://github.com/KarypisLab/METIS.git}"
TORCH_STRUMPACK_REPO="${TORCH_STRUMPACK_REPO:-https://github.com/sparsexlab/torch-strumpack.git}"
STRUMPACK_REPO="${STRUMPACK_REPO:-https://github.com/pghysels/STRUMPACK.git}"
STRUMPACK_REF="${STRUMPACK_REF:-v8.0.0}"

GKLIB_DIR="${GKLIB_DIR:-${INSTALL_ROOT}/GKlib}"
METIS_DIR="${METIS_DIR:-${INSTALL_ROOT}/METIS}"
TORCH_STRUMPACK_DIR="${TORCH_STRUMPACK_DIR:-${INSTALL_ROOT}/torch-strumpack}"
STRUMPACK_SRC="${STRUMPACK_SRC:-${INSTALL_ROOT}/STRUMPACK}"
GKLIB_PREFIX="${GKLIB_PREFIX:-${GKLIB_DIR}/install}"
METIS_PREFIX="${METIS_PREFIX:-${METIS_DIR}/install}"
STRUMPACK_PREFIX="${STRUMPACK_PREFIX:-${TORCH_STRUMPACK_DIR}/install/strumpack}"

usage() {
  cat <<'EOF'
Usage: bash install_strumpack_rocm72.sh

Builds a source-compatible torch-strumpack wheel for ROCm 7.2 and installs it
into PYTHON_BIN.  Run from an allocated build node, or set LOAD_MODULES=0 when
the required compiler and ROCm modules are already loaded.

Environment variables:
  PYTHON_BIN            Python interpreter to install into. Default: python3
  INSTALL_ROOT          Root for sources and dependency installs.
                        Default: ./third_party/strumpack-rocm72
  JOBS                  Parallel make jobs. Default: 16
  UPDATE_SOURCES        Fetch updates for existing Git clones (1) or use
                        the local checkout as-is (0, default).
  ROCM_MODULE           ROCm module to load. Default: rocm/7.2
  GCC_MODULE            GCC module to load. Default: gcc/13.3.1-magic
  LOAD_MODULES          Load ROCm/GCC modules (1) or use current environment (0).
  HIP_ARCHS             GPU target. Default: gfx942 (MI300A)
  NINJA_MODULE           Ninja module to load. Default: ninja
  STRUMPACK_CMAKE_GENERATOR
                        Generator for the native STRUMPACK build. Default:
                        Ninja.
  TORCH_STRUMPACK_CMAKE_GENERATOR
                        Generator for the torch-strumpack extension build.
                        Default: Ninja.
  SKIP_AUDITWHEEL        Skip Linux wheel repair. Default: 1 on this system,
                        whose available patchelf is too old for auditwheel.
  WHEEL_TAG             torch-strumpack wheel platform tag. Default: rocm7x
  ZFP_PREFIX             Prefix containing the ZFP shared library. Default:
                        /usr/tce/packages/python/python-3.13.2
  GKLIB_REPO            GKlib Git URL.
  METIS_REPO            METIS Git URL.
  TORCH_STRUMPACK_REPO  torch-strumpack Git URL.
  STRUMPACK_REPO        Native STRUMPACK Git URL.
  STRUMPACK_REF         Native STRUMPACK revision. Default: v8.0.0
  STRUMPACK_SRC         Native STRUMPACK source checkout. Default:
                        $INSTALL_ROOT/STRUMPACK
  *_DIR, *_PREFIX       Override individual source or install directories.

The installed library also needs STRUMPACK and METIS library directories on
LD_LIBRARY_PATH at runtime; the command is printed on successful completion.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${LOAD_MODULES}" == "1" ]]; then
  if ! type module >/dev/null 2>&1; then
    echo "Error: environment modules are unavailable; set LOAD_MODULES=0 after loading ROCm 7.2 and GCC." >&2
    exit 1
  fi
  module load "${ROCM_MODULE}"
  module load "${GCC_MODULE}"
  module load "${NINJA_MODULE}"
fi

for command in git make cmake gfortran hipcc "${PYTHON_BIN}"; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Error: ${command} is not available on PATH." >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - <<'PY'
import torch

if torch.version.hip is None:
    raise SystemExit("Error: PYTHON_BIN must provide a ROCm-enabled PyTorch build.")
if not torch.version.hip.startswith("7.2"):
    raise SystemExit(
        "Error: expected a ROCm 7.2 PyTorch build, found " + torch.version.hip
    )
print("Using PyTorch", torch.__version__, "with ROCm", torch.version.hip)
PY

clone_or_update() {
  local repo_url="$1"
  local clone_dir="$2"
  local ref="${3:-}"

  if [[ -d "${clone_dir}/.git" ]]; then
    if [[ "${UPDATE_SOURCES}" == "1" ]]; then
      echo "Updating ${clone_dir}"
      git -C "${clone_dir}" pull --ff-only
    else
      echo "Using existing local clone ${clone_dir}"
    fi
    return
  fi
  if [[ -e "${clone_dir}" && ! -d "${clone_dir}" ]]; then
    echo "Error: ${clone_dir} exists and is not a directory." >&2
    exit 1
  fi
  if [[ -d "${clone_dir}" && -n "$(ls -A "${clone_dir}")" ]]; then
    echo "Error: ${clone_dir} exists but is neither a Git clone nor empty." >&2
    exit 1
  fi
  echo "Cloning ${repo_url} into ${clone_dir}"
  if [[ -n "${ref}" ]]; then
    git clone --depth 1 --branch "${ref}" "${repo_url}" "${clone_dir}"
  else
    git clone --depth 1 "${repo_url}" "${clone_dir}"
  fi
}

mkdir -p "${INSTALL_ROOT}"
clone_or_update "${GKLIB_REPO}" "${GKLIB_DIR}"
clone_or_update "${METIS_REPO}" "${METIS_DIR}"
clone_or_update "${TORCH_STRUMPACK_REPO}" "${TORCH_STRUMPACK_DIR}"
clone_or_update "${STRUMPACK_REPO}" "${STRUMPACK_SRC}" "${STRUMPACK_REF}"

echo "Building GKlib"
make -C "${GKLIB_DIR}" config prefix="${GKLIB_PREFIX}"
make -C "${GKLIB_DIR}" -j "${JOBS}"
make -C "${GKLIB_DIR}" install

echo "Building METIS"
make -C "${METIS_DIR}" config shared=1 gklib_path="${GKLIB_PREFIX}" prefix="${METIS_PREFIX}"
# METIS's Makefile recursively invokes make.  On the LC module stack that
# wrapper can misparse inherited MAKEFLAGS ("No rule to make target 'w'").
# Build the CMake tree directly after `make config` has generated it.
cmake --build "${METIS_DIR}/build" --parallel "${JOBS}"
cmake --install "${METIS_DIR}/build"

METIS_LIBRARY_DIR="${METIS_PREFIX}/lib64"
if [[ ! -d "${METIS_LIBRARY_DIR}" ]]; then
  METIS_LIBRARY_DIR="${METIS_PREFIX}/lib"
fi
if [[ ! -f "${METIS_LIBRARY_DIR}/libmetis.so" ]]; then
  echo "Error: METIS shared library was not found under ${METIS_PREFIX}." >&2
  exit 1
fi

export STRUMPACK_BACKEND="rocm"
export STRUMPACK_PREFIX
export STRUMPACK_SRC
export STRUMPACK_REF
export HIP_ARCHS
export CMAKE_HIP_ARCHITECTURES="${HIP_ARCHS}"
export ROCM_PATH="${ROCM_PATH:-$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)}"
export CMAKE_Fortran_COMPILER="$(command -v gfortran)"
export METIS_PREFIX
export METIS_LIBRARIES="${METIS_LIBRARY_DIR}/libmetis.so"
export METIS_INCLUDE_DIR="${METIS_PREFIX}/include"
export metis_PREFIX="${METIS_PREFIX}"
export metis_INCLUDE_DIR="${METIS_INCLUDE_DIR}"
export metis_LIBRARY_DIR="${METIS_LIBRARY_DIR}"
export metis_LIBRARIES="${METIS_LIBRARIES}"
export CMAKE_PREFIX_PATH="${ROCM_PATH};${METIS_PREFIX}${CMAKE_PREFIX_PATH:+;${CMAKE_PREFIX_PATH}}"
export STRUMPACK_CMAKE_GENERATOR="${STRUMPACK_CMAKE_GENERATOR:-Ninja}"
export TORCH_STRUMPACK_CMAKE_GENERATOR="${TORCH_STRUMPACK_CMAKE_GENERATOR:-Ninja}"
export SKIP_AUDITWHEEL="${SKIP_AUDITWHEEL:-1}"

echo "Building STRUMPACK with ROCm 7.2 for ${HIP_ARCHS}"
(
  cd "${TORCH_STRUMPACK_DIR}"
  ./scripts/build_strumpack.sh
)

STRUMPACK_DIR="${STRUMPACK_PREFIX}/lib64/cmake/STRUMPACK"
if [[ ! -d "${STRUMPACK_DIR}" ]]; then
  STRUMPACK_DIR="${STRUMPACK_PREFIX}/lib/cmake/STRUMPACK"
fi
if [[ ! -d "${STRUMPACK_DIR}" ]]; then
  echo "Error: STRUMPACK CMake package was not installed under ${STRUMPACK_PREFIX}." >&2
  exit 1
fi
export STRUMPACK_DIR
export PYBIN="$(dirname "${PYTHON_BIN}")"

echo "Building torch-strumpack ${WHEEL_TAG} wheel"
(
  cd "${TORCH_STRUMPACK_DIR}"
  ./scripts/ci_build_wheel.sh "${WHEEL_TAG}"
)

mapfile -t wheels < <(find "${TORCH_STRUMPACK_DIR}" -type f -name '*.whl' -print | sort)
if [[ "${#wheels[@]}" -eq 0 ]]; then
  echo "Error: no torch-strumpack wheel was produced." >&2
  exit 1
fi
wheel="${wheels[$((${#wheels[@]} - 1))]}"
echo "Installing ${wheel} into ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --no-deps --force-reinstall "${wheel}"

STRUMPACK_LIBRARY_DIR="${STRUMPACK_PREFIX}/lib64"
if [[ ! -d "${STRUMPACK_LIBRARY_DIR}" ]]; then
  STRUMPACK_LIBRARY_DIR="${STRUMPACK_PREFIX}/lib"
fi
ZFP_LIBRARY_DIR="${ZFP_LIBRARY_DIR:-${ZFP_PREFIX}/lib}"
if [[ ! -f "${ZFP_LIBRARY_DIR}/libzfp.so.1" ]]; then
  echo "Error: ZFP shared library was not found under ${ZFP_LIBRARY_DIR}." >&2
  exit 1
fi

cat <<EOF

torch-strumpack was installed successfully.
Before running Python jobs, make the native libraries visible:
  export LD_LIBRARY_PATH="${STRUMPACK_LIBRARY_DIR}:${METIS_LIBRARY_DIR}:${ZFP_LIBRARY_DIR}:\${LD_LIBRARY_PATH:-}"

For the DD-NM-ROM direct solver, select it with:
  export DDNMROM_FORCE_SERIAL_SOLVE=1
  export DDNMROM_FORCE_SOLVE_BACKEND=strumpack
EOF
