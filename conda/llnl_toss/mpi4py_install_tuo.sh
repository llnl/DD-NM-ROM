#!/usr/bin/env bash
set -euo pipefail

# Build mpi4py against the Cray MPICH installation used by the Tuolumne
# benchmarks, including the HSA GPU Transfer Layer (GTL).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO="$SCRIPT_DIR"
PYTHON="$REPO/.venv/bin/python"

: "${ROCM_VERSION:=7.2.1}"
: "${MPICH_VERSION:=9.1.0}"
: "${MPI4PY_VERSION:=4.1.1}"

export ROCM_VERSION MPICH_VERSION

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing repository Python executable: $PYTHON" >&2
  exit 1
fi

if ! declare -F module >/dev/null 2>&1; then
  echo "The environment-module command is unavailable; run this from a" >&2
  echo "Tuolumne login/allocation shell with the Cray modules initialized." >&2
  exit 1
fi

cd "$REPO"
source "$REPO/benchmarks/env/tuo.bash"

# cray-libsci/25.09.0 provides GNU 12.2 libraries on this system.  The
# benchmark environment initially selects GNU 11.2, which makes Cray's cc
# reject the mpi4py link test before the extension can be built.
module swap gcc-native/11.2 gcc-native/12.2
hash -r

CRAY_CC=/opt/cray/pe/craype/2.7.35/bin/cc
GTL_LIB="/opt/cray/pe/mpich/${MPICH_VERSION}/gtl/lib"
GTL_SO="$GTL_LIB/libmpi_gtl_hsa.so"

if [[ ! -x "$CRAY_CC" ]]; then
  echo "Missing Cray compiler wrapper: $CRAY_CC" >&2
  exit 1
fi
if [[ ! -r "$GTL_SO" ]]; then
  echo "Missing MPICH HSA GTL library: $GTL_SO" >&2
  exit 1
fi

# The Python interpreter comes from Anaconda and its sysconfig normally adds
# '-B .../compiler_compat'.  That linker cannot resolve Cray MPICH's system
# dependencies.  Override both compiler variables so the Cray wrapper is
# used without the Anaconda compiler-compatibility linker.
export MPICC="$CRAY_CC"
export CC="$CRAY_CC"
export LDSHARED="$CRAY_CC -shared"

# These are runtime settings, not build settings.  In particular,
# MPICH_GPU_SUPPORT_ENABLED=1 can initialize Cray MPICH/GTL when mpi4py is
# imported and interfere with PyTorch HIP initialization.  The benchmark
# launcher may set the runtime flag, but the benchmark initializes HIP before
# importing mpi4py through dd_nm_rom.backend.
unset MPICH_GPU_SUPPORT_ENABLED HSA_XNACK

# Force GTL to remain a DT_NEEDED dependency of mpi4py.MPI.  The explicit
# rpath is useful when mpi4py is imported outside a loaded module shell.
GTL_FLAGS="-L${GTL_LIB} -Wl,-rpath,${GTL_LIB} -Wl,--no-as-needed -lmpi_gtl_hsa -Wl,--as-needed"
export MPI4PY_BUILD_LDFLAGS="$GTL_FLAGS"
export LDFLAGS="$GTL_FLAGS"

echo "Python:       $PYTHON"
echo "MPICC:        $MPICC"
echo "ROCm:         $ROCM_VERSION"
echo "Cray MPICH:   $MPICH_VERSION"
echo "HSA GTL:      $GTL_SO"

"$PYTHON" -m pip install \
  --force-reinstall \
  --no-cache-dir \
  --no-binary=mpi4py \
  "mpi4py==${MPI4PY_VERSION}"

# Locate the extension without importing mpi4py.MPI (which initializes MPI
# during this installation/verification shell).
MPI_SO=$("$PYTHON" -c 'import importlib.util; print(importlib.util.find_spec("mpi4py.MPI").origin)')
if ! readelf -d "$MPI_SO" | rg -q 'libmpi_gtl_hsa'; then
  echo "mpi4py was installed without libmpi_gtl_hsa:" >&2
  readelf -d "$MPI_SO" >&2
  exit 1
fi

echo "Installed mpi4py extension: $MPI_SO"
readelf -d "$MPI_SO" | rg 'libmpi_gnu|libmpi_gtl_hsa|RPATH|RUNPATH'
