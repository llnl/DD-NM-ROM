"""
Generate solutions for the 2D Poisson' equation.
"""

import os
import sys
import json
import argparse

# Inputs
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument("--inpfile", type=str, help="path to JSON input file")
args = parser.parse_args()

with open(args.inpfile) as file:
  inputs = json.load(file)

# Import 'dd_nm_rom' package
# =====================================
with open(inputs["pathfile"]) as file:
  paths = file.read().splitlines()
sys.path.extend(paths)

# Environment
# =====================================
from dd_nm_rom import env
env.set(**inputs["env"])

# Libraries
# =====================================
import shutil
import numpy as np
import dill as pickle

from dd_nm_rom import ops
from dd_nm_rom import utils
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import mesh as mesh_mod

# Initialization
# =====================================
print("\nInitialization ...")
# Mesh
mesh = utils.get_class(modules=[mesh_mod], **inputs["mesh"])
mesh.build()
# Field
field = utils.get_class(
  modules=[field_mod],
  name=inputs["field"]["name"]
)(mesh=mesh, **inputs["field"]["kwargs"])
# FOM
fom = utils.get_class(
  modules=[fom_mod],
  name="Poisson2D"
)(mesh=mesh, **inputs["fom"]["kwargs"])
# DD-FOM
dd_fom = utils.get_class(
  modules=[fom_mod],
  name="DDPoisson2D"
)(monolithic=fom, **inputs["dd_fom"]["kwargs"])
# Construct design matrix
n_samples = inputs["data_gen"]["n_samples"]

#mu = field.construct_design_mat(n_samples=inputs["data_gen"]["n_samples"])
mu = {"train": [], "test": []}
if (n_samples.get("train", 0) > 0):
    mu["train"] = field.construct_design_mat(n_samples["train"])
if (n_samples.get("test", 0) > 0):
    mu["test"] = field.construct_design_mat_test(
    n_samples=n_samples["test"], dmat_train=mu["train"], tol=1e-1
    )

# Saving path
path_to_save = {}
for dset in ("train", "test"):
  if (len(mu[dset]) > 0):
    path_to_save[dset] = inputs["save_dir"] + f"/{dset}/"
ops.map_nested_dict(path_to_save, lambda p: os.makedirs(p, exist_ok=True))

# Data generation
# =====================================
# Solution function
# -------------------------------------
def make_compute_sol(dset="train"):
  def compute_sol(index):
    mu_i = mu[dset][index]
    # Set current parameters
    field.set_params(mu_i)
    # Build and solve
    force = field.get_force()
    fom.build(field, force=force)
    if (dset == "train"):
      # > Use FOM
      uv, *_, converged = fom.solve(**inputs["solver"])
    else:
      # > Use DD-FOM
      dd_fom.build()
      dd_fom.get_force(force)
      uv, *_, converged = dd_fom.solve(**inputs["solver"])
    # Save solution
    if converged:
      case_i = {
        "index": index,
        "mu": mu_i,
        "mesh": inputs["mesh"],
        "solver": inputs["solver"]
      }
      if (dset == "train"):
        # > Store FOM-related data
        case_i.update({
          "snapshots": np.vstack([uv["u"], uv["v"]]).T,
          "runtime": fom.runtime
        })
      else:
        # > Store DD-FOM-related data
        case_i.update({
          "snapshots": np.vstack([uv["res"]["u"], uv["res"]["v"]]).T,
          "solution": uv,
          "runtime": dd_fom.runtime
        })
      utils.save_case(path=path_to_save[dset], index=index, data=case_i)
    return int(converged)
  return compute_sol

# Parallel data generation
# -------------------------------------
print("\nData generation ...")
for dset in ("Train", "Test"):
  dsetl = dset.lower()
  if (len(mu[dsetl]) > 0):
    utils.generate_case_parallel(
      sol_fun=make_compute_sol(dsetl),
      n_samples=n_samples[dsetl],
      n_workers=1 if (dsetl == "test") else inputs["data_gen"]["n_workers"],
      desc=f"> {dset} data"
    )
# Save parameters
filename = inputs["save_dir"] + "/mu.p"
pickle.dump(mu, open(filename, "wb"))
# Copy input file
filename = inputs["save_dir"] + "/inputs.json"
with open(filename, "w") as file:
  json.dump(inputs, file, indent=2)

print("\nDone!\n")
