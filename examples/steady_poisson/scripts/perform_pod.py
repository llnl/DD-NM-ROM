"""
Perform SVD on steady-state solutions for the 2D Burgers' equation.
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
from dd_nm_rom.rom.utils import pod as pod_mod
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
field.set_params(mu=field.sample_design_space())
# FOM
fom = utils.get_class(
  modules=[fom_mod],
  name="Burgers2D"
)(mesh=mesh, **inputs["fom"]["kwargs"])
fom.build(field)
# DD-FOM
dd_fom = utils.get_class(
  modules=[fom_mod],
  name="DDBurgers2D"
)(monolithic=fom, constraint_type="strong")
dd_fom.build()
# Saving path
path_to_save = inputs["save_dir"] + "/" + inputs["data_load"]["key"]
os.makedirs(path_to_save, exist_ok=True)

# Data Loading
# =====================================
print("\nLoading data ...")
dataset = utils.load_case_parallel(**inputs["data_load"])
dataset = np.vstack([x for x in dataset if x is not None])
print("> Map dataset on DD elements")
dataset = dd_fom.map_sol_on_elements(dataset, map_on_ports=True)

# SVD
# =====================================
print("\nDecomposition for multiple subdomains ...")
# Perform SVD
svd, bases = pod_mod.compute_svd(
  data=dataset,
  **inputs["svd"]
)
# Saving
path_i = path_to_save + "/multi/"
os.makedirs(path_i, exist_ok=True)
pickle.dump(svd, open(path_i+"/svd.p", "wb"))
if (bases is not None):
  pickle.dump(bases, open(path_i+"/bases.p", "wb"))
  n_bases = ops.map_nested_dict(bases, lambda x: x.shape[1])
  with open(path_i+"/n_bases.json", 'w') as file:
    json.dump(n_bases, file, indent=2)

if (mesh.n_sub > 1):
  print("\nDecomposition for merged subdomains ...")
  # Map data
  dataset_merged = {}
  for (e_k, data_k) in dataset.items():
    if (e_k != "port"):
      dataset_merged[e_k] = [np.vstack(dataset[e_k])]
    else:
      dataset_merged[e_k] = []
      mapping = dd_fom.dd_indices.orientsize_to_ports
      for (orient, size_to_ports) in mapping.items():
        for (size, ports) in size_to_ports.items():
          data_p = [dataset[e_k][p] for p in ports]
          dataset_merged[e_k].append(np.vstack(data_p))

  # Perform SVD
  svd, bases = pod_mod.compute_svd(
    data=dataset_merged,
    **inputs["svd"]
  )
  # Saving
  path_i = path_to_save + "/merged/"
  os.makedirs(path_i, exist_ok=True)
  pickle.dump(svd, open(path_i+"/svd.p", "wb"))
  if (bases is not None):
    pickle.dump(bases, open(path_i+"/bases.p", "wb"))
    n_bases = ops.map_nested_dict(bases, lambda x: x.shape[1])
    with open(path_i+"/n_bases.json", 'w') as file:
      json.dump(n_bases, file, indent=2)

# Copy input file
filename = path_to_save + "/inputs.json"
shutil.copyfile(args.inpfile, filename)

print("\nDone!\n")
