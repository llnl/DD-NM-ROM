"""
Compute interior and interface SVD for DD-LS-ROM.
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
from tqdm import tqdm

from dd_nm_rom import ops
from dd_nm_rom import utils
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.rom.linear import Data, Model
from dd_nm_rom import postproc
from dd_nm_rom.rom.utils import pod as pod_mod

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

# Data Loading
# =====================================
print("\nLoading data ...")
dataset = utils.load_case_parallel(**inputs["data_load"])
dataset = np.vstack([x for x in dataset if x is not None])
print("> Map dataset on DD elements")
dataset = dd_fom.map_sol_on_elements(dataset, map_on_ports=True)
print((dataset['res'][0].shape))

# Linear subspace
# =====================================
print("\nBuilding linear subspace...")
for emin in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]:
  ls_models = []
  path = inputs["model"]["path"]
  # Define subdomains used for training
  subs = inputs["trainable"]["subdomains"]
  if (subs is None):
      subs = np.arange(mesh.n_sub).tolist()
  # Collect all ls models for interior/interface training
  if inputs["trainable"]["merged"]:
      for e in inputs["trainable"]["elements"]:
          ls_models.append((e,subs))
          inputs["svd"]["energy_min"][e] = emin
      path += "/merged/"
  else:
      for e in inputs["trainable"]["elements"]:
          for s in subs:
              ls_models.append((e,[s]))
      path += "/multi/"

  for model in ls_models:
      e, subs = model
      print(f"\n> Training '{e}' element on subdomains {subs} ...")
      dataset_merged = {}
      dataset_merged[e] = [np.vstack(dataset[e])]

      path_i = path + f"/{e}_emin_{emin}/subs"
      for s in subs:
        path_i += f"_{s}"
#     path_i +=f"_emin_{emin}" 

      path_i = path_i + "/scratch/"
      print(path_i)

      # Perform SVD
      svd, bases = pod_mod.compute_svd(
        data=dataset_merged,
        **inputs["svd"]
      )
      # Saving
      os.makedirs(path_i, exist_ok=True)
      pickle.dump(svd, open(path_i+"/svd.p", "wb"))
      if (bases is not None):
        pickle.dump(bases, open(path_i+"/bases.p", "wb"))
        n_bases = ops.map_nested_dict(bases, lambda x: x.shape[1])
        with open(path_i+"/n_bases.json", 'w') as file:
          json.dump(n_bases, file, indent=2)

      # Copy input file
      shutil.copyfile(args.inpfile, path_i+"/inputs.json")

  print("\nDone!\n")