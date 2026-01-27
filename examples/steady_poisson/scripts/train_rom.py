"""
Train interior and interface autoencoders for DD-NM-ROM.
"""

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

from dd_nm_rom import utils
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.rom.nonlinear import Autoencoder, Data, Model

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
  name="Poisson2D"
)(mesh=mesh, **inputs["fom"]["kwargs"])
fom.build(field)
# DD-FOM
dd_fom = utils.get_class(
  modules=[fom_mod],
  name="DDPoisson2D"
)(monolithic=fom, constraint_type="strong")
dd_fom.build()

# Data Loading
# =====================================
print("\nLoading data ...")
dataset1 = utils.load_case_parallel(**inputs["data_load"])
dataset1 = np.vstack([x for x in dataset1 if x is not None])
print(dataset1.shape)
dataset2 = utils.load_case_parallel(**inputs["data_load_2"])
dataset2 = np.vstack([x for x in dataset2 if x is not None])
print(dataset2.shape)
dataset = np.vstack([dataset1, dataset2])
print(dataset.shape)
print("> Map dataset on DD elements")
dataset = dd_fom.map_sol_on_elements(dataset, map_on_ports=True)

# Autoencoders
# =====================================
print("\nTraining autoencoders ...")
nn_models = []
path = inputs["model"]["path"]
# Define subdomains used for training
subs = inputs["trainable"]["subdomains"]
if (subs is None):
  subs = np.arange(mesh.n_sub).tolist()
# Collect all nn models for interior/interface training
if inputs["trainable"]["merged"]:
  for e in inputs["trainable"]["elements"]:
    nn_models.append((e,subs))
  path += "/merged/"
else:
  for e in inputs["trainable"]["elements"]:
    for s in subs:
      nn_models.append((e,[s]))
  path += "/multi/"

for model in nn_models:
  e, subs = model
  print(f"\n> Training '{e}' element on subdomains {subs} ...")
  # Set dataset and saving path
  dset_i = []
  path_i = path + f"/{e}/subs"
  for s in subs:
    dset_i.append(dataset[e][s])
    path_i += f"_{s}"
  dset_i = np.vstack(dset_i)
  # Initialize data object
  data = Data(
    snapshots=dset_i,
    **inputs["data"]
  )
  # Initialize autoencoder object
  refine = inputs["autoencoder"].get("refine", False)
  if refine:
    saved_model = path_i + "/scratch/training/ckpt/model_best_torch.p"
    inputs["autoencoder"]["common"]["loading"] = True
    inputs["autoencoder"]["common"]["saved_model"] = saved_model
    path_i = path_i + "/refine/"
  else:
    path_i = path_i + "/scratch/"
  net = Autoencoder(
    ref=data.ref,
    scale=data.scale,
    input_dim=data.snapshots.shape[1],
    **inputs["autoencoder"]["dim"][e],
    **inputs["autoencoder"]["common"]
  )
  # Initialize model object
  nn_mdl = Model(
    net=net,
    data=data,
    path=path_i
  )
  nn_mdl.compile(**inputs["model"]["compile"])
  nn_mdl.train(**inputs["model"]["train"])

  # Copy input file
  shutil.copyfile(args.inpfile, path_i+"/inputs.json")

print("\nDone!\n")