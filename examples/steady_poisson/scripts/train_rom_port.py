"""
Train ports autoencoders for DD-NM-ROM.
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
# Define working path
path = inputs["model"]["path"]
path += "/merged/" if inputs["trainable"]["merged"] else "/multi/"
path += "/port/"
# Collect all nn models for ports training
nn_models = []
for (orient, size_to_ports) in dd_fom.dd_indices.orientsize_to_ports.items():
  for (size, ports) in size_to_ports.items():
    if inputs["trainable"]["merged"]:
      nn_models.append((orient, size, ports))
    else:
      for p in ports:
        nn_models.append((orient, size, [p]))

for model in nn_models:
  orient, size, ports = model
  print(
    f"\n> Training ports {ports} of '{orient}' orientation and size {size} ..."
  )
  # Set dataset and saving path
  dset_i = []
  # path_i = path + f"/orient_{orient}_size_{size}_port"
  path_i = path + f"/orient_{orient}_size_{size}_ports"
  for p in ports:
    dset_i.append(dataset["port"][p])
    path_i += f"_{p}"
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
    **inputs["autoencoder"]["dim"]["port"],
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
