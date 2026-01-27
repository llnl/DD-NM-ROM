"""
Test DD-NM-ROM on time-dependent 2D Burgers' equation.
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

# Import "dd_nm_rom" package
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
import numpy as np
import dill as pickle

from tqdm import tqdm
from dd_nm_rom import ops
from dd_nm_rom import utils
from dd_nm_rom import postproc
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import rom as rom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.rom.utils import pod as pod_mod
from dd_nm_rom.elements import mesh as mesh_mod

# Initialization
# =====================================
print("\nInitialization ...")
# Mesh
mesh = utils.get_class(modules=[mesh_mod], **inputs["mesh"])
mesh.build()
X, Y = mesh.grid
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
)(monolithic=fom, **inputs["dd_fom"]["kwargs"])
dd_fom.build()

Mu = pickle.load(open("/home/pinghsuan/Research/DD-NM-ROM/run/Poisson/dset.2by2/datagen/mu.p", "rb"))
mu_train = Mu['train'][:2000, :]
mu_test = Mu['test']
print(mu_train[(mu_train[:, -2] == 2) & (mu_train[:, -1] == 2)])
print(mu_test[(mu_test[:, -2] == 2) & (mu_test[:, -1] == 2)])

# Data loading
# =====================================
print("\nLoading test cases ...")
test_cases = utils.load_case_parallel(**inputs["data_load"])
test_cases = [case for case in test_cases if case is not None]

print("\nLoad training data for RBF")
snapshots = utils.load_case_parallel(
  n_workers=32,
  ranges=[0,2000],
  key="snapshots",
  path="/home/pinghsuan/Research/DD-NM-ROM/run/Poisson/dset.2by2/datagen/train/"
)
snapshots = np.vstack([x for x in snapshots if x is not None])

print("\nLoading POD data ...")
filename = inputs["paths"]["pod_dir"] + "/merged/svd.p"
svd = pickle.load(open(filename, "rb"))["res"][0]

# NN models loading
# =====================================
print("\nLoading NN configuration files ...")
path_to_nets = {}
for (element, tag) in inputs["paths"]["nets_tag"].items():
  suffix = f"/{tag}/merged/{element}/"
  path_to_nets[element] = inputs["paths"]["nets_dir"] + suffix
nn_configfiles = rom_mod.nonlinear.domain_dec.load_nn_configfiles(
  mesh=mesh, dd_fom=dd_fom, path_to_nets=path_to_nets
)
print("\nNN configuration files keys:")
print(nn_configfiles.keys())
print(nn_configfiles['port'])

# Solving test cases
# =====================================
def update_stats(stats, istats):
  return {k: np.append(stats[k], istats[k]) for k in istats.keys()}

print(f"\nTesting DD-NM-ROM model ...")
# Saving path
tag = [tag for tag in inputs["paths"]["nets_tag"].values()]
tag = "_".join(tag)
path = inputs["paths"]["figs_dir"] + f"/{tag}/"
os.makedirs(path, exist_ok=True)
# Loop over cases
error, speedup, runtime, converged = {}, {}, {}, {}
for icase in tqdm(test_cases, desc="> Test cases", ncols=80, file=sys.stdout):
  # Saving path
  # ---------------
  path_i = path + f"/case_{str(icase['index']+1).zfill(5)}/"
  os.makedirs(path_i, exist_ok=True)
  # Test case set up
  # ---------------
  x0 = icase["snapshots"][0]
  solver = icase["solver"]
  solver['tol'] = 1e-07
  uv_fom = icase["solution"]
  runtime_fom = icase["runtime"]
  # DD-FOM
  # ---------------
  # > Building
  field.set_params(icase["mu"])
  force = field.get_force()
  fom.build(field, force=force)
  dd_fom.build()
  dd_fom.get_force(force)
  # > Solution
  path_ij = path_i + "/fom/"
  # >> Statistics
  os.makedirs(path_ij, exist_ok=True)
  with open(path_ij + "/runtime.json", "w") as file:
    json.dump(runtime_fom, file, indent=2)
  # >> Postprocessing
  postproc.plot_field_fom_rom(
    path=path_ij,
    mesh=mesh,
    uv_fom=uv_fom,
    uv_rom=None,
    index=None,
    show_labels=False
  )
  print(icase["mu"])
# postproc.animate_fom_rom(path_ij, mesh, uv_fom)
  # DD-NM-ROM
  # ---------------
  #for (r, e_min) in enumerate((None, 1e-6, 1e-7, 1e-8)):
  for (r, e_min) in enumerate((None, )):
    rom_id = "srpc_" + str(r+1).zfill(2)
    # > Configuration
    hr_active = False if (e_min is None) else True
    e_min = 1e-6 if (e_min is None) else e_min
    res_bases = pod_mod.get_pod_bases(svd=svd, energy_min=e_min)
    hr_n_samples = int(2*res_bases.shape[1])
    # > Building
    dd_rom = rom_mod.DD_NM_ROM(
      dd_fom=dd_fom,
      nn_configfiles=nn_configfiles,
      res_bases=res_bases,
      hr_active=hr_active,
      hr_n_samples=hr_n_samples,
      hr_n_edge_samples_ratio=2.0/3.0,
      hr_sample_small_ports=True,
      hr_small_ports_dim=5,
      constraint_type="strong",
      n_constraints_weak=-1,
      scaling=-1
    )
    # > Static solution
    if (not hr_active):
      # >> Saving path
      path_ij = path_i + "/rom/srpc_rec/"
      os.makedirs(path_ij, exist_ok=True)
      # >> ROM solution
      print(uv_fom["res"]["u"].shape)
      x_fom = np.vstack([uv_fom["res"]["u"], uv_fom["res"]["v"]])
      uv_rom_rec = dd_rom.reconstruct_static(x_fom)
      # >> Postprocessing
      postproc.plot_field_fom_rom(
        path=path_ij,
        mesh=mesh,
        uv_fom=uv_fom,
        uv_rom=uv_rom_rec,
        index=None,
        show_labels=False
      )
#     postproc.animate_fom_rom(path_ij, mesh, uv_fom, uv_rom_rec)
    # > Dynamic solution
    # >> Saving path
    path_ij = path_i + f"/rom/{rom_id}/"
    os.makedirs(path_ij, exist_ok=True)
    # >> Configuration
    cfg = {
      "hr_active": hr_active,
      "hr_n_samples": hr_n_samples,
      "energy_min": e_min
    }
    with open(path_ij + "/cfg.json", "w") as file:
      json.dump(cfg, file, indent=2)
    # >> HR nodes
    if hr_active:
      postproc.plot_hr_nodes(
        mesh=mesh,
        dd_rom=dd_rom,
        path=path_ij
      )
#   print('Computing RBF interpolant ...')
#   dd_data = dd_fom.map_sol_on_elements(np.vstack(snapshots), map_on_ports=False)
#   dd_rom.rbf_model.build(dd_data, mu_train, smoothing=0.0, kernel='linear')
#   print('Interpolant computed!')
    # >> Solving
#   solver["verbose"] = False
    solver["verbose"] = True
    solver['maxit'] = 300
    uv_rom, *_, iconverged = dd_rom.solve(
      x0=dd_rom.get_init_sol(x=x0),
      runtime=0.0,
      use_guess=False,
      **solver
    )
#   uv_rom, *_, iconverged = dd_rom.solve(
#     mu=icase["mu"],
#     runtime=0.0,
#     use_guess=False,
#     **solver
#   )
    if (rom_id not in converged):
      converged[rom_id] = np.array([])
    converged[rom_id] = np.append(converged[rom_id], int(iconverged))
    if (not iconverged):
      with open(path_ij+"/error.txt", "w") as file:
        file.write("Solver not converged.")
      continue
    # >> Statistics - Single case
    iruntime = dd_rom.runtime
    ierror = dd_rom.compute_error(
      uv_fom, uv_rom, scaling=True, relative=False, axis=0
    )
    ispeedup = {k: tk/iruntime[k] for (k, tk) in runtime_fom.items()}
    stats = {"runtime": iruntime, "speedup": ispeedup, "error": ierror}
    with open(path_ij + "/stats.json", "w") as file:
      json.dump(stats, file, indent=2)
    # >> Statistics - Global
    if (rom_id not in error):
      error[rom_id] = ierror
      speedup[rom_id] = ispeedup
      runtime[rom_id] = iruntime
    else:
      error[rom_id] = np.append(error[rom_id], ierror)
      speedup[rom_id] = update_stats(speedup[rom_id], ispeedup)
      runtime[rom_id] = update_stats(runtime[rom_id], iruntime)
    # >> Postprocessing
    postproc.plot_field_fom_rom(
      path=path_ij,
      mesh=mesh,
      uv_fom=uv_fom,
      uv_rom=uv_rom,
      index=None,
      show_labels=False
    )
#   postproc.animate_fom_rom(path_ij, mesh, uv_fom, uv_rom)

# Mean statistics
stats_mean = {
  "speedup": ops.map_nested_dict(speedup, ops.compute_stats),
  "runtime": ops.map_nested_dict(runtime, ops.compute_stats),
  "error": ops.map_nested_dict(error, ops.compute_stats),
  "conv_perc": ops.map_nested_dict(converged, lambda x: 100*np.sum(x)/len(x))
}
with open(path+"/stats_mean.json", "w") as file:
  json.dump(stats_mean, file, indent=2)
# Copy input file
with open(path+"/inputs.json", "w") as file:
  json.dump(inputs, file, indent=2)

print("\nDone!\n")
