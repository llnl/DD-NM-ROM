import pytest

# TODO: skip data on lustre for now, remake this test
pytestmark = pytest.mark.skip(
  reason="requires extra model data"
)

import numpy as np
import torch

if __name__ == "__main__":
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
      sys.path.insert(0, str(root))

from dd_nm_rom import backend as bkd
from dd_nm_rom import ops

from dd_nm_rom import fom as fom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom import rom as rom_mod
from dd_nm_rom.rom.nonlinear.autoencoder import AutoencoderNP


def setup_module(module):
  bkd.set_backend("torch")
  bkd.set_device("cpu")
  bkd.set_floatx("float64")

def teardown_module(module):
  bkd.finalize_distributed()

@pytest.mark.setup("backend_numpy")
def test_autoencoder_clone():
  # Test the makeShared op on Autoencoder - all pointers should be equal between two instances

  model = "/p/lustre5/colekend/ddrom_dist_scaling_may19/run_3by3_new2/nets/multi/interior/0/scratch/training/ckpt/model_best_numpy.p"

  load_opts = {"weights_only":False, "map_location":{"cuda:0":"cpu"}}

  bkd._start_mem_trace()
  model_config = torch.load(model, **load_opts)
  mem_after_load = bkd._end_mem_trace()
  autoencoder = AutoencoderNP(model_config)

  #mem_after_load = bkd._end_mem_trace()

  # Create a new instance
  bkd._start_mem_trace()
  clone = AutoencoderNP.makeShared(autoencoder)
  mem_after_clone = bkd._end_mem_trace()

  print(" MEM AFTER LOAD: {}".format(mem_after_load))
  print(" MEM AFTER CLONE: {}".format(mem_after_clone))

  models = [autoencoder, clone]
  for i in range(len(models) - 1):
    # compare underlying model data
    m1 = models[i]
    m2 = models[i+1]

    # compare encoders
    for (k, v) in m1.encoder.w.items():
      ptr_1 = bkd.get_tensor_ptr(m1.encoder.w[k])
      ptr_2 = bkd.get_tensor_ptr(m2.encoder.w[k])
      assert (ptr_1 == ptr_2)

    # compare decoders
    for (k, v) in m1.decoder.w.items():
      ptr_1 = bkd.get_tensor_ptr(m1.decoder.w[k])
      ptr_2 = bkd.get_tensor_ptr(m2.decoder.w[k])
      print(k, ptr_1, ptr_2)
      assert (ptr_1 == ptr_2)
    
    
def test_rom_model_load():
  # DD Mesh
  nx_intr = 48
  ny_intr = 48
  lx_sub = 0.5
  ly_sub = 0.5
  x0 = 0.0
  y0 = 0.0
  n_sub_x = 3
  n_sub_y = 3
  # Time grid
  dt = 0.03
  nt = 1
  t_lim = [0, dt*nt]
  # PDE
  viscosity = 1e-3

  mesh = mesh_mod.MeshDD(
      nx_intr=nx_intr,
      ny_intr=ny_intr,
      lx_sub=lx_sub,
      ly_sub=ly_sub,
      x0=x0,
      y0=y0,
      n_sub_x=n_sub_x,
      n_sub_y=n_sub_y,
      #with_bounds=False
      with_bounds=True
  )
  mesh.build()

  field = field_mod.SinPeak(mesh=mesh, mu_lim=[0.9,1.1], bc_type="periodic")
  #field = field_mod.SinMultiPeak(mesh=mesh, mu_lim=[0.9,1.1], bc_type="neumann")
  field.set_params(mu=field.sample_design_space())
  U0 = field.u()
  V0 = field.v()


  #fom = fom_mod.Burgers2D(nu=viscosity, mesh=mesh)
  fom = fom_mod.Burgers2D(nu=viscosity, mesh=mesh, upwind=True, upwind_order=2, compact=True)
  fom.build(field)

  dd_fom = fom_mod.DDBurgers2D(fom, constraint_type='strong')
  dd_fom.build()


  nets_dir = "/p/lustre5/colekend/ddrom_dist_scaling_may19/run_3by3_new2/nets"
  nets_tag = {}
  nets_tag["interior"] = ""
  nets_tag["port"] = ""

  path_to_nets = {}
  for (element, tag) in nets_tag.items():
    suffix = f"/{tag}/merged/{element}/"
    path_to_nets[element] = nets_dir + suffix

  print("\n  PATH TO NETS = {}".format(path_to_nets))

  nn_configfiles = rom_mod.nonlinear.domain_dec.load_nn_configfiles_new(
    mesh=mesh, dd_fom=dd_fom, path_to_nets=path_to_nets)

  # reduce set to load for testing
  #nn_configfiles["interior"] = nn_configfiles["interior"][0:2]
  #nn_configfiles["port"] = nn_configfiles["port"][0:2]

  mem_start = bkd._start_mem_trace()
  dd_rom = rom_mod.DD_NM_ROM(dd_fom=dd_fom,
                            nn_configfiles=nn_configfiles,
                            constraint_type="strong",
                            #constraint_type="weak",
                            n_constraints_weak=-1,
                            scaling=-1,
                            subs_per_rank=mesh.n_sub,
                            check_unique_models=True)
  print(" MEM USAGE AFTER DD-ROM CONSTRUCTION (check unique=False):")
  mem_end = bkd._end_mem_trace()
  bkd._get_mem_trace_stats(mem_start, mem_end, True)


def test_rom_model_load_scale():
  # DD Mesh
  nx_intr = 48
  ny_intr = 48
  lx_sub = 0.5
  ly_sub = 0.5
  x0 = 0.0
  y0 = 0.0
  n_sub_x = 5
  n_sub_y = 5
  # Time grid
  dt = 0.03
  nt = 1
  t_lim = [0, dt*nt]
  # PDE
  viscosity = 1e-3

  mesh = mesh_mod.MeshDD(
      nx_intr=nx_intr,
      ny_intr=ny_intr,
      lx_sub=lx_sub,
      ly_sub=ly_sub,
      x0=x0,
      y0=y0,
      n_sub_x=n_sub_x,
      n_sub_y=n_sub_y,
      #with_bounds=False
      with_bounds=True
  )
  mesh.build()

  field = field_mod.SinPeak(mesh=mesh, mu_lim=[0.9,1.1], bc_type="periodic")
  #field = field_mod.SinMultiPeak(mesh=mesh, mu_lim=[0.9,1.1], bc_type="neumann")
  field.set_params(mu=field.sample_design_space())
  U0 = field.u()
  V0 = field.v()


  #fom = fom_mod.Burgers2D(nu=viscosity, mesh=mesh)
  fom = fom_mod.Burgers2D(nu=viscosity, mesh=mesh, upwind=True, upwind_order=2, compact=True)
  fom.build(field)

  dd_fom = fom_mod.DDBurgers2D(fom, constraint_type='strong')
  dd_fom.build()


  nets_dir = "/p/lustre5/colekend/ddrom_dist_scaling_may19/run_3by3_new2/nets"
  nets_tag = {}
  nets_tag["interior"] = ""
  nets_tag["port"] = ""

  path_to_nets = {}
  for (element, tag) in nets_tag.items():
    suffix = f"/{tag}/merged/{element}/"
    path_to_nets[element] = nets_dir + suffix

  print("\n  PATH TO NETS = {}".format(path_to_nets))

  nn_configfiles = rom_mod.nonlinear.domain_dec.load_nn_configfiles_new(
    mesh=mesh, dd_fom=dd_fom, path_to_nets=path_to_nets)

  # reduce set to load for testing
  #nn_configfiles["interior"] = nn_configfiles["interior"][0:2]
  #nn_configfiles["port"] = nn_configfiles["port"][0:2]

  mem_start = bkd._start_mem_trace()
  dd_rom = rom_mod.DD_NM_ROM(dd_fom=dd_fom,
                            nn_configfiles=nn_configfiles,
                            constraint_type="strong",
                            #constraint_type="weak",
                            n_constraints_weak=-1,
                            scaling=-1,
                            subs_per_rank=mesh.n_sub,
                            check_unique_models=True)
  print(" MEM USAGE AFTER DD-ROM CONSTRUCTION (check unique=False):")
  mem_end = bkd._end_mem_trace()
  bkd._get_mem_trace_stats(mem_start, mem_end, True)



if __name__ == "__main__":
  setup_module(None)
  #test_autoencoder_clone()
  #test_rom_model_load()
  test_rom_model_load_scale()
  teardown_module(None)
