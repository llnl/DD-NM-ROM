import os
import numpy as np

from dd_nm_rom import ops, utils
import dd_nm_rom.backend as bkd


def load_nn_configfiles(mesh, dd_fom, path_to_nets):
  nn_configfiles = {}
  for (element, path_to_net) in path_to_nets.items():
    utils.check_path(path_to_net)
    nn_configfiles[element] = []
    if (element == "port"):
      # > Port
      for p in dd_fom.dd_indices.ports:
        orient, size = dd_fom.dd_indices.port_to_orientsize[p]
        print("  LOAD NN CONFIGFILES: RANK {}: PORT {} -- checking for orient = {} size = {}".format(bkd.get_rank(), p, orient, size))
        for file in os.scandir(path_to_net):
          if ((file.is_dir()) and (element in file.name)):
            if ((str(orient) in file.name) and (str(size) in file.name)):
              filename = file.path + "/refine/"
              if (not os.path.exists(filename)):
                filename = file.path + "/scratch/"
              if (size == 1):
                filename += "/saving/model_last_numpy.p"
              else:
                filename += "/training/ckpt/model_best_numpy.p"
              print("  LOAD NN CONFIGFILES: RANK {}: PORT {} FILE '{}' for orient '{}' size = '{}'".format(bkd.get_rank(), p, filename, orient, size))
              nn_configfiles[element].append(filename)
    else:
      # > Interior/Interface
      for s in range(mesh.n_sub):
        #if bkd.distributed() and bkd.get_rank() != s:
        # TODO: fix - this uses the same subdomains as the FOM, change to ROM to allow custom subdomain/rank mapping per model
        if bkd.distributed() and np.isin(s, dd_fom.global_subdomains, invert=True):
          continue
        print("  LOAD NN CONFIGFILES: RANK {}: INTERIOR SUB {}/{}".format(bkd.get_rank(), s, mesh.n_sub))
        for file in os.scandir(path_to_net):
          print("  LOAD NN CONFIGFILES: RANK {}: INTERIOR SUB {}/{}:: checking file '{}'".format(bkd.get_rank(), s, mesh.n_sub, file.path))
          if ((file.is_dir()) and ("subs" in file.name)):
            filename = file.path + "/refine/"
            if (not os.path.exists(filename)):
              filename = file.path + "/scratch/"
            filename += "/training/ckpt/model_best_numpy.p"
            print("  LOAD NN CONFIGFILES: RANK {}: INTERIOR SUB {}/{}:: SUBDOMAIN FILE '{}' for element '{}'".format(bkd.get_rank(), s, mesh.n_sub, filename, element))
            nn_configfiles[element].append(filename)
  ops.map_nested_dict(nn_configfiles, utils.check_path)
  nn_configfiles = ops.map_nested_dict(nn_configfiles, os.path.abspath)
  return nn_configfiles


def get_model_paths(fpath, fname="models.txt"):
  #keys, values = np.loadtxt(os.path.join(fpath, fname), delimiter=':', unpack=True, skiprows=1, dtype=str)
  # keys, values = np.genfromtxt(os.path.join(fpath, fname), delimiter=':', unpack=True, skiprows=1)

  model_path = os.path.join(fpath, fname)
  if not os.path.exists(model_path):
    return None

  values = np.genfromtxt(model_path, delimiter='\n', dtype='str')
  if values.ndim == 0:
    values = [str(values.item())]

  data_dict = {}
  for line in values:
    k, v = line.split(":")
    v = v.lstrip()
    data_dict[k] = v

  return data_dict


def load_nn_configfiles_new(mesh, dd_fom, path_to_nets):
  print("-- loading nn configs in directory '{}'...".format(path_to_nets))

  nn_configfiles = {}
  for (element, path_to_net) in path_to_nets.items():
    utils.check_path(path_to_net)

    model_paths = get_model_paths(path_to_net)
    # whether we found a models.txt file in the trained model directory, use those instead
    # this can help avoid long file path names if the number of ports/interior subdomains is large
    # if the file doesn't exist, then we just fall back to scanning the directory, which may cause a OSError if filepaths are long
    use_model_paths = True if model_paths is not None else False

    nn_configfiles[element] = []
    if (element == "port"):
      # > Port
      for p in dd_fom.dd_indices.ports:
        orient, size = dd_fom.dd_indices.port_to_orientsize[p]
        #print("  LOAD NN CONFIGFILES: RANK {}: PORT {} -- checking for orient = {} size = {}".format(bkd.get_rank(), p, orient, size))
        for file in os.scandir(path_to_net):
          if not file.is_dir():
            continue
          if use_model_paths:
            if file.name not in model_paths:
              raise RuntimeWarning(f"Expected to find {file.name} in {model_paths} but it does not exist!")
            # get actual full filename
            fname = str(model_paths[file.name])
          else:
            fname = file.name
          if element not in fname:
            raise RuntimeWarning(f"Looking for element {element} but it is not a part of {fname}")
          if ((str(orient) in fname) and (str(size) in fname)):
            filename = file.path + "/refine/"
            if (not os.path.exists(filename)):
              filename = file.path + "/scratch/"
            if (size == 1):
              filename += "/saving/model_last_numpy.p"
            else:
              filename += "/training/ckpt/model_best_numpy.p"
            #print("  LOAD NN CONFIGFILES: RANK {}: PORT {} FILE '{}' for orient '{}' size = '{}'".format(bkd.get_rank(), p, filename, orient, size))
            nn_configfiles[element].append(filename)
    else:
      # > Interior/Interface
      for s in range(mesh.n_sub):
        #if bkd.distributed() and bkd.get_rank() != s:
        # TODO: fix - this uses the same subdomains as the FOM, change to ROM to allow custom subdomain/rank mapping per model
        if bkd.distributed() and np.isin(s, dd_fom.global_subdomains, invert=True):
          continue
        print("  LOAD NN CONFIGFILES: RANK {}: INTERIOR SUB {}/{}".format(bkd.get_rank(), s, mesh.n_sub))
        for file in os.scandir(path_to_net):
          #print("  LOAD NN CONFIGFILES: RANK {}: INTERIOR SUB {}/{}:: checking file '{}'".format(bkd.get_rank(), s, mesh.n_sub, file.path))
          if not file.is_dir():
            continue
          if use_model_paths:
            if file.name not in model_paths:
              raise RuntimeWarning(f"Expected to find {file.name} in {model_paths} but it does not exist!")
            # get actual full filename
            fname = model_paths[file.name]
            print(" SUB {} FILE = '{}' MODEL FILENAME = '{}'".format(s, file.name, fname))
          else:
            fname = file.name
          if "subs" not in fname:
            raise RuntimeWarning(f"Looking for 'subs' but it is not a part of {fname}")
          filename = file.path + "/refine/"
          if (not os.path.exists(filename)):
            filename = file.path + "/scratch/"
          filename += "/training/ckpt/model_best_numpy.p"
          print("  LOAD NN CONFIGFILES: RANK {}: INTERIOR SUB {}/{}:: SUBDOMAIN FILE '{}' for element '{}'".format(bkd.get_rank(), s, mesh.n_sub, filename, element))
          nn_configfiles[element].append(filename)
  ops.map_nested_dict(nn_configfiles, utils.check_path)
  nn_configfiles = ops.map_nested_dict(nn_configfiles, os.path.abspath)

  # if bkd.root():
  #   for (element, configs) in nn_configfiles.items():
  #     print("NN CONFIG FILES: element '{}'".format(element))
  #     for (i, config) in enumerate(configs):
  #       print(" CONFIG {}/{} = '{}'".format(i, len(configs), config))

  return nn_configfiles
