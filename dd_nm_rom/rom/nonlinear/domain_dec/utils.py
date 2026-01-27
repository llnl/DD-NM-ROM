import os

from dd_nm_rom import ops, utils


def load_nn_configfiles(mesh, dd_fom, path_to_nets):
  nn_configfiles = {}
  for (element, path_to_net) in path_to_nets.items():
    utils.check_path(path_to_net)
    nn_configfiles[element] = []
    if (element == "port"):
      # > Port
      for p in dd_fom.dd_indices.ports:
        orient, size = dd_fom.dd_indices.port_to_orientsize[p]
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
              nn_configfiles[element].append(filename)
    else:
      # > Interior/Interface
      for _ in range(mesh.n_sub):
        for file in os.scandir(path_to_net):
          if ((file.is_dir()) and ("subs" in file.name)):
            filename = file.path + "/refine/"
            if (not os.path.exists(filename)):
              filename = file.path + "/scratch/"
            filename += "/training/ckpt/model_best_numpy.p"
            nn_configfiles[element].append(filename)
  ops.map_nested_dict(nn_configfiles, utils.check_path)
  nn_configfiles = ops.map_nested_dict(nn_configfiles, os.path.abspath)
  return nn_configfiles
