import os

from dd_nm_rom import ops, utils

def load_ls_configfiles(mesh, dd_fom, path_to_nets) -> dict:
  # mesh provides the number of subdomains
  # dd_fom provides domain decomposition indices
  ls_configfiles = {}
  for (element, path_to_net) in path_to_nets.items():
    utils.check_path(path_to_net)
    ls_configfiles[element] = []
    if (element == "port"):
      # > Port
      for p in dd_fom.dd_indices.ports:
        orient, size = dd_fom.dd_indices.port_to_orientsize[p]
        print(orient,size)
        for file in os.scandir(path_to_net):
          if ((file.is_dir()) and (element in file.name)):
            if ((str(orient) in file.name) and (str(size) in file.name)):
              filename = file.path + "/refine/"
              if (not os.path.exists(filename)):
                filename = file.path + "/scratch/"
              if (size == 1):
                filename += "/bases.p"
              else:
                filename += "/bases.p"
              ls_configfiles[element].append(filename)
    else:
      # > Interior/Interface
      for _ in range(mesh.n_sub):
        for file in os.scandir(path_to_net):
          if ((file.is_dir()) and ("subs" in file.name)):
            filename = file.path + "/refine/"
            if (not os.path.exists(filename)):
              filename = file.path + "/scratch/"
            filename += "/bases.p"
            ls_configfiles[element].append(filename)
  ops.map_nested_dict(ls_configfiles, utils.check_path)
  ls_configfiles = ops.map_nested_dict(ls_configfiles, os.path.abspath)
  return ls_configfiles
