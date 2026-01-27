def set_style(pyplot):
  pyplot.rcParams.update({
    "lines.linewidth": 1.5,
    "lines.markersize": 10,
    "text.usetex": False,
    "font.size": 20,
    "font.family": "serif",
#   "font.serif": "Computer Modern",
    "font.serif": "Dejavu Serif",
    "figure.max_open_warning": int(1e3)
  })
  return pyplot
