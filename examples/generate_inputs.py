import os
from pathlib import Path
import sys
import json
import argparse

_EXAMPLES = {"unsteady", "steady", "steady_poisson"}

# default paths if not provided, relative to the prefix (user supplied or current directory)
default_paths = {"datagen_dir": "./run/datagen/",
                 "pod_dir": "./run/pod/",
                 "train_dir": "./run/nets/",
                 "figs_dir": "./run/figs/"}

parser = argparse.ArgumentParser()
parser.add_argument("--example", type=str, help="Name of example to generate inputs for", default="unsteady")
parser.add_argument("--pathfile", type=str, help="Path to optional JSON file defining paths and common options", default="")
parser.add_argument("--prefix", type=str, help="Path of base directory to generate paths from", default="")
args = parser.parse_args()

if args.example not in _EXAMPLES:
    raise RuntimeError("Invalid example name! Expected one of: {}".format(str(_EXAMPLES)))

if args.pathfile == "":
    inputs = dict()
else:
    with open(args.pathfile) as file:
        inputs = json.load(file)

if args.prefix == "":
    prefix = Path.cwd()
else:
    prefix = Path(args.prefix).resolve()

if "paths" not in inputs:
    # no provided paths found, use defaults
    inputs["paths"] = default_paths
    inputs["paths"]["prefix"] = prefix
elif "prefix" in inputs["paths"]:
    # update prefix using user provided value
    prefix = Path(inputs["paths"]["prefix"]).resolve()

print("Setting run prefix to: {}".format(prefix))

# prepend the prefix to all paths
for path in inputs["paths"]:
    if path == "prefix": continue
    inputs["paths"][path] = Path(prefix, inputs["paths"][path]).resolve()
    print("  {} = {}".format(path, inputs["paths"][path]))

example_dir = Path(f"./{args.example}")
inputs_dir = Path(example_dir, "inputs")

datagen_files = sorted(inputs_dir.glob("generate_data*.json"))
pod_files = sorted(inputs_dir.glob("perform_pod*.json"))
train_files = sorted(inputs_dir.glob("train_*.json"))
test_files = sorted(inputs_dir.glob("test_*.json"))

# Update data generation inputs
for inp_file in datagen_files:
    with open(inp_file, "r") as file:
        inp_file_i = json.load(file)
    inp_file_i["save_dir"] = str(inputs["paths"]["datagen_dir"])
    with open(inp_file, "w") as file:
        json.dump(inp_file_i, file, indent=2)

# Update POD inputs
for inp_file in pod_files:
    with open(inp_file, "r") as file:
        inp_file_i = json.load(file)
    inp_file_i["data_load"]["path"] = str(Path(inputs["paths"]["datagen_dir"], "train/"))
    inp_file_i["save_dir"] = str(inputs["paths"]["pod_dir"])
    with open(inp_file, "w") as file:
        json.dump(inp_file_i, file, indent=2)

# Update training inputs
for inp_file in train_files:
    with open(inp_file, "r") as file:
        inp_file_i = json.load(file)
    inp_file_i["data_load"]["path"] = str(Path(inputs["paths"]["datagen_dir"], "train/"))
    inp_file_i["model"]["path"] = str(inputs["paths"]["train_dir"])
    with open(inp_file, "w") as file:
        json.dump(inp_file_i, file, indent=2)

# Update testing inputs
for inp_file in test_files:
    with open(inp_file, "r") as file:
        inp_file_i = json.load(file)
    inp_file_i["data_load"]["path"] = str(Path(inputs["paths"]["datagen_dir"], "test/"))

    inp_file_i["paths"]["pod_dir"] = str(Path(inputs["paths"]["pod_dir"], "snapshots/"))
    inp_file_i["paths"]["nets_dir"] = str(inputs["paths"]["train_dir"])
    inp_file_i["paths"]["figs_dir"] = str(inputs["paths"]["figs_dir"])
    with open(inp_file, "w") as file:
        json.dump(inp_file_i, file, indent=2)

print("Updated input files:")
for inp_file in (datagen_files + pod_files + train_files + test_files):
    print("  -- {}".format(str(inp_file)))
