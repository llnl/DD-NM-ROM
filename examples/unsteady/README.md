## DD-NM-ROM Example

### Unsteady Burgers' Equation

This example evaluates the performance of a "bottom-up" strategy with DD-NM-ROM for solving the unsteady Burgers' equation.

Unless specified otherwise, the following steps should be executed in the `commands` directory, using the syntax of the specific LLNL computing platform or by running `bash <script>.sh`.

#### 1. Generating Snapshot Data for Training
Generate the necessary data by running `generate_data.sh` in the `commands` directory. Modify the input file `inputs/generate_data.json` as needed.

#### 2. Performing SVD Decomposition
Perform the SVD decomposition of the data by running `perform_pod.sh` in the `commands` directory. Update the input file `inputs/perform_pod.json` accordingly.

#### 3. Training Models
Train the following autoencoders:

- Interior states: Run `train_rom.sh` in the `commands` directory. Edit the input file `inputs/train_rom.json` as required.
- Port states: Run `train_rom_port.sh` in the `commands` directory. Modify the input file `inputs/train_rom_port.json`.

You can explore multiple latent space dimensions by running parallel training sessions. Use the command `bash command.sh` in the `parallel_train` directory.

#### 4. Testing Models
- Test the model on the same domain (e.g., 2-by-2) by running `test_dd_nmrom.sh` in the `commands` directory. Update the input file `inputs/test_dd_nmrom.json` as needed. For parallel testing, combining different latent space dimensions for port states and interior states autoencoders, run `bash command.sh` in the `parallel_test` directory.
- Test the models on a larger domain (e.g., 10-by-10) by running `test_dd_nmrom.sh` in the `commands` directory, using the `inputs/test_dd_nmrom_10by10.json` input file.
