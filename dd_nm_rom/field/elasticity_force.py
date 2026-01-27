import numpy as np

from typing import List, Union
# Assuming imports from dd_nm_rom package
from dd_nm_rom import ops
from dd_nm_rom.elements import mesh as mesh_mod
from dd_nm_rom.elements import bound_cond as bc_mod

# Assuming BasicField is defined in .basic
from .basic import BasicField


class ElasticityForce(BasicField):
    """
    Defines the external force (body force) field for the 2D Linear Elasticity
    problem, generating independent x- (fx) and y- (fy) components.
    The force is defined by independent magnitudes per subdomain for fx and fy,
    and shared global sine wave frequencies.
    """

    # Initialization
    # ===================================
    def __init__(
        self,
        mesh: mesh_mod.MeshDD,
        # Now uses two sets of magnitude limits for x and y forces
        mu_lim_x: List[float] = [0.9, 1.1],
        mu_lim_y: List[float] = [0.9, 1.1],
        forced_config: Union[List[int], np.ndarray, None] = None,
        bc_type: str = "periodic"
    ) -> None:
        super(ElasticityForce, self).__init__(mesh)
        bc_mod.check_bc_type(bc_type)
        self.bc_type = bc_type
        # Store separate limits for force magnitudes
        self.mu_lim_x = mu_lim_x
        self.mu_lim_y = mu_lim_y
        self.configs = None
        self.forced_config = forced_config
        if (self.forced_config is not None):
            self.forced_config = np.array(self.forced_config).reshape(-1)

    # Design space
    # ===================================
    def _init_design_space(self) -> None:
        """
        Initializes the design space for force parameters.
        The design space is now larger: [Config, Magnitudes_x, Magnitudes_y, Freq_x, Freq_y]
        """
        # Define possible combinations, exclude an array of all zeros
        # Cretae 15 x 4 array for 4 subdomains for example
        self.configs = ops.generate_combs([np.arange(2)]*self.mesh.n_sub)[1:]
        if (self.forced_config is not None):
            self.configs += self.forced_config.reshape(1,-1)
            self.configs = self.configs.astype(bool).astype(int)
            self.configs = np.unique(self.configs, axis=0)
        
        # Define design space:
        # [0] Config index
        # [1:1+n_sub] Force Magnitudes for x-displacement (fx)
        # [1+n_sub:1+2*n_sub] Force Magnitudes for y-displacement (fy)
        # [-2:] Frequencies (freq_x, freq_y)
        self.design_space = (
            [[0,len(self.configs)]] + 
            [self.mu_lim_x]*self.mesh.n_sub + 
            [self.mu_lim_y]*self.mesh.n_sub + 
            [[2, 6]]*2
        )
        self.design_space = np.array(self.design_space).T

    def sample_design_space(self) -> np.ndarray:
        """Samples the design space for a single set of parameters."""
        # Force Magnitudes for x (fx)
        force_magnitudes_x = np.random.uniform(*self.mu_lim_x, size=self.mesh.n_sub)
        # Force Magnitudes for y (fy)
        force_magnitudes_y = np.random.uniform(*self.mu_lim_y, size=self.mesh.n_sub)

        # Apply the zero-mean constraint (if necessary, here applied only to fx for simplicity)
        force_magnitudes_x[3] = force_magnitudes_x[1] + force_magnitudes_x[2] - force_magnitudes_x[0]

        # Generate frequency parameters (integer and even values between 2 and 6)
        # Note: The original code used np.random.randint(1, 2) which only yields 2. 
        # I'm adjusting to make it a proper range (2, 4, 6)
        freq_x = 2 * np.random.randint(1, 4)
        freq_y = 2 * np.random.randint(1, 4)

        mu = np.concatenate([force_magnitudes_x, force_magnitudes_y, [freq_x, freq_y]])
        return mu

    def construct_design_mat(
        self,
        n_samples: int
    ) -> np.ndarray:
        """Generates a Latin Hypercube sample matrix and enforces constraints."""
        # Generate Latin Hypercube samples
        dmat = super(ElasticityForce, self).construct_design_mat(n_samples)
        
        # Check if the matrix size matches the expected size for 4 subdomains (1 + 4 + 4 + 2 = 11 columns)
        if dmat.shape[1] == 11:
            # Enforce the zero-mean constraint (here applied to fx, column 4)
            # col[4] = col[2] + col[3] - col[1]
            dmat[:, 4] = dmat[:, 2] + dmat[:, 3] - dmat[:, 1]
            # No constraint applied to fy (columns 5-8) for simplicity
            
            # Clip back to the valid range for fx
            min_val_x, max_val_x = self.mu_lim_x
            dmat[:, 4] = np.clip(dmat[:, 4], min_val_x, max_val_x)
            
        return self._convert_dmat_to_mu(dmat)

    def _convert_dmat_to_mu(
        self,
        dmat: np.ndarray
    ) -> np.ndarray:
        """Converts the sampled design matrix back to the parameter vector (mu)."""
        cfg = np.floor(dmat[:,0]).astype(np.int32)
        # Ensure the last two columns (frequencies) are integer and even
        dmat[:, -2:] = (2 * np.round(dmat[:, -2:] / 2)).astype(np.int32)
        # Return all columns except the config index
        return dmat[:,1:]

    def set_params(
        self,
        mu: np.ndarray
    ) -> None:
        """Sets the internal parameters (magnitudes for fx/fy and frequencies)."""
        mu = mu.reshape(-1)
        n_sub = self.mesh.n_sub
        
        # First n_sub elements are magnitudes for fx (x-force)
        self.mu_x = mu[:n_sub]
        # Next n_sub elements are magnitudes for fy (y-force)
        self.mu_y = mu[n_sub:2*n_sub]
        
        # Last two elements are global frequencies
        self.freq_x = mu[-2]
        self.freq_y = mu[-1]

    # Force fields
    # ===================================
    def u(self) -> np.ndarray:
        """Generates the x-component of the force field (fx)."""
        return self.generate_field(self.mu_x)

    def v(self) -> np.ndarray:
        """Generates the y-component of the force field (fy)."""
        return self.generate_field(self.mu_y)

    def generate_field(
        self,
        mu_magnitudes: np.ndarray
    ) -> np.ndarray:
        """
        Generates the force field (fx or fy) based on the provided magnitudes.
        """
        f = np.zeros(self.mesh.nxy)
        x, y = self.mesh.nodes_val.T
        for (i, mu_i) in enumerate(mu_magnitudes):
            ind = self.mesh.res_nodes[i]
            f[ind] = self._phi(x[ind], y[ind], mu_i)
        return f.reshape(self.mesh.n["y"], self.mesh.n["x"])

    def _phi(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray
    ) -> np.ndarray:
        """
        The fundamental forcing function (e.g., a 2D sine wave).
        """
        # A simple 2D sine wave forcing function
        return mu * np.sin(self.freq_x * np.pi * x) * np.sin(self.freq_y * np.pi * y)

    def get_force(
        self,
        mu: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """
        Returns the concatenated DD force vector: [fx, fy]
        """
        if (mu is not None):
            self.set_params(mu)
        # Concatenate the x-force (fx) and y-force (fy) into the expected vector format
        fx_vec = self.u().reshape(-1)
        fy_vec = self.v().reshape(-1)
        return np.concatenate([fx_vec, fy_vec])
