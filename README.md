# DD-NM-ROM
DD-NM-ROM integrates nonlinear-manifold reduced order models (NM-ROMs) with
domain decomposition (DD). NM-ROMs approximate the full order model (FOM) state
in a nonlinear-manifold by training a shallow, sparse autoencoder using FOM
snapshot data. These NM-ROMs can be advantageous over linear-subspace ROMs
(LS-ROMs) for problems with slowly decaying Kolmogorov-width. However, the
number of NM-ROM parameters that need to be trained scales with the size of the
FOM. Moreover, for "extreme-scale" problems, the storage of high-dimensional
FOM snapshots alone can make ROM training expensive. To alleviate the training
cost, DD-NM-ROM applies DD to the FOM, computes NM-ROMs on each subdomain, and
couples them to obtain a global NM-ROM. This approach has several advantages:
subdomain NM-ROMs can be trained in parallel, involve fewer parameters to be
trained than global NM-ROMs, require smaller subdomain FOM dimensional training
data, and can be tailored to subdomain-specific features of the FOM. The
shallow, sparse architecture of the autoencoder used in each subdomain NM-ROM
allows application of hyper-reduction (HR), reducing the complexity caused by
nonlinearity and yielding computational speedup of the NM-ROM. DD-NM-ROM
provides the first application of NM-ROM (with HR) to a DD problem. In
particular, DD-NM-ROM implements an algebraic DD reformulation of the FOM,
training a NM-ROM with HR for each subdomain, and a sequential quadratic
programming (SQP) solver to evaluate the coupled global NM-ROM. The proposed 
DD-NM-ROM approach is numerically tested for the Burgers' equation.


## Installation
Follow the instructions in `conda/README.md`.


## Examples
Explore the DD-NM-ROM approach for steady and unsteady problems for the Burgers' equation. See the `examples` directory.


## Citations
```
@article{Diaz_CMAME_2024,
  title = {A fast and accurate domain decomposition nonlinear manifold reduced order model},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume = {425},
  pages = {116943},
  year = {2024},
  issn = {0045-7825},
  doi = {https://doi.org/10.1016/j.cma.2024.116943},
  url = {https://www.sciencedirect.com/science/article/pii/S0045782524001993},
  author = {Alejandro N. Diaz and Youngsoo Choi and Matthias Heinkenschloss}
}
@phdthesis{Diaz_Thesis_2024,
  title = {Domain decomposition-based reduced-order models using nonlinear-manifolds and interpolatory projections},
  author = {Alejandro N. Diaz},
  year = {2024},
  month = {April},
  school = {Rice University},
  type = {PhD thesis},
  url = {https://hdl.handle.net/1911/116152}
}
```


## Contributors
- Alejandro Diaz (Sandia National Laboratories)
- Ivan Zanardi (University of Illinois Urbana-Champaign)
- Seung Whan Chung (Lawrence Livermore National Laboratory)
- Youngsoo Choi (Lawrence Livermore National Laboratory)
- Matthias Heinkenschloss (Rice University)


## Aknowledgement
- A. Diaz was supported for this work by a Defense Science and Technology
Internship (DSTI) at Lawrence Livermore National Laboratory and a 2021 National
Defense Science and Engineering Graduate Fellowship, United States.  
- I. Zanardi was supported for this work by a Data Science Summer Insitute (DSSI) 
at Lawrence Livermore National Laboratory.
- Y. Choi was supported for this work by the US Department of Energy under the
Mathematical Multifaceted Integrated Capability Centers -- DoE Grant DE --
SC0023164; The Center for Hierarchical and Robust Modeling of Non-Equilibrium
Transport (CHaRMNET).


## License
DD-NM-ROM is distributed under the terms of the MIT license. All new contributions must be made under the MIT. See
[LICENSE-MIT](https://github.com/LLNL/DD-NM-ROM/blob/main/LICENSE)

LLNL Release Nubmer: LLNL-CODE-864366
