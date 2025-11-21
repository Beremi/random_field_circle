# Random field on a circular tunnel

Gaussian random fields are used to model log permeability around a tunnel. Two 1D Gaussian fields live on the tunnel wall: an amplitude field \(A(\theta)\) that lowers log k at the boundary and a decay-rate field \(\lambda(\theta)\) that controls how fast the rock recovers with distance. Both use a truncated Karhunen-Loeve expansion of the periodized exponential kernel on the circle; eigenvalues are
\[
  \lambda_n = \frac{2 \sigma^2 \ell}{1 + \ell^2 n^2}, \qquad n = 0,1,\dots
\]
The 2D log-permeability around a tunnel of radius \(R_t\) is
\[
  \log k(r, \theta) = \mu_{\log k} + A(\theta) \exp(-\lambda(\theta)[r - R_t]),
\]
with values masked inside the tunnel. A thin-plate spline morph can warp this circular field to a measured polygonal niche.

## Repository map
- `tunnel_random_field.py` - core KL utilities: eigenvalues for the periodized exponential kernel, sampling of \(A(\theta)\) and \(\lambda(\theta)\), construction of 2D log k fields, and plotting helpers. Running this file executes a minimal demo.
- `niche_geometry.py` - reads VTK/VTU boundaries (`cd-a_niche4.vtu`), extracts segments, and builds a thin-plate RBF map (`CircleToPolygonMorpher`) that warps circular grids to the measured niche.
- `demo.ipynb` - quick, code-only example of sampling the circle fields and plotting one 2D realization.
- `demo_polygon.ipynb` - inspects horizontal and vertical permeability logs, fits simple exponential depth profiles, sets priors for \(A\) and \(\lambda\), samples a realization, and warps it to the polygonal niche.
- `Bayes_circular_RF.ipynb` - Metropolis-Hastings inversion on the circular tunnel: priors on background log k and a lower floor, exponential likelihood for horizontal/vertical log-permeability profiles, MAP sample visualization, and warped posterior draws. Produces figures like `circular_fit.png` (depth-profile fit) and `circular_permeability.png` (2D log k map).
- `Bayes_square_exponential.ipynb` - same inference workflow but with a squared-exponential prior on the square \([-5,5]^2\) using the precomputed KL basis from `square_exponential_decomposition.h5`. Uses a `SquareExponentialField` helper to evaluate and sample fields.
- `square_exponential_decomposition.ipynb` - builds the separable squared-exponential Galerkin system on a Legendre basis, assembles the covariance, computes the KL modes, and exports them to HDF5.
- `square_exponential_decomposition.h5` - stored KL decomposition: 101-point grids on \([-5,5]^2\), 152 eigenvalues/eigenvectors, attributes reporting mean=-41, variance=16, and correlation length 0.75 m.
- `Bayes.ipynb` - small toy inversion with Gaussian bump basis functions on \([-5,5]^2\); illustrates sampling and prediction without the tunnel geometry.
- Figures: `circular_fit.png`, `circular_permeability.png`, plus intermediate plots inside the notebooks.

## Quick start
1) Install dependencies (Python 3.10+): `pip install numpy matplotlib scipy meshio h5py`.
2) Run the minimal circular example:
   ```bash
   python tunnel_random_field.py
   ```
   or open `demo.ipynb` to reproduce the same workflow in a notebook.
3) To work with the polygonal niche, ensure `cd-a_niche4.vtu` stays alongside the notebooks and run `demo_polygon.ipynb` or the circular Bayesian notebook.

## Bayesian workflows
- **Circular prior (`Bayes_circular_RF.ipynb`)**: builds priors for \(A(\theta)\), \(\lambda(\theta)\), background log k, and a log-floor parameter; fits measurement profiles along two rays with MH sampling; exports MAP fields and posterior samples; warps samples to the polygon using `CircleToPolygonMorpher`.
- **Square SE prior (`Bayes_square_exponential.ipynb`)**: loads the KL basis from `square_exponential_decomposition.h5`, augments it with a global background shift, and applies the same MH routine to the measurements; produces matched profile plots and 2D posterior draws (with optional niche outline).

## Data and geometry
- Measurement inputs live directly in the notebooks as depth-permeability arrays for horizontal and vertical lines.
- The niche boundary comes from `cd-a_niche4.vtu` (mixed line/line3 elements). Use `load_niche_segments` and `segments_to_unique_points` from `niche_geometry.py` to inspect or reuse the geometry.

## Tips
- The random field sampler assumes a unit-radius tunnel; adjust `R_tunnel` in `build_tunnel_logk_field` if you change the geometry.
- Posterior likelihoods are implemented in-place in the notebooks; tweak noise levels or priors near the top of each workflow cell block.
- When morphing to polygons, increase `n_boundary_ctrl` or `n_outer_ctrl` in `CircleToPolygonMorpher` if the warp needs to be smoother farther from the wall.
