"""
tunnel_random_field.py

KL-based random log-permeability fields around a circular tunnel.

- Explicit KL expansion for a Gaussian field on the circle with
  periodized exponential covariance.
- Construction of 2D log k(r, θ) from two circle fields:
    A(θ)   ... reduction at the tunnel wall
    λ(θ)   ... radial recovery rate
- Simple plotting utilities.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. KL eigenvalues and sampling on the circle
# ----------------------------------------------------------------------

def kl_eigenvalues_exponential_periodic(ell, sigma, n_modes, L=2*np.pi):
    """
    Eigenvalues λ_n for a zero-mean stationary Gaussian field on a circle
    of length L with periodized exponential covariance.

        C_∞(r) = σ² exp(-|r|/ℓ),   r ∈ ℝ
        C_p(τ) = ∑_m C_∞(τ + mL).

    The Fourier modes with wavenumbers k_n = 2π n / L are eigenfunctions
    and the eigenvalues are λ_n = S(k_n), where S is the spectral density
    of C_∞:

        S(k) = ∫_{ℝ} C_∞(r) e^{-ikr} dr = 2 σ² ℓ / (1 + ℓ² k²).

    Parameters
    ----------
    ell : float
        Correlation length ℓ.
    sigma : float
        Standard deviation σ of the field.
    n_modes : int
        Number of Fourier modes n = 0, …, n_modes for which to compute λ_n.
    L : float, optional
        Length of the circle, default 2π.

    Returns
    -------
    lam : (n_modes+1,) ndarray
        Eigenvalues λ_n, n = 0,…,n_modes.
    """
    n = np.arange(0, n_modes + 1)
    k_n = 2.0 * np.pi * n / L
    lam = 2.0 * sigma**2 * ell / (1.0 + (ell * k_n) ** 2)
    return lam


def sample_periodic_gaussian(ell, sigma, n_modes, n_theta, rng=None, L=2*np.pi):
    """
    Sample a zero-mean Gaussian random field on [0, L) with periodic
    boundary conditions and exponential covariance, using the truncated
    KL expansion in Fourier basis.

    Eigenfunctions (real basis):
        φ_0(θ)      = 1/√L,
        φ_n^cos(θ)  = √(2/L) cos(2π n θ / L),
        φ_n^sin(θ)  = √(2/L) sin(2π n θ / L).

    Parameters
    ----------
    ell : float
        Correlation length ℓ along the circle.
    sigma : float
        Standard deviation σ of the Gaussian field.
    n_modes : int
        Truncation index N; modes 0,…,N are kept.
    n_theta : int
        Number of points in the uniform θ-grid.
    rng : np.random.Generator, optional
        Random number generator. If None, a new default is created.
    L : float, optional
        Length of the circle, default 2π.

    Returns
    -------
    theta : (n_theta,) ndarray
        Uniform grid of angles in [0, L).
    u : (n_theta,) ndarray
        One realization of the Gaussian field evaluated on theta.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = np.linspace(0.0, L, n_theta, endpoint=False)

    # Eigenvalues λ_n
    lam = kl_eigenvalues_exponential_periodic(ell, sigma, n_modes, L=L)
    u = np.zeros_like(theta)

    # n = 0 term
    xi0 = rng.normal()
    phi0 = 1.0 / np.sqrt(L)
    u += np.sqrt(lam[0]) * xi0 * phi0

    # n >= 1 cosine/sine pairs
    for n in range(1, n_modes + 1):
        lam_n = lam[n]
        xi_c = rng.normal()
        xi_s = rng.normal()
        k = 2.0 * np.pi * n / L
        phi_c = np.sqrt(2.0 / L) * np.cos(k * theta)
        phi_s = np.sqrt(2.0 / L) * np.sin(k * theta)
        u += np.sqrt(lam_n) * (xi_c * phi_c + xi_s * phi_s)

    return theta, u


# ----------------------------------------------------------------------
# 2. Circle fields for the tunnel model: A(θ) and λ(θ)
# ----------------------------------------------------------------------

def sample_tunnel_circle_fields(
    n_theta=1024,
    n_modes_A=30,
    n_modes_lambda=30,
    ell_A=1.0,
    sigma_A_field=1.0,
    mu_A=-1.0,
    sigma_A=0.5,
    ell_lambda=1.5,
    sigma_lambda_field=1.0,
    lambda0=1.0,
    sigma_log_lambda=0.25,
    force_negative=True,
    rng=None,
):
    """
    Sample the two circle fields used in the tunnel model:

      - amplitude field A(θ): reduction of log k at the tunnel wall,
      - decay-rate field λ(θ): radial speed of recovery to background.

    Both are derived from underlying zero-mean Gaussian fields with
    exponential covariance on the circle via KL expansion:

        A(θ)      = μ_A + σ_A u_A(θ), then forced negative,
        λ(θ)      = exp(log λ_0 + σ_{logλ} u_λ(θ)).

    Parameters
    ----------
    n_theta : int
        Number of angular grid points.
    n_modes_A : int
        KL truncation for the amplitude field.
    n_modes_lambda : int
        KL truncation for the decay-rate field.
    ell_A, sigma_A_field : float
        Correlation length and std of the underlying u_A field.
    mu_A, sigma_A : float
        Mean and scaling of A(θ). Set `force_negative=False` to allow
        positive amplitudes.
    ell_lambda, sigma_lambda_field : float
        Correlation length and std of the underlying u_λ field.
    lambda0, sigma_log_lambda : float
        Mean and std in log-scale for λ(θ).
    force_negative : bool
        If True (default) enforce negative amplitudes to model damage.
        Set to False when data support an increase in log k near the wall.
    rng : np.random.Generator, optional
        RNG used for both fields.

    Returns
    -------
    theta : (n_theta,) ndarray
        Angular grid on [0, 2π).
    A_theta : (n_theta,) ndarray
        Amplitude field A(θ), typically negative.
    lambda_theta : (n_theta,) ndarray
        Positive decay-rate field λ(θ).
    """
    if rng is None:
        rng = np.random.default_rng()

    theta, u_A = sample_periodic_gaussian(
        ell_A, sigma_A_field, n_modes_A, n_theta, rng=rng
    )
    _, u_lambda = sample_periodic_gaussian(
        ell_lambda, sigma_lambda_field, n_modes_lambda, n_theta, rng=rng
    )

    # Amplitude (reduction at the wall)
    A_theta = mu_A + sigma_A * u_A
    if force_negative:
        A_theta = -np.abs(A_theta)  # enforce reduction (negative)

    # Radial decay-rate
    log_lambda_theta = np.log(lambda0) + sigma_log_lambda * u_lambda
    lambda_theta = np.exp(log_lambda_theta)

    return theta, A_theta, lambda_theta


# ----------------------------------------------------------------------
# 3. 2D log k field around a circular tunnel
# ----------------------------------------------------------------------

def build_tunnel_logk_field(
    theta_circle,
    A_theta,
    lambda_theta,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    nx=300,
    ny=300,
    R_tunnel=1.0,
    mu_logk=0.5,
    mask_inside=True,
):
    """
    Construct a 2D log-permeability field log k(x,y) around a circular
    tunnel of radius R_tunnel from given circle fields A(θ) and λ(θ).

    In polar coordinates (r, θ) with r ≥ R_tunnel we use
        log k(r, θ) = μ_logk + A(θ) exp(-λ(θ) (r - R_tunnel)).

    Inside the tunnel (r < R_tunnel) the field is masked with NaN by default.

    Parameters
    ----------
    theta_circle : (n_theta,) ndarray
        Angles at which A_theta and lambda_theta are defined (0,…,2π).
    A_theta : (n_theta,) ndarray
        Amplitude field on the circle.
    lambda_theta : (n_theta,) ndarray
        Decay-rate field on the circle (positive).
    xlim, ylim : tuple of float
        Domain extents [xmin, xmax], [ymin, ymax].
    nx, ny : int
        Number of grid points in x and y directions.
    R_tunnel : float
        Tunnel radius.
    mu_logk : float
        Background log-permeability of undisturbed rock.
    mask_inside : bool
        If True, set values for r < R_tunnel to NaN.

    Returns
    -------
    log_k : (ny, nx) ndarray
        2D grid of log k values.
    x, y : (nx,), (ny,) ndarray
        Grid coordinates such that log_k[j,i] corresponds to (x[i], y[j]).
    """
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    Theta[Theta < 0.0] += 2.0 * np.pi  # map to [0, 2π)

    # Interpolate circle fields onto the 2D angular grid (periodic)
    A_on_grid = np.interp(
        Theta.ravel(), theta_circle, A_theta, period=2.0 * np.pi
    ).reshape(Theta.shape)
    lambda_on_grid = np.interp(
        Theta.ravel(), theta_circle, lambda_theta, period=2.0 * np.pi
    ).reshape(Theta.shape)

    log_k = np.full_like(R, np.nan, dtype=float)

    mask_outside = R >= R_tunnel
    dr = R[mask_outside] - R_tunnel
    log_k[mask_outside] = mu_logk + A_on_grid[mask_outside] * np.exp(
        -lambda_on_grid[mask_outside] * dr
    )

    if not mask_inside:
        # Optionally fill inside with the background
        log_k[~mask_outside] = mu_logk

    return log_k, x, y


# ----------------------------------------------------------------------
# 4. Plotting helpers
# ----------------------------------------------------------------------

def plot_circle_field(theta, field, label=None, ax=None):
    """
    Plot a 1D field defined along the circle as a function of θ.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(theta, field)
    ax.set_xlabel(r"$\theta$")
    if label is not None:
        ax.set_ylabel(label)
    ax.grid(True, linestyle=":")
    return ax


def plot_circle_field_as_radius(theta, field, ax=None):
    """
    Plot a 1D field on the circle as a radial deformation of a unit circle.

    The radius is a shifted/rescaled version of the field so the plot
    is purely qualitative.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Normalize to [0,1] and shift around radius=1
    f = field
    f_min, f_max = np.min(f), np.max(f)
    if f_max > f_min:
        f_norm = (f - f_min) / (f_max - f_min)
    else:
        f_norm = np.zeros_like(f)
    r = 1.0 + 0.3 * (f_norm - 0.5)  # ±0.15 variation

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, label="field pattern")
    # Reference circle
    t = np.linspace(0, 2.0 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), linestyle="--", label="unit circle")
    ax.set_aspect("equal", "box")
    ax.legend()
    return ax


def plot_logk_field(log_k, x, y, title=None, ax=None):
    """
    Visualize a 2D log k field on a rectangular grid with a circular tunnel.
    """
    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(
        log_k,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max() ],
        interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("log k")

    # Plot tunnel boundary at radius 1
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal", "box")
    return ax


# ----------------------------------------------------------------------
# 5. Minimal example of usage
# ----------------------------------------------------------------------

def demo():
    """
    Example usage:

    1) Sample amplitude and decay-rate fields on the circle.
    2) Plot them in 1D and as deformed circles.
    3) Build a 2D log k field around a tunnel and plot it.
    """
    rng = np.random.default_rng(123)

    # --- 1D circle fields ---
    theta, A_theta, lambda_theta = sample_tunnel_circle_fields(
        n_theta=1024,
        n_modes_A=20,
        n_modes_lambda=20,
        ell_A=0.6,
        sigma_A_field=1.0,
        mu_A=-1.0,
        sigma_A=0.5,
        ell_lambda=1.5,
        sigma_lambda_field=1.0,
        lambda0=1.0,
        sigma_log_lambda=0.25,
        rng=rng,
    )

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    plot_circle_field(theta, A_theta, label=r"$A(\theta)$", ax=ax1)
    ax1.set_title("Amplitude field along tunnel")
    plot_circle_field_as_radius(theta, A_theta, ax=ax2)
    ax2.set_title("Amplitude pattern as deformed circle")
    fig1.tight_layout()

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))
    plot_circle_field(theta, lambda_theta, label=r"$\lambda(\theta)$", ax=ax3)
    ax3.set_title("Decay-rate field along tunnel")
    plot_circle_field_as_radius(theta, lambda_theta, ax=ax4)
    ax4.set_title("Decay-rate pattern as deformed circle")
    fig2.tight_layout()

    # --- 2D log k field ---
    log_k, x, y = build_tunnel_logk_field(
        theta_circle=theta,
        A_theta=A_theta,
        lambda_theta=lambda_theta,
        xlim=(-3.0, 3.0),
        ylim=(-3.0, 3.0),
        nx=400,
        ny=400,
        R_tunnel=1.0,
        mu_logk=0.5,
        mask_inside=True,
    )

    fig3, ax5 = plt.subplots(figsize=(6, 5))
    plot_logk_field(log_k, x, y, title="Random log k around circular tunnel", ax=ax5)
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    demo()
