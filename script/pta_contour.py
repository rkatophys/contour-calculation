
import os
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg
import jax
import jax.numpy as jnp
import deterministic_jnp

class DummyPsr:
    def __init__(
        self,
        psr_name,
        psr_radec,
        psr_seed,
        period=None,
        cadence=None,
        pdist=None,
        pdist_sigma=None,
        white_sigma=None,
        white_seed=None,
    ):
        """
        A class representing a pulsar with basic parameters and simulated observation data.

        Args:
            psr_name: Name of the pulsar
            psr_radec: Pulsar's (RA, Dec) in radians as a tuple (ra, dec)
            psr_seed: Random seed for pulsar-specific noise generation
            period: Total observation time in years
            cadence: Observation cadence in weeks
            pdist: Distance to the pulsar [kpc]
            pdist_sigma: Uncertainty in distance [kpc]
            white_sigma: Standard deviation of white noise [s]
            white_seed: Random seed used to generate white noise
        """
        self._name = psr_name
        self._psr_seed = psr_seed
        self._white_seed = white_seed
        self._white_sigma = white_sigma
        self._distance = (pdist, pdist_sigma)

        # Generate observation times (TOAs)
        obstime = period * 365.25 * 86400.0  # seconds
        obsnum = int(obstime / (cadence * 7 * 86400.0))  # number of samples
        self._toas = np.linspace(0, obstime, obsnum)

        # Convert RA/Dec to spherical coordinates (theta, phi)
        dec, ra = psr_radec[1], psr_radec[0]
        ptheta = np.pi / 2 - dec
        pphi = ra
        self._pixels = (ptheta, pphi)

        self._raj = pphi
        self._decj = dec

        # Unit vector pointing toward the pulsar
        self._pos = np.array([
            np.cos(self._raj) * np.cos(self._decj),
            np.sin(self._raj) * np.cos(self._decj),
            np.sin(self._decj),
        ])

        # Generate white noise data
        np.random.seed(self._white_seed + self._psr_seed)
        self._whitedata = self._white_sigma * np.random.randn(obsnum)

    @property
    def psr_seed(self):
        """The random seed assigned to the pulsar (used for noise generation)."""
        return self._psr_seed

    @property
    def name(self):
        """Pulsar name."""
        return self._name

    @property
    def toas(self):
        """Pulse times of arrival (TOAs) in seconds."""
        return self._toas

    @property
    def distance(self):
        """Tuple of pulsar distance [kpc] and its uncertainty."""
        return self._distance

    @property
    def pixels(self):
        """Spherical coordinates (theta, phi) [rad], used in Healpix-based calculations."""
        return self._pixels

    @property
    def radec(self):
        """Equatorial coordinates (RA, Dec) in radians."""
        return self._raj, self._decj

    @property
    def pos(self):
        """Unit vector pointing toward the pulsar."""
        return self._pos

    @property
    def white_sigma(self):
        """Standard deviation of white noise [s]."""
        return self._white_sigma

    @property
    def whitedata(self):
        """Simulated white noise data."""
        return self._whitedata


@jax.jit
def assign_func(assign_params, index, toas, pos):
    """
    Generate a CW signal using assigned parameters and pulsar data.

    Args:
        assign_params: 1D array of waveform parameters in the following order:
            [gwtheta, gwphi, inc, log10_mc, log10_fgw, log10_dist, phase0, psi, pdist]
        index: Index of the pulsar (used to select the correct TOAs)
        toas: Array of TOAs for all pulsars
        pos: Position vector of the pulsar (unit vector)

    Returns:
        signal: The calculated CW signal (delay)
    """
    params_jax_dict = {
        "gwtheta": assign_params[0],
        "gwphi": assign_params[1],
        "inc": assign_params[2],
        "log10_mc": assign_params[3],
        "log10_fgw": assign_params[4],
        "log10_dist": assign_params[5],
        "phase0": assign_params[6],
        "psi": assign_params[7],
        "pdist": [assign_params[8], 0.0],
    }

    signal = deterministic_jnp.cw_delay(
        toas[index],
        pos,
        **params_jax_dict,
        log10_h=None,
        psrTerm=True,
        p_dist=0.0,
        p_phase=None,
        evolve=True,
        phase_approx=False,
        check=False,
        tref=0.0,
    )

    return signal


def loop_func(angnum, psrs, thetas, phis, white_sigma, folder, calc_snr):
    """
    Calculates the posterior mean and covariance of gravitational wave (GW) parameters
    at a given sky location using a linearized (Gaussian) model approximation.
    (Optional) Computes the approximate signal-to-noise ratio (SNR^2) .

    Args:
        angnum (int): Index of the sky pixel (into thetas, phis).
        psrs (list): List of DummyPsr objects, each representing a pulsar.
        thetas (np.ndarray): Array of polar angles θ for each pixel.
        phis (np.ndarray): Array of azimuthal angles φ for each pixel.
        white_sigma (float): Standard deviation of white noise [seconds].
        folder (str): Directory to save output files.
        calc_snr (bool): If True, compute and save the squared SNR.
    """
    # Set GW source parameters for the current pixel (angnum)
    params_func_dict = {
        "gwtheta": thetas[angnum],  # GW source polar angle
        "gwphi": phis[angnum],      # GW source azimuthal angle
        "inc": 1.0,                 # inclination angle cosine
        "log10_mc": 9.0,            # log10 of chirp mass
        "log10_fgw": -8.0,          # log10 of GW frequency
        "log10_dist": 2.0,          # log10 of distance to GW source
        "phase0": 1.0,              # initial GW phase
        "psi": 1.0,                 # polarization angle
    }

    sigpsrs = []
    for psr in psrs:
        # Combine GW parameters and pulsar distance for gradient calculation
        params = list(params_func_dict.values()) + [psr.distance[0]]

        # Convert TOAs and positions to JAX arrays for JIT compatibility
        jtoas = jnp.array(psr.toas)
        jpos = jnp.array(psr.pos)

        # Define a function to compute the gradient of the signal wrt parameters at each TOA index
        def assign_index(index):
            grad_signal = jax.grad(assign_func)  # gradient function of assign_func
            return grad_signal(params, index, jtoas, jpos)

        # Vectorize gradient computation over all TOA indices for the pulsar
        sigpsr = jax.vmap(assign_index)(jnp.arange(len(psr.toas)))
        sigpsrs.append(sigpsr)

    gradarr = jnp.array(sigpsrs)  # shape: (num_psrs, num_toas, num_params+1)

    # Extract gradients for all but the last TOA (exclude pulsar distance parameter gradients)
    grad_e_arrs = [gradarr[i, :-1, :] for i in range(len(psrs))]
    grad_e = jnp.concatenate(grad_e_arrs, axis=1)  # concatenate along TOAs axis

    # Extract gradients for the last TOA (pulsar distance parameter), pad so alignment matches overall shape
    grad_p_arrs = [
        jnp.pad(
            gradarr[i, -1, :],  # gradients for pulsar distance parameter
            (len(psr.toas) * i, len(psr.toas) * (len(psrs) - i - 1)),
            mode="constant",
        )
        for i, psr in enumerate(psrs)
    ]
    grad_p = jnp.array(grad_p_arrs)

    # Combine gradients of extrinsic parameters and pulsar distance parameters into a single array
    grads = jnp.concatenate((grad_e, grad_p))

    # Collect pulsar distance measurement uncertainties (sigma)
    pdist_sigmas_list = [psr.distance[-1] for psr in psrs]
    pdist_sigmas = jnp.array(pdist_sigmas_list)

    numparams_e = len(params_func_dict)  # number of extrinsic GW parameters
    # Construct inverse covariance matrix: signal + noise (white noise variance + distance uncertainties)
    covinv = (
        jnp.dot(grads, grads.T) / white_sigma**2
        + jnp.diag(jnp.concatenate((jnp.zeros(numparams_e), 1.0 / pdist_sigmas**2)))
    )

    # Invert to get covariance matrix
    covjax = jnp.linalg.inv(covinv)

    # Combine white noise residuals from all pulsars
    noise = jnp.concatenate([psr.whitedata for psr in psrs])

    # Compute weighted sum of noise projected onto gradients
    nhmat = jnp.dot(noise, grads.T) / white_sigma**2

    # Compute mean parameter estimates (including distance sigmas)
    meanjax = covjax @ nhmat + jnp.array(list(params_func_dict.values()) + pdist_sigmas_list)

    # Convert to numpy arrays for saving
    cov = np.array(covjax)
    mean = np.array(meanjax)

    # Create output folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # Save mean and covariance to text files with angle number suffix
    np.savetxt(f"{folder}/mean_{angnum}.txt", mean)
    np.savetxt(f"{folder}/cov_{angnum}.txt", cov)

    if calc_snr:
        snr2s = []
        for psr in psrs:
            # Build full parameter dict including pulsar distance
            params_full_dict = params_func_dict | {"pdist": [psr.distance[0], 0.0]}

            # Compute gravitational wave timing residuals for the pulsar
            gwdata = deterministic_jnp.cw_delay(
                psr.toas,
                psr.pos,
                **params_full_dict,
                log10_h=None,
                psrTerm=True,
                p_dist=0.0,
                p_phase=None,
                evolve=True,
                phase_approx=False,
                check=False,
                tref=0.0,
            )

            # Update pulsar residuals including white noise
            psr.residuals = gwdata + psr.whitedata

            # Calculate squared signal-to-noise ratio for the pulsar
            psr.snr2 = np.sum((gwdata / psr.white_sigma) ** 2)
            snr2s.append(psr.snr2)

        # Save all pulsars' SNR^2 values to a file
        np.savetxt(f"{folder}/snr2_{angnum}.txt", np.array(snr2s))


def ang_rot(phi, theta):
    # Rotate angles for plotting purposes
    rotphi = -(phi - np.pi)
    rottheta = np.pi / 2.0 - theta
    return rotphi, rottheta

def prob_chisq(prob):
    # Convert a cumulative probability to the corresponding
    # value of the chi-square distribution's quantile for 2 degrees of freedom
    chisqcdf = np.sqrt(-2.0 * np.log(1 - prob))
    return chisqcdf
def get_ellipse_params_from_cov(cov):
    """
    Perform SVD on a 2D covariance matrix to obtain ellipse width, height, and orientation angle.

    Args:
        cov (np.ndarray): 2×2 covariance matrix.

    Returns:
        tuple: (width, height, angle_in_degrees)
    """
    U, s, _ = linalg.svd(cov)
    orientation = np.atan2(U[1, 0], U[0, 0])  # orientation in radians
    width = np.sqrt(s[0])
    height = np.sqrt(s[1])
    angle = np.degrees(orientation)
    return width, height, angle
