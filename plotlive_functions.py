"""Functions for live plotting 
"""
import numpy as np

def rabi_osc(t, rabi_f, delta, phi, amplitude, offset, decay_time):
    """Calculate a decaying Rabi oscillation signal.

    Args:
        t: Time or array of times at which to evaluate the signal.
        rabi_f: Rabi frequency.
        delta: Detuning from resonance.
        phi: Oscillation phase in radians.
        amplitude: Signal amplitude.
        offset: Baseline signal offset.
        decay_time: Exponential decay time.

    Returns:
        The modeled signal at each time in ``t``.
    """
    f_rabi_eff = np.sqrt(rabi_f**2 + delta**2)
    return offset + amplitude * np.sin(2 * np.pi * f_rabi_eff * t / 2 + phi)**2 * np.exp(- t / decay_time)

def ramsey_spectro(f, offset, amplitude, f_rabi, f_resonance, T):
    """Calculate the Ramsey spectroscopy signal for the given frequencies.

    Args:
        f: Frequency or array of frequencies to evaluate.
        offset: Baseline signal offset.
        amplitude: Signal amplitude.
        f_rabi: Rabi frequency during the pulses.
        f_resonance: Resonance frequency.
        T: Free-evolution time between the pulses.

    Returns:
        The modeled Ramsey signal at each frequency in ``f``.
    """
    tau = 1 / (4 * f_rabi)                       # pi/2 pulse
    delta = f - f_resonance
    f_eff = np.sqrt(delta**2 + f_rabi**2)
    a = np.pi * f_eff * tau                      # Omega_eff * tau / 2
    b = np.pi * delta * T                        # delta * T / 2 (angular)
    p = 4 * (f_rabi / f_eff)**2 * np.sin(a)**2 * (np.cos(a) * np.cos(b) - delta / f_eff * np.sin(a) * np.sin(b))**2
    return offset + amplitude * p