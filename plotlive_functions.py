"""Functions for live plotting 
"""
import numpy as np

def Rabi_osc(x, Omega, Delta, phi, Amplitude, offset, decay):
    return offset + Amplitude * np.sin((Omega**2 + Delta**2)**0.5 * x + phi) * np.exp(-decay * x)