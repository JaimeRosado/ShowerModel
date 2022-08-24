import numpy as np

def test_shower():
    from showermodel import Shower
    theta = 20.0
    # Check the value obtained with the settings below is the following
    h_top = 112.8292
    shower = Shower(1.e6, theta=theta, az=45.0, x0=0.1, y0=0.2, atm_model=17)
    assert shower.alt == 90.0 - theta
    assert shower.az == 45.0
    assert shower.h_top == h_top
    assert np.isclose(shower.atmosphere.P[100], 49.69842905334434)
    assert np.isclose(shower.track.t[200], 254.50685251195043)
    assert np.isclose(shower.profile.s[10], 1.6504072224436146)
    assert np.isclose(shower.cherenkov.N_ph[50], 757147.620313149)
