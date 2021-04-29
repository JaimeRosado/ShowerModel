def test_shower():
    from showermodel import Shower
    theta = 20.0
    # Check the value obtained with the settings below is the following
    h_top = 112.8292
    shower = Shower(1.e6, theta=theta, az=45.0, x0=0.1, y0=0.2)
    assert shower.alt == 90.0 - theta
    assert shower.az == 45.0
    assert shower.h_top == h_top
