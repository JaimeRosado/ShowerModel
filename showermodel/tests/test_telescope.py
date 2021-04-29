def test_telescope():
    from showermodel import Telescope

    # Default telescope type: IACT
    # Default angular aperture in degrees: 8.0
    # Default detection area in m2: 113.097
    telescope = Telescope(tel_type='IACT')
    assert telescope.tel_type == 'IACT'
    assert telescope.apert == 8.0
    assert telescope.area == 113.097

