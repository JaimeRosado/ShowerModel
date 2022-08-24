def test_telescope():
    from showermodel.telescope import Telescope

    # Default telescope type: IACT
    # Default angular aperture in degrees: 8.0
    # Default detection area in m2: 113.097
    telescope = Telescope()
    assert telescope.tel_type == 'generic'
    assert telescope.apert == 10.0
    assert telescope.area == 100.0
