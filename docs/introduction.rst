.. _introduction:

Introduction
************

Very-high-energy cosmic rays and gamma rays induce extensive air showers (EAS) when entering the atmosphere.
Cherenkov and fluorescence light emitted by secondary charged particles is used as a proxy for studying the
primary particles that initiate the particle cascades.

Design, calibration and data analysis of cosmic-ray and gamma-ray observatories strongly rely on Monte Carlo
simulations of both the air shower and detector response.
CORSIKA program [Heck1998]_ is widely used for carrying out the first step of the simulation
whereas the second step depends on the detection technique.
For example, in the case of imaging atmospheric Cherenkov telescopes (IACT), the program `sim_telarray`
[Bernlohr2008]_ is commonly used. These detailed simulations are currently very demanding computationally.

`ShowerModel` is a python package to easily and quickly compute both Cherenkov and fluorescence light production
in air showers and its detection. This tool can be used to speed up some studies that do not require a full
simulation as well as to cross-check complex Monte Carlo simulations and data analyses.



