"""
This script reads the result of a SOLPS simulation and calculates the
total radiated power. It could be used for example to benchmark the
performance of the bolometer system against simulations.
"""
import numpy as np
from raysect.optical import World

from cherab.core.atomic import carbon, deuterium, Isotope
from cherab.solps import load_solps_from_mdsplus
from cherab.openadas import OpenADAS

def no_radiation(*_):
    """Placeholder for when rates are not found"""
    return 0

world = World()
adas = OpenADAS(permit_extrapolation=True)  # create atomic data source


SPECIES_MAP = {'D0': (deuterium, 0), 'D+1': (deuterium, 1),
               'C0': (carbon, 0), 'C+1': (carbon, 1), 'C+2': (carbon, 2), 'C+3': (carbon, 3),
               'C+4': (carbon, 4), 'C+5': (carbon, 5), 'C+6': (carbon, 6)}


mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
ref_number = 69636
sim = load_solps_from_mdsplus(mds_server, ref_number)
mesh = sim.mesh
plasma = sim.create_plasma()
plasma.parent = world
plasma.atomic_data = adas

sim_te = sim.electron_temperature
sim_ne = sim.electron_density


total_rad_data = sim.total_radiation
vol = mesh.vol

solps_total_power = 0
for i in range(mesh.nx):
    for j in range(mesh.ny):
        solps_total_power += total_rad_data[i, j] * vol[i, j]

cherab_total_power = 0
b2_neutral_i = 0  # counter for B2 neutrals
for k, species in enumerate(sim.species_list):

    element, charge = SPECIES_MAP[species]
    if isinstance(element, Isotope):
        element = element.element
    if element.atomic_number == charge:
        continue

    try:
        plt = adas.line_radiated_power_rate(element, charge)
    except RuntimeError as ex:
        print("Line radiation error:", ex)
        plt = no_radiation
    try:
        prb = adas.continuum_radiated_power_rate(element, charge+1)
    except RuntimeError as ex:
        print("Continuum radiation error:", ex)
        prb = no_radiation

    if isinstance(sim.b2_neutral_densities, np.ndarray) and charge == 0:
        imp_dens = sim.b2_neutral_densities[:, :, b2_neutral_i]
        b2_neutral_i += 1
    else:
        imp_dens = sim.species_density[:, :, k]

    upper_density = sim.species_density[:, :, k+1]

    species_total_power = 0
    for i in range(mesh.nx):
        for j in range(mesh.ny):
            ne = sim_ne[i, j]
            te = sim_te[i, j]
            nimp = imp_dens[i, j]
            n_upper = upper_density[i, j]
            species_total_power += (plt(ne, te) * ne * nimp + prb(ne, te) * ne * n_upper) * vol[i, j]

    print("{} - {} - {:.4G} W".format(element.name, charge, species_total_power))
    cherab_total_power += species_total_power

print("SOLPS total radiated power => {:.4G} W".format(solps_total_power))
print("CHERAB total radiated power => {:.4G} W".format(cherab_total_power))