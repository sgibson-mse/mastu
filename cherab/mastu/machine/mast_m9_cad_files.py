
import os

from raysect.optical.spectralfunction import ConstantSF
from raysect.optical.material import Lambert
from raysect.optical.library.metal import RoughTungsten

try:
    CADMESH_PATH = os.environ['CHERAB_CADMESH']
except KeyError:
    if os.path.isdir('/projects/cadmesh/'):
        CADMESH_PATH = '/projects/cadmesh/'
    else:
        raise ValueError("CHERAB's CAD file path environment variable 'CHERAB_CADMESH' is not set.")

metal_roughness = 0.25
lambertian_absorption = 0.25
METAL = RoughTungsten(metal_roughness)
CARBON = Lambert(ConstantSF(lambertian_absorption))


BEAM_DUMPS = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-BEAM DUMPS + GDC.obj'), CARBON)
]

CENTRE_COLUMN = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-CENTRE COLUMN ARMOUR.obj'), CARBON)
]

LOWER_COIL_ARMOUR = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-LOWER COIL ARMOUR.obj'), CARBON)
]

LOWER_COILS = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-LOWER COILS.obj'), METAL)
]

LOWER_ELM_COILS = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-LOWER ELM COILS.obj'), METAL)
]

MAST_VESSEL = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-MAST VESSEL.obj'), METAL)
]

P3_COILS_LOW_RES = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9_P3_COILS_LOW_RES.obj'), METAL)
]

UPPER_COIL_ARMOUR = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-UPPER COIL ARMOUR.obj'), CARBON)
]

UPPER_COILS = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-UPPER COILS.obj'), METAL)
]

UPPER_ELM_COILS = [
    (os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-UPPER ELM COILS.obj'), METAL)
]

MAST_FULL_MESH = (
    BEAM_DUMPS + CENTRE_COLUMN + LOWER_COIL_ARMOUR + LOWER_COILS + LOWER_ELM_COILS
    + MAST_VESSEL + P3_COILS_LOW_RES + UPPER_COIL_ARMOUR + UPPER_COILS + UPPER_ELM_COILS
)
