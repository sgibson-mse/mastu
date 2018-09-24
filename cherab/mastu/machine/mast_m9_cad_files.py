
import os

try:
    CADMESH_PATH = os.environ['CHERAB_CADMESH']
except KeyError:
    if os.path.isdir('/projects/cadmesh/'):
        CADMESH_PATH = '/projects/cadmesh/'
    else:
        raise ValueError("CHERAB's CAD file path environment variable 'CHERAB_CADMESH' is not set.")

BEAM_DUMPS = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-BEAM DUMPS + GDC.obj')
]

CENTRE_COLUMN = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-CENTRE COLUMN ARMOUR.obj')
]

LOWER_COIL_ARMOUR = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-LOWER COIL ARMOUR.obj')
]

LOWER_COILS = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-LOWER COILS.obj')
]

LOWER_ELM_COILS = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-LOWER ELM COILS.obj')
]

MAST_VESSEL = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-MAST VESSEL.obj')
]

P3_COILS_LOW_RES = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9_P3_COILS_LOW_RES.obj')
]

UPPER_COIL_ARMOUR = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-UPPER COIL ARMOUR.obj')
]

UPPER_COILS = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-UPPER COILS.obj')
]

UPPER_ELM_COILS = [
    os.path.join(CADMESH_PATH, 'mast/mast-m9/MAST-M9-UPPER ELM COILS.obj')
]

MAST_FULL_MESH = (
    BEAM_DUMPS + CENTRE_COLUMN + LOWER_COIL_ARMOUR + LOWER_COILS + LOWER_ELM_COILS
    + MAST_VESSEL + P3_COILS_LOW_RES + UPPER_COIL_ARMOUR + UPPER_COILS + UPPER_ELM_COILS
)
