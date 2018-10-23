
from raysect.optical.observer import VectorCamera

from cherab.tools.observers import load_calcam_calibration


def load_camera(camera_id, world):

    if camera_id == "DIVCAM-SXD":
        camera_config = load_calcam_calibration('/home/mcarr/mastu/cameras/mug_divcam_sxd.nc')
        pixels_shape, pixel_origins, pixel_directions = camera_config
        camera = VectorCamera(pixel_origins, pixel_directions, parent=world)
        camera.spectral_bins = 15
        return camera

    elif camera_id == "DIVCAM-ISP":
        camera_config = load_calcam_calibration('/home/mcarr/mastu/cameras/mug_divcam_isp.nc')
        pixels_shape, pixel_origins, pixel_directions = camera_config
        camera = VectorCamera(pixel_origins, pixel_directions, parent=world)
        camera.spectral_bins = 15
        return camera

    elif camera_id == "BULLET-MIDPLANE":

        camera_config = load_calcam_calibration('/home/mcarr/mastu/cameras/mug_bulletb_midplane.nc')
        pixels_shape, pixel_origins, pixel_directions = camera_config
        camera = VectorCamera(pixel_origins, pixel_directions, parent=world)
        camera.spectral_bins = 15
        return camera

    else:
        raise ValueError("MAST-U camera ID field is invalid.")
