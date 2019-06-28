from raysect.optical import Point3D, Vector3D

def load_mse_sightlines():
    los = Point3D(-0.949, -2.228,  0)
    los_vector = Vector3D(0.75, 0.662, 0).normalise()
    # # los = los - los_vector * 3


    # los = Point3D(-8, -1,  0)
    # los_vector = Vector3D(1, 0, 0).normalise()

    return (los, los_vector)