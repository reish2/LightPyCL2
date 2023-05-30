# Copyright (C) 2023 Reish2
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Python built-in modules
import os
from time import time
from typing import List

# External modules
import numpy as np

from utils import iterative_tracer as it, geo_optical_elements as goe
from utils.Pose import Pose
# Custom modules
from utils.light_source import LightSource

# Enable PyOpenCL compiler output
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def setup_mesh_elements() -> List[goe.GeoObject]:
    """Set up mesh elements for the scene.

    Returns
    -------
    List[goe.Mesh]
        List of mesh elements.
    """
    # Create an instance of optical elements to generate meshes for the scene
    oe = goe.OpticalElements()

    # Setup a hemisphere to measure the light sources spatial power distribution
    measure_surf = oe.hemisphere(center=[0, 0, 0], radius=1000.0)
    measure_surf.set_material(mat_type="measure")
    meshes = [measure_surf]

    m2 = oe.lens_spherical_biconcave(focus=(0, 0, 0), r1=60., r2=6000., diameter=50.0, IOR=1.5)
    m2.transform(Pose.from_axis_angle(np.array([0, 1, 0]) * -np.pi / 2.0, np.array([0, 0, 0])))
    meshes.append(m2)

    return meshes


def main():
    start_time = time()

    # Amount of rays to trace
    ray_count = 1000

    # Tracer termination conditions. If either is reached, the tracer will stop tracing and return all results collected until that point.
    iterations = 16  # If this amount of iterations has been done, the tracer will stop.
    power_dissipated = 0.99  # If 99% (0.99) amount of power has left the scene or was terminated/measured, the tracer will stop.

    # Create one light source
    ls0 = LightSource(center=np.array([0, 0, 0], dtype=np.float32), direction=(0, 0, 1),
                      directivity=lambda x, y: np.cos(y), power=1000., ray_count=ray_count)

    # Tracer code expects a list of light sources
    light_sources = [ls0]

    # Define the maximum length a ray can travel before being terminated
    max_ray_len = np.float32(2.0e3)

    # Set index of refraction of environment
    ior_env = np.float32(1.0)

    # Setup mesh elements
    meshes = setup_mesh_elements()

    # Record preparation time
    prep_time = time() - start_time

    # Setup the tracer and select the CL platform and device to run the tracer on
    tracer = it.CLTracer(platform_name="NVIDIA", device_name="460")

    # Run the iterative tracer
    tracer.iterative_tracer(light_source=light_sources, meshes=meshes, trace_iterations=iterations,
                            trace_until_dissipated=power_dissipated, max_ray_len=max_ray_len, ior_env=ior_env)

    # Record simulation time
    sim_time = time() - start_time

    # Fetch results for further processing
    resulting_rays = tracer.results

    processed_ray_count = 0
    for res in resulting_rays:
        processed_ray_count += len(res[3])

    # Fetch measured rays termination position and powers
    measured_rays = tracer.get_measured_rays()

    # Plot the data
    # Extent of measurement surface. ((xmin, xmax), (ymin, ymax))
    # Setting to +-pi/2 for hemisphere
    m_surf_extent = ((-np.pi / 2.0, np.pi / 2.0), (-np.pi / 2.0, np.pi / 2.0))
    # Spatial resolution of binning
    m_points = 30
    tracer.plot_binned_data(limits=m_surf_extent, points=m_points, use_angular=True, use_3d=True)

    # Save traced scene to dxf file if the amount of rays is not too large.
    if ray_count < 10000:
        tracer.show()

    # Show overall stats
    print(f"Processed {processed_ray_count} rays in {sim_time} s")
    print(f"Measured {len(measured_rays[1])} rays.")
    print(f"On average performed {processed_ray_count * int(tracer.tri_count) / float(sim_time)} RI/s")

if __name__ == "__main__":
    main()
