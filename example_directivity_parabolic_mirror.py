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

import os
from time import time

import numpy as np

from utils import iterative_tracer as it, geo_optical_elements as goe
from utils.Pose import Pose
from utils.light_source import LightSource

# Enable PyOpenCL compiler output
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def setup_light_sources(ray_count):
    """Setup light sources for the scene."""
    print("Setting up lightsources ...")
    directivity = lambda x, y: np.cos(y)
    ls0 = LightSource(center=np.array([0, 0, 0], dtype=np.float32), direction=(0, 0, -1),
                      directivity=directivity, power=1.0, ray_count=ray_count)
    return [ls0]  # Tracer code expects a list of light sources


def setup_mesh_elements():
    """Setup mesh elements for the scene."""
    print("Setting up scene geometry ...")
    oe = goe.OpticalElements(mesh_angular_resolution=2*np.pi/300)

    # Setup a hemisphere to measure the light sources spatial power distribution
    measure_surf = oe.hemisphere(center=[0, 0, 0], radius=100.0)
    measure_surf.set_material(mat_type="measure")

    # Parabolic mirror setup
    parabolic_mirror = oe.parabolic_mirror(focus=(0, 0, 0), focal_length=5.0, diameter=20.0, reflectivity=0.98)
    parabolic_mirror.transform(Pose.from_axis_angle(np.array([0, 1, 0]) * -np.pi / 2, np.array([0, 0, 0])))

    return [measure_surf, parabolic_mirror]


def run_ray_tracer(iterations, power_dissipated, scene):
    # Define the maximum length a ray can travel before being terminated
    max_ray_len = np.float32(1.0e3)

    # Set index of refraction of environment
    ior_env = np.float32(1.0)

    print("Initializing raytracer ... ")
    tracer = it.CLTracer(platform_name="AMD", device_name="i5")

    start_time = time()
    print("Starting raytracer ...")
    tracer.iterative_tracer(light_source=scene["LightSources"], meshes=scene["Geometry"], trace_iterations=iterations,
                            trace_until_dissipated=power_dissipated, max_ray_len=max_ray_len, ior_env=ior_env)

    sim_time = time() - start_time

    return tracer, sim_time


def process_results(tracer):
    print("Processing Results ...")
    resulting_rays = tracer.results

    proc_ray_count = 0
    for res in resulting_rays:
        proc_ray_count += len(res[3])

    # Fetch measured rays termination position and powers
    measured_rays = tracer.get_measured_rays()

    return proc_ray_count, measured_rays


def plot_and_save(tracer, ray_count):
    # Plotting settings
    nf = 2.0
    m_surf_extent = ((-np.pi / nf, np.pi / nf), (-np.pi / nf, np.pi / nf))
    m_points = 100
    tracer.get_beam_width_half_power(points=m_points, pole=[0, 0, 1, 0])
    tracer.get_beam_HWHM(points=m_points, pole=[0, 0, 1, 0])
    tracer.plot_binned_data(limits=m_surf_extent, points=m_points, use_angular=True, use_3d=True)
    tracer.plot_elevation_histogram(points=90, pole=[0, 0, 1, 0])

    # Save the traced scene if the ray count is not too high
    if ray_count < 1000000:
        tracer.show()


def main():
    # Amount of rays to trace
    ray_count = 50000
    # Tracer termination conditions
    iterations = 16  # If this amount of iterations has been done, the tracer will stop.
    power_dissipated = 0.99  # If 99% (0.99) amount of power has left the scene or was terminated/measured, the tracer will stop.

    # Setup light sources and mesh elements
    scene = {}
    scene["LightSources"] = setup_light_sources(ray_count)
    scene["Geometry"] = setup_mesh_elements()

    tracer, sim_time = run_ray_tracer(iterations, power_dissipated, scene)

    proc_ray_count, measured_rays = process_results(tracer)

    plot_and_save(tracer, ray_count)

    print(f"Processed {proc_ray_count} rays in {sim_time} s")
    print(f"Measured {len(measured_rays[1])} rays.")
    print(f"On average performed {proc_ray_count * int(tracer.tri_count) / float(sim_time)} RI/s")

if __name__ == "__main__":
    main()
