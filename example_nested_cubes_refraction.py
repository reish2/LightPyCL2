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
from typing import List

import numpy as np

from utils import iterative_tracer as it, geo_optical_elements as goe
from utils.light_source import LightSource

# Setup environment for OpenCL
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def run_simulation(ray_count: int, iterations: int, power_dissipated: float, max_ray_len: float, ior_env: float):
    """Run the optical ray tracing simulation."""
    start_time = time()

    # Create a light source
    ls0: LightSource = LightSource(center=np.array([0, 0, 0],
                                   dtype=np.float32),
                                   direction=(0, 0, 1),
                                   directivity=lambda x, y: np.cos(y),
                                   power=1000.,
                                   ray_count=ray_count)
    ls: List[LightSource] = [ls0]

    # Create an instance of OpticalElements
    optical_elements = goe.OpticalElements()

    # Setup optical meshes
    meshes = []

    m2 = optical_elements.cube(center=(0, 0, 20, 0),
                               size=[100, 100, 10, 0])
    m2.set_material(mat_type="refractive", index_of_refraction=1.5)
    meshes.append(m2)

    m2 = optical_elements.cube(center=(-20, 0, 22.5 * (1.0 - 1e-6), 0),
                               size=[30, 30, 5, 0])
    m2.set_material(mat_type="refractive", index_of_refraction=-2.0)
    meshes.append(m2)

    m2 = optical_elements.cube(center=(-20, 0, 27.5 * (1.0 + 1e-6), 0),
                               size=[30, 30, 5, 0])
    m2.set_material(mat_type="refractive", index_of_refraction=-2.0)
    meshes.append(m2)

    m2 = optical_elements.cube(center=(20, 0, 20, 0), size=[30, 30, 5, 0])
    m2.set_material(mat_type="refractive", index_of_refraction=-2.0)
    meshes.append(m2)

    # Initialize the ray tracer
    tracer = it.CLTracer(platform_name="NVIDIA", device_name="460")

    # Run the iterative tracer
    tracer.iterative_tracer(
        light_source=ls,
        meshes=meshes,
        trace_iterations=iterations,
        trace_until_dissipated=power_dissipated,
        max_ray_len=max_ray_len,
        ior_env=ior_env
    )

    # fetch results for further processing
    resulting_rays = tracer.results

    processed_ray_count = sum(len(res[3]) for res in resulting_rays)

    # fetch measured rays termination position and powers
    measured_rays = tracer.get_measured_rays()
    tracer.show(draw_wireframes=True, ray_power_render_threshold_percentile=0)

    # Show overall stats
    end_time = time()
    sim_time = end_time - start_time
    print(f"Processed {processed_ray_count} rays in {sim_time} s")
    print(f"Measured {len(measured_rays[1])} rays.")
    print(f"On average performed {processed_ray_count * np.int64(tracer.tri_count) / np.float64(sim_time)} RI/s")

if __name__ == "__main__":
    run_simulation(ray_count=10, iterations=16, power_dissipated=0.99, max_ray_len=np.float32(2.0e2), ior_env=np.float32(1.0))