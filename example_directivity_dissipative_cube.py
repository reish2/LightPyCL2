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

from time import time
from typing import List

import numpy as np

from utils import iterative_tracer as it, geo_optical_elements as goe
from utils.light_source import LightSource


def main():
    """
    This example demonstrates:
    - The generation of optical elements and light sources for the tracer
    - Raytracer initialization and configuration
    - Post-processing and plotting the results
    """
    start_time = time()

    # Parameters for ray tracer and light sources
    ray_count = 3000
    iterations = 16
    power_dissipated = 0.99
    max_ray_len = np.float32(2.0e3)
    ior_env = np.float32(1.0)

    # Setup the light sources
    ls0: LightSource = LightSource(center=np.array([0, 0, 0], dtype=np.float32),
                                   direction=(0, 0, 1),
                                   directivity=lambda x, y: np.cos(y),
                                   power=1000.,
                                   ray_count=ray_count)
    ls: List[LightSource] = [ls0]

    # Setup the optical elements
    oe = goe.OpticalElements()
    measureSurf = oe.hemisphere(center=[0, 0, 0], radius=1000.0)
    measureSurf.set_material(mat_type="measure")
    meshes = [measureSurf]

    m2 = oe.cube(center=(0, 0, 20, 0), size=[10, 10, 10, 0])
    m2.set_material(mat_type="refractive", index_of_refraction=1.0, dissipation=1.0)
    meshes.append(m2)

    prep_time = time() - start_time

    # Setup the tracer
    tr = it.CLTracer(platform_name="NVIDIA", device_name="460")

    start_time = time()

    # Run the iterative tracer
    tr.iterative_tracer(light_source=ls,
                        meshes=meshes,
                        trace_iterations=iterations,
                        trace_until_dissipated=power_dissipated,
                        max_ray_len=max_ray_len,
                        ior_env=ior_env)

    sim_time = time() - start_time

    # Fetch results for further processing
    resulting_rays = tr.results
    proc_ray_count = sum(len(res[3]) for res in resulting_rays)

    # Fetch measured rays termination position and powers
    measured_rays = tr.get_measured_rays()

    # Plot the data
    m_surf_extent = ((-np.pi/2.0, np.pi/2.0), (-np.pi/2.0, np.pi/2.0))
    m_points = 30
    tr.plot_binned_data(limits=m_surf_extent,
                        points=m_points,
                        use_angular=True,
                        use_3d=True)

    # Save traced scene to dxf file if the amount of rays is not too large
    if ray_count < 10000:
        tr.show()

    # Show overall stats
    print(f"Processed {proc_ray_count} rays in {sim_time}s")
    print(f"Measured {len(measured_rays[1])} rays.")
    print(f"On average performed {proc_ray_count * tr.tri_count / sim_time} RI/s")

if __name__ == "__main__":
    main()
