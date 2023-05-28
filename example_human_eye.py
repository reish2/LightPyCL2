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

"""
This script demonstrates:
- the generation of meshes to resemble the optics of the human eye
- setting up a collimated light source
- postprocessing and plotting the light distribution on the retina
"""

import os
from time import time
from typing import List

import numpy as np

from utils import iterative_tracer as it, geo_optical_elements as goe
from utils.light_source import LightSource

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

time1 = time()
# tracer termination conditions
ray_count: int = 1000  # amount of rays to trace
iterations: int = 16  # if this amount of iterations has been done, the tracer will stop.
power_dissipated: float = 0.99  # if 99% (0.99) amount of power has left the scene or was terminated/measured, the tracer will stop

# create a light source
light_source = LightSource(center=np.array([0, 0, -10], dtype=np.float32), 
                           direction=(0, 0.01, 1),
                           directivity=lambda x, y: 1.0 + 0.0 * np.cos(y), 
                           power=1000., 
                           ray_count=ray_count)
light_source.random_collimated_rays(diameter=5.0)
light_sources: List[LightSource] = [light_source]
print("Lights setup complete")

# maximum length a ray can travel before being terminated
max_ray_len: float = np.float32(4.0e1)

# create an instance of optical elements to generate meshes for the scene
oe = goe.OpticalElements()

# index of refraction of environment
ior_env: float = np.float32(1.0)

# eye specifications (data from: "Optics of the Human Eye" by W. N. Charman)
eye_spec = {
    "r_cornea": 5.0,
    "r_lens": 5.0,
    "r_ac": 7.8,
    "d_ac": 0.0,
    "r_pc": 6.5,
    "d_pc": 0.55,
    "r_al": 10.2,
    "d_al": 3.6,
    "r_pl": -6.0,
    "d_pl": 7.6,
    "r_r": -12.1,
    "d_r": 24.2,
    "IOR_c": 1.3771,
    "IOR_l": 1.4200,
    "IOR_ah": 1.3374,
    "IOR_vh": 1.336
}

# setup meshes (eye elements)
meshes = oe.setup_eye_elements(eye_spec)

time2 = time()
prep_time = time2 - time1

# initialize and configure tracer
tracer = it.CLTracer(platform_name="NVIDIA", device_name="460")
print("Raytracer initialized")

time1 = time()

# run the iterative tracer
tracer.iterative_tracer(light_source=light_sources, meshes=meshes, trace_iterations=iterations,
                        trace_until_dissipated=power_dissipated, max_ray_len=max_ray_len, ior_env=ior_env)

time2 = time()
sim_time = time2 - time1

# fetch results for further processing
resulting_rays = tracer.results
proc_ray_count = sum(len(res[3]) for res in resulting_rays)

# fetch measured rays termination position and powers
measured_rays = tracer.get_measured_rays()

# plot the data
m_surf_extent = ((-np.pi / 2.0, np.pi / 2.0), (-np.pi / 2.0, np.pi / 2.0))  # extent of measurement surface
m_points = 90  # spatial resolution of binning
tracer.plot_binned_data(limits=m_surf_extent, points=m_points, use_angular=True, use_3d=True)

# save traced scene to dxf file if the amount of rays is not too large.
if ray_count < 10000:
    #tracer.save_traced_scene("./eye.dxf")
    tracer.show()

# show overall stats
print(f"Processed {proc_ray_count} rays in {sim_time} s")
print(f"Measured {len(measured_rays[1])} rays.")
print(f"On average performed {proc_ray_count * np.int64(tracer.tri_count) / np.float64(sim_time)} RI/s")
