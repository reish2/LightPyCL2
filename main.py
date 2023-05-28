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

import numpy as np
from utils import iterative_tracer as it, geo_optical_elements as goe
from utils.Pose import Pose
from utils.light_source import LightSource

# Create a light source with specific characteristics
light_source_1 = LightSource(
    center=np.array([0, 0, 0]),
    direction=(0, 0, -1),
    directivity=lambda phi, theta: np.cos(theta),
    power=1.0,
    ray_count=100,
)

# The raytracer works with a list of light sources
light_sources = [light_source_1]

# Set up the scene geometry
optical_elements_generator = goe.OpticalElements(mesh_angular_resolution=2*np.pi/1000)

# Create a parabolic mirror and a hemisphere for measurements
parabolic_mirror = optical_elements_generator.parabolic_mirror(focus=(0, 0, 0), focal_length=5.0, diameter=20.0, reflectivity=0.98)
parabolic_mirror.transform(Pose.from_axis_angle(np.array([0, 1, 0]) * -np.pi / 2, np.array([0, 0, 0])))

measurement_hemisphere = optical_elements_generator.hemisphere(center=[0, 0, 0], radius=100.0)
measurement_hemisphere.set_material(mat_type="measure")

# The scene consists of a list of optical elements
scene = [measurement_hemisphere, parabolic_mirror]

# Set up a tracer instance and run it
tracer = it.CLTracer(platform_name="AMD", device_name="i5")
tracer.iterative_tracer(
    light_source=light_sources,
    meshes=scene,
    trace_iterations=16,
    trace_until_dissipated=0.99,
    max_ray_len=1e3,
    ior_env=1.0,
)

# Fetch results and display them
measured_rays = tracer.get_measured_rays()
tracer.show(draw_wireframes=False, ray_power_render_threshold_percentile=0.2)
