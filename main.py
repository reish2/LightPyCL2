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

# For Open3D CPU rendering, uncomment the following two lines
#import os
#os.environ['OPEN3D_CPU_RENDERING'] = 'true'  # Ubuntu 18.04

import numpy as np
from utils import iterative_tracer as it, geo_optical_elements as goe
from utils.Pose import Pose
from utils.light_source import LightSource

# Instantiate a LightSource object with specified parameters.
light_source_1 = LightSource(
    # Set the center of the light source to the origin (0, 0, 0).
    center=np.array([0, 0, 0]),

    # Set the rotation of the light source using an axis-angle vector (x, y, z). The direction vector
    # points along the rotation axis and its magnitude corresponds to the rotation angle in radians.
    # Here, the light source is rotated by pi radians (180 degrees) about the z-axis.
    direction=np.array([0, 0, 1]) * -np.pi,

    # Set the directivity function of the light source. This function describes the directional
    # intensity of the light source as a function of the spherical coordinates phi (azimuthal angle)
    # and theta (polar angle). Here, a lambda function is used to implement Lambertian directivity,
    # which corresponds to a surface that emits light uniformly in all directions (perfect diffuser).
    directivity=lambda phi, theta: np.cos(theta),

    # Set the total power (in arbitrary units) emitted by the light source.
    power=1.0,

    # Set the number of rays that will be traced from the light source. The more rays, the more
    # accurately the scene will be sampled, but the greater the computational cost.
    ray_count=100
)

# Create a list of light sources for the scene. While we only have one light source here,
# this list can include multiple LightSource objects for complex lighting setups.
light_sources = [light_source_1]

# Instantiate an OpticalElements object for generating the optical elements in the scene.
# Set the mesh_angular_resolution parameter to define the angular resolution of the generated meshes.
# This parameter sets the step size for the discretization of the 2π range into mesh points.
# Smaller values result in finer meshes, providing more accurate optical simulations at the cost of increased computation time.
optical_elements_generator = goe.OpticalElements(mesh_angular_resolution=2*np.pi/1000)

# Use the OpticalElements object to generate a parabolic mirror. The parabolic mirror is defined by its focus,
# focal length, diameter, and reflectivity. The focus of the mirror is set at the origin (0, 0, 0), its focal length is 5.0 units,
# its diameter is 20.0 units, and its reflectivity is 0.98.
parabolic_mirror = optical_elements_generator.parabolic_mirror(focus=(0, 0, 0), focal_length=5.0, diameter=20.0, reflectivity=0.98)

# Apply a Pose transformation to the parabolic mirror. A Pose is defined by an axis-angle rotation vector and a translation vector.
# The axis-angle vector [0, 1, 0] * -π/2 rotates the mirror by π/2 radians (90 degrees) around the y-axis.
# The translation vector [0, 0, 0] doesn't change the position of the mirror, keeping it at the origin.
parabolic_mirror.transform(Pose.from_axis_angle(np.array([0, 1, 0]) * -np.pi / 2, np.array([0, 0, 0])))

# Use the OpticalElements object to generate a measurement hemisphere. The hemisphere is centered at the origin (0, 0, 0)
# and has a radius of 100.0 units. The hemisphere is used to measure the spatial power distribution of light in the scene.
measurement_hemisphere = optical_elements_generator.hemisphere(center=[0, 0, 0], radius=100.0)

# Set the material type of the hemisphere to "measure", which designates it as a surface for light measurement.
measurement_hemisphere.set_material(mat_type="measure")

# Define the scene as a list of optical elements. In this case, the scene consists of the measurement hemisphere and the parabolic mirror.
scene = [measurement_hemisphere, parabolic_mirror]

# Instantiate a CLTracer object for raytracing the scene. The platform_name and device_name parameters specify the
# computing platform and device to be used for calculations. Here, an AMD platform and an i5 device are chosen.
# Depending on your system's hardware and drivers, different platforms (e.g., NVIDIA) and devices could be used.
tracer = it.CLTracer(platform_name="AMD", device_name="i5")

# Execute the iterative raytracing process with the defined light sources and scene. Various parameters are set:
# light_source: List of light sources illuminating the scene.
# meshes: List of optical elements forming the scene.
# trace_iterations: The maximum number of reflection/refraction iterations for each light ray. The raytracing process will stop
#                   if this number of iterations is reached, even if some of the light power is not yet dissipated.
# trace_until_dissipated: A fraction of the total light power that needs to be dissipated (absorbed or measured) for the raytracing
#                         process to stop. In this case, the raytracing will stop once 99% of the total light power is dissipated.
# max_ray_len: The maximum length that a ray can travel. Rays exceeding this length are terminated.
# ior_env: The index of refraction of the environment surrounding the optical elements.
tracer.iterative_tracer(
    light_source=light_sources,
    meshes=scene,
    trace_iterations=16,
    trace_until_dissipated=0.99,
    max_ray_len=1e3,
    ior_env=1.0,
)

# Retrieve the rays that terminated at a measurement surface. This method fetches these rays from the CL device and
# returns them as a numpy array. This array contains information about the origin, direction, and power of each measured ray.
measured_rays = tracer.get_measured_rays()

# Visualize the scene and raytracing results using Open3D. This method shows the optical elements in the scene and
# the path of the rays that were traced. The arguments are:
# draw_wireframes: Whether or not to render the mesh wireframes of the optical elements. Setting this to False results
#                  in a cleaner, solid-color visualization of the elements.
# ray_power_render_threshold_percentile: This parameter sets a percentile threshold for the power of the rays to be visualized
#                                        at each trace depth independently. Only rays with power above this percentile are
#                                        drawn for each trace depth. This helps focus on the most significant rays and declutters
#                                        the visualization when dealing with large numbers of rays. In this case, only the top 20%
#                                        most powerful rays are drawn at each trace depth level.
tracer.show(draw_wireframes=False, ray_power_render_threshold_percentile=0.2)
