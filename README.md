# LightPyCL2

LightPyCL2 is an advanced physical 3D raytracer, harnessing the power of Python and PyOpenCL to provide high-performance computations. Its core purpose is to deliver accurate simulations of complex optical elements interacting within a variety of lighting scenarios, each with unique directional characteristics and configurations. Importantly, LightPyCL2's primary goal isn't to create aesthetically pleasing graphics, but rather to produce precise illumination maps in an efficient and effective manner.

Released under the MIT license, LightPyCL2 is a freely available software. Its creators passionately offer it to the community with the hope that it will serve as a valuable tool for innovators, researchers, and enthusiasts in the field of optics and lighting simulation.

## Features

**Simplicity:**
- LightPyCL2 uses Python for scene scripting, as well as pre- and post-processing of data. This allows for quick setup and simulation of scenes, as well as effortless result processing and visualization.

**Performance:**
- The tool utilizes OpenCL for computations, allowing for efficient execution on CPUs or GPUs. It effortlessly handles complex simulations involving 1,000,000 rays and 100,000 polygons for accurate scene sampling.

**Materials:**
- LightPyCL2 supports various materials, including real-valued positive and negative index materials, mirrors, termination surfaces, and measurement surfaces. These materials can be applied to any mesh, providing flexibility to measure light distribution on any surface, terminate rays at any point in the scene, and manage light reflection and refraction on objects of arbitrary shape.

**Nested Meshes:**
- The software allows for nested meshes and direct material-to-material transitions, simulating compound materials without requiring an environment pass. This helps avoid unrealistic reflection coefficients during transitions, such as from glass to water.

**Trace Convergence and Completion Conditions:**
- LightPyCL2 provides options for power dissipation and trace depth to manage trace convergence and completion.

**Physical Propagation and Analysis:**
- The tool ensures physically correct propagation of unpolarized rays, offering power transmission and directivity analysis for unpolarized light.

**Mesh Transformations:**
- Basic mesh transformations on optical elements are supported.

**Data Storage and Output:**
- Results can be preserved in a Python pickle for later evaluation. The tool also provides DXF output of traced and untraced scenes.

**Optimization and Element Generation:**
- LightPyCL2 leverages scene symmetries to simulate multiple light sources from a single source's results. Additionally, it can generate optical elements directly from Python either by revolving a 2D curve or parametrically generating a mesh.

## LightPyCL2 Tracer - how it works

The LightPyCL2 tracer, embodied by the `CLTracer` class, is the core engine of this library, designed to carry out 
efficient and accurate light ray tracing through a scene defined by the user. This is accomplished through a process 
known as iterative ray tracing, which allows for the simulation of complex light behaviors such as reflection, 
refraction, and absorption.

At the heart of the `CLTracer` is its `iterative_tracer()` method. This function is called with a set of light sources, 
optical elements (the scene), and various settings that control the tracing process.

The tracing process occurs in the following way:

1. The tracer begins by emitting rays from each light source, using their specified power, ray count, and directivity function.

2. It then iterates through a set number of trace depths or until a certain percentage of the total power is dissipated. 
In each trace iteration, the rays interact with the scene's optical elements. Their interactions (reflection, refraction, 
and absorption) are determined based on the characteristics of the optical elements they encounter.

3. For each interaction, the ray power is updated, and the rays are redirected accordingly. The trace-depth of the ray also increases
by one for every scene interaction. When a ray encounters a 'measure' type surface, its information is stored for later retrieval.

4. The process continues until the termination conditions are met - either reaching the maximum trace iterations (trace-depth) 
or when the specified percentage of the total power is dissipated.

Once tracing is complete, `CLTracer` provides methods to retrieve the results. `get_measured_rays()` fetches the rays 
that terminated at measurement surfaces, useful for further analysis of the scene's optical properties.

Moreover, the tracer also provides built-in visualization functionality. `show()` allows users to visualize the scene 
geometry and the paths of traced rays using Open3D. This helps users better understand how light interacts within their scene.

In sum, LightPyCL2's `CLTracer` provides a comprehensive, efficient, and easy-to-use toolset for performing physically 
accurate light ray tracing simulations.

Setup
-----
To install the SubtileEdgeDetector library, simply clone the repository and install the required dependencies:

First install the prerequisites (assuming a debian based OS here)
```bash
sudo apt-get update
sudo apt-get install python3-venv python3-pip
```

Now clone the repo and enter its root folder
```bash
git clone https://github.com/reish2/LightPyCL2.git
cd LightPyCL2
```

You can either use the provided script `setup.sh` as follows or skip to the next step for manual setup.
```bash
chmod +x setup.sh
./setup.sh
```

Set up a venv and install requirements
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install -r requirements.txt
echo "Setup done."
echo "Now activate your fresh virtualenv:"
echo "source venv/bin/activate"
```

Activate the venv and run `main.py` to verify that everything worked
```bash
source venv/bin/activate
python3 main.py
```

If everything worked, you should see an Open3D window:
![main_output1.png](images%2Fmain_output4.png)
![main_output3.png](images%2Fmain_output5.png)

Note the color coding. Magenta indicates measurment surfaces and cyan optical elements.
The rays are coded from yellow (lower power) to red (higher power).

## Usage

LightPyCL2 is easy to use. 
1. **Scene Construction**: Assemble a scene made up of lights and optical elements using the LightPyCL2 classes:
   * Lights are built with the `LightSource` class found in the [light_source.py](utils%2Flight_source.py) module.
   * Optical elements are created with the `OpticalElements` class housed in the [geo_optical_elements.py](utils%2Fgeo_optical_elements.py) module.
2. **Tracer Configuration**: Instantiate and configure a tracer instance with the `CLTracer` class from the [iterative_tracer.py](utils%2Fiterative_tracer.py) module.
3. **Simulation Execution**: Initiate the ray tracing process by running the tracer with your constructed scene.
4. **Results Retrieval and Visualization**: Retrieve and display the results directly with built-in visualisation methods provided by LightPyCL2.

### Scene Construction

The scene is composed of a list of lights and a list of optical elements.
First we create a light source and add it to our light sources list.
```python
import numpy as np
from utils.light_source import LightSource

# Instantiate a LightSource object with specified parameters.
light_source_1 = LightSource(
    # Set the center of the light source to the origin (0, 0, 0).
    center=np.array([0, 0, 0]),
   
    # Set the rotation of the light source using an axis-angle vector (x, y, z). The direction vector
    # points along the rotation axis and its magnitude corresponds to the rotation angle in radians.
    # Here, the light source is rotated by pi radians (180 degrees) about the z-axis.
    direction=np.array([0, 0, 1])*-np.pi,
   
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
```

The complementary part of the scene is the scene geometry for light to interact with -- the optical elements.
```python
from utils import geo_optical_elements as goe
from utils.Pose import Pose

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
```

### Tracer Configuration & Simulation Execution

The raytracer only needs to know on which device it should run when being instantiated and can be used to run many scene setups.

```python
from utils import iterative_tracer as it

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
```

### Results Retrieval and Visualization

```python
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

```

## Classes

### LightSource Class

The `LightSource` class in `utils.light_source` is used to generate lights with varying properties. This class comes 
with various methods to construct different ray distributions:

* `grid_rays()`: This method generates rays from a point source in regular angles, effectively creating a grid-like 
distribution of rays.
* `random_rays()`: This method generates rays emanating from a point source in random directions. The randomness 
ensures a wide spread of rays.
* `random_collimated_rays()`: This method generates a set of rays that are parallel to each other, emulating collimated 
light. The initial points of these rays are randomly determined.

Here is an example for how to change the light sources emission pattern.
```python
# Instantiate a LightSource object with specified parameters.
light_source_1 = LightSource(
    center=np.array([0, 0, 0]),
    direction=np.array([0, 0, 1])*-np.pi,
    directivity=lambda phi, theta: np.cos(theta),
    power=1.0,
    ray_count=100
)

# Adjust the light emission pattern of light_source_1 to generate randomly positioned but collimated rays. 
# This simulates a light source that produces parallel light, such as a laser or a distant light source.
# The 'diameter' parameter sets the size of the area within which the rays' starting points are randomly distributed.
light_source_1.random_collimated_rays(diameter=3)
```

These methods give users a variety of options when setting up light sources, thus providing flexibility to cater to different simulation needs.

![light_sources2.png](images%2Flight_sources2.png)

### OpticalElements class

The `OpticalElements` class from `geo_optical_elements.py` module is used to generate different types of optical 
elements for a ray-tracing scene. You can create an instance as shown below:

```python
from utils import geo_optical_elements as goe
optical_elements_generator = goe.OpticalElements(mesh_angular_resolution=2*np.pi/1000)
```

The instance `optical_elements_generator` allows the creation of basic geometric forms such as cubes, spheres, hemispheres, and cylinders. 
It also enables the creation of more complex surfaces, including parabolic mirrors, spherical lenses, and other 
surfaces of revolution and extrusion derived from 2D curves.

Here are brief descriptions of some of the key methods within the `OpticalElements` class:

- `cube()`  
  This method generates a cube-shaped optical element, providing you with the option to specify parameters like center 
  and size of the cube.

- `hemisphere()`  
  Creates a hemisphere-shaped optical element, with configurable parameters such as the center and the radius of the hemisphere.

- `lens_spherical_biconcave()`  
  Constructs a biconcave spherical lens with controllable parameters such as focus point, radii of both spherical 
  surfaces, diameter, and the index of refraction of the lens material.

- `parabolic_mirror()`  
  Generates a parabolic mirror. You have control over parameters such as the position of the focus, the focal length and 
  the diameter of the mirror, as well as the reflectivity of the mirror.

- `setup_eye_elements()`  
  This function sets up the optical elements to model an eye, giving you control over various aspects of the eye, 
  such as the index of refraction for different elements of the eye, the radius of the eye and the lens, and the 
  thickness of the lens.

- `sphere()`  
  Generates a sphere-shaped optical element, with options to specify parameters like the center and radius of the sphere.

- `spherical_lens_nofoc()`  
  Creates a lens with a spherical surface but no specific focus, providing you with the option to specify parameters 
  such as the center, diameter, index of refraction, and the thickness of the lens.

Here are some examples for optical elements generated using the mentioned functions.
Note how the color of the element indicates if an element is reflective (cyan; parabolic mirror on the middle left), refractive (blue), or measurement 
surface (magenta; back of the eye model on the lower right). 
![OpticalElements.png](images%2FOpticalElements.png)

# CLTracer Class

The `CLTracer` class is an essential component of the ray tracing system. It manages the OpenCL context, compiles the 
OpenCL kernels, and conducts ray tracing calculations.

Key features include:
- **Initialization**: The class facilitates the setup and control of the OpenCL context and kernels, offering an entry point to harnessing the power of parallel computing for ray tracing.
- **Ray Tracing**: Using a light source and 3D meshes, the class provides an iterative ray tracing method, which includes intersections, reflections, and refractions, considering the complexities of light propagation.
- **Results Filtering and Binning**: The class includes methods to filter rays that hit specific surfaces and bin the ray data for better visualization and interpretation.
- **Data Analysis Tools**: A suite of tools is included to analyze ray data, like calculating the beam's width at half power, its half-width half-maximum (HWHM), and generating histograms to visualize ray elevations.
- **Visualization Support**: The class supports different visualization styles, including 2D, 3D, stereographic, and angular projections of ray endpoints.
- **Saving and Loading Results**: The class supports saving traced results and 3D meshes to a pickle file and loading them when needed, preserving the state of the simulation for further analysis or visualization.
- **Scene Export**: Users can save the traced geometry and rays to a DXF file, facilitating visualization in other platforms.
- **Interactive Viewing**: An interactive viewer, powered by Open3D, offers a real-time 3D visualization of the rays, with control over aspects like wireframe rendering and ray power thresholding.

Here are some of the most important methods in the `CLTracer` class:
- `iterative_tracer`: Executes ray tracing iterations using given light source and meshes.
- `get_measured_rays`: Filters rays that hit the measurement surface, returning their positions and powers.
- `plot_elevation_histogram`: Collects the elevation of all rays and plots a histogram.
- `get_beam_width_half_power`: Calculates the beam width at half power using ray elevation data.
- `get_beam_HWHM`: Computes the half-width half-maximum (HWHM) of the beam based on ray elevation data.
- `get_binned_data_stereographic`: Projects measured ray endpoints stereographically and bins them for visualization.
- `get_binned_data_angular`: Maps measured ray endpoints to a circular azimuth/elevation representation and bins them for visualization.
- `plot_binned_data`: Plots binned data in 2D or 3D.
- `pickle_results`: Saves the results and meshes to a pickle file for future usage.
- `load_pickle_results`: Loads previously saved results from a pickle file.
- `save_traced_scene`: Writes the traced scene's geometry and rays to a DXF file.
- `show`: Visualizes rays and geometry using Open3D, including options for wireframe and thresholded power rendering.


### Postprocessing and Plotting

After tracing, you can analyze the results through different plots: 2D directivity with stereographic or circular azimuth/elevation mapping, and 1D azimuth/elevation histograms. These plots are corrected for mapping distortion and display accurate power values.

1. 3D surface plot showing power distribution: This visualizes power distribution across the measurement surface mapped to elevation and azimuth of rays from the scene origin. 

```python
nf=2.0
m_surf_extent=((-np.pi/nf,np.pi/nf),(-np.pi/nf,np.pi/nf))
m_points=100 
tracer.plot_binned_data(limits=m_surf_extent,points=m_points,use_angular=True,use_3d=True)
```

2. Aggregated elevation plot: This plot, assuming rotational symmetry around a defined axis, uses 90 bins. 

```python
tracer.plot_elevation_histogram(points=90,pole=[0,0,1,0])
```

3. Beam width at half power: To determine the cone's angle encapsulating half of the beam's power, use:

```python
tracer.get_beam_width_half_power(points=90,pole=[0,0,1,0])
```

4. Beam's half-width at half maximum (HWHM): To find the HWHM, use:

```python
tracer.get_beam_HWHM(points=90,pole=[0,0,1,0])
```

Note: `get_beam_HWHM` may require a large number of rays for accurate results, depending on the scene.

The results are stored in `tracer.results` as tuples (ray_origins, ray_destinations, ray_powers, state_measured). `state_measured[i]` indicates the ray's status: terminated (-1), measured (+1), or still traversing the scene (0).

Feel free to contribute your evaluation code or any other improvements to the LightPyCL project.

### Putting it all together

The files *example_directivity_dissipative_cube.py*, *example_directivity_lens.py*, *example_directivity_parabolic_mirror.py*, *python example_human_eye.py*, *python example_nested_cubes_refraction.py* and *example_directivity_sphericalMeasSurf.py* provided with LightPyCL are constructed just as described in the previous subsections of this *README* and are ready to run. To try out the tracer examples just run an example file as follows

`python example_human_eye.py`

in your favourite terminal emulator.

## Known Issues
### Open3D GLFW Error

Issue:
```bash
[Open3D WARNING] GLFW Error: GLX: Failed to create context: GLXBadFBConfig
[Open3D WARNING] Failed to create window
[Open3D WARNING] [DrawGeometries] Failed creating OpenGL window.
```

Cause: No GPU available for rendering

Fix: place the following at the begining of the main (or uncomment in main.py)
```python
import os
os.environ['OPEN3D_CPU_RENDERING'] = 'true'  # Ubuntu 18.04
```

### libGL error with Virtual Box VM

issue: libLLVM-10.so.1 can not be found
```bash
libGL error: MESA-LOADER: failed to open swrast: libLLVM-10.so.1: cannot open shared object file: No such file or directory (search paths /path/to/LightPyCL2/venv/lib/python3.10/site-packages/open3d, suffix _dri)
libGL error: failed to load driver: swrast
Segmentation fault (core dumped)
```

Fix: link libLLVM-10.so.1 to vmwgfx_dri.so instead
```bash
cd /path/to/LightPyCL2/venv/lib/python3.10/site-packages/open3d
mv swrast_dri.so swrast_dri.bak
ln -s /usr/lib/x86_64-linux-gnu/dri/vmwgfx_dri.so swrast_dri.so
```

## Performance

The performance of the raytracer is determined by measuring the time __T__ a combined intersection and reflection/refraction cycle takes for __N__ input rays and __M__ triangles in a scene. Because every ray has to search all triangles in a scene for a valid closest intersection, __N__ * __M__ gives the number of performed refractive intersections or, put differently, the number of rays that could be intersected and refracted if the scene consisted of one triangle. Thus a comparative measure of performance is __N__ * __M__ / __T__ given in "refractive intersections/s" or "RI/s".

Here are some results from various platforms:
<table>
	<tr><td>Intel i5-8400</td>		<td>~ 0.5e9 RI/s</td></tr>
	<tr><td>nVidia GTX460</td>	<td>~ 4.1e9 RI/s</td></tr>
	<tr><td>nVidia GTX770</td>	<td>~ 9.9e9 RI/s</td></tr>
    <tr><td>nVidia GTX1650 Max-Q</td>	<td>~ 22.3e9 RI/s</td></tr>
    <tr><td>nVidia RTX3080 Ti</td>	<td>~ 130.1e9 RI/s</td></tr>
</table>

Performance results are printed in the console during simulation, so if you would like to share those results, drop me a line!

## License
This project is licensed under the MIT License. See the LICENSE file for details.