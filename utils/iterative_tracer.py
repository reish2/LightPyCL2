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

# Standard library imports

import open3d as o3d

import pickle
from time import time
import datetime

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array as cl_array
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from dxfwrite import DXFEngine as dxf

from utils.light_source import LightSource
from utils.geo_optical_elements import OpticalElements
from typing import Optional, Union, Tuple, List
from pathlib import Path


class CLTracer:
    """
    Class to initialize the OpenCL platform and device, setup the OpenCL context and queue,
    and load and compile the OpenCL kernel.
    """

    def __init__(self, platform_name: str = "NVIDIA", device_name: str = "770", debug: bool = False):
        """
        Initialize OpenCL platform and device, context, and queue, and load and compile the OpenCL kernel.

        Args:
            platform_name (str): Name of the OpenCL platform. Defaults to "NVIDIA".
            device_name (str): Name of the OpenCL device. Defaults to "770".
            debug (bool): If true, prints debug information. Defaults to False.
        """
        self.results = None
        self.geometry = None
        self.debug = debug

        # Obtain an OpenCL platform
        platforms = cl.get_platforms()
        self.platform = next((pf for pf in platforms if platform_name in pf.name), platforms[0])

        # Obtain a device id for at least one device (accelerator)
        devices = self.platform.get_devices()
        self.device = next((dev for dev in devices if device_name in dev.name), devices[0])

        print(f"Using CL device: {self.platform.name} - {self.device.name}")

        # Create a context for the selected device
        self.context = cl.Context([self.device])

        # Load and build kernel
        print("Loading and building kernel.")
        with open("utils/kernel_reflect_refract_intersect.cl", "r") as f:
            kernel_str = f.read()

        self.prg = cl.Program(self.context, kernel_str).build("")

        # Create a command queue for the target device
        self.queue = cl.CommandQueue(self.context)

    def iterative_tracer(self, light_source: LightSource, meshes: List[OpticalElements], trace_iterations: int = 100,
                         trace_until_dissipated: float = 0.99, max_ray_len: float = np.float32(1e3),
                         ior_env: float = np.float32(1.0)):
        """
        The method takes the light_source and meshes objects and converts them to cl_arrays. It then intersects,
        reflects, and refracts all the rays from the light source(s) repeated 'trace_iterations' number of times.
        New rays (reflected/refracted ones) are the result of every iteration and are used as the input for the next.
        Results are stored in the 'results' class property as a list of tuples (rays_origin, rays_dest, power, measured)
        for every iteration. For very large numbers of input rays, rays are partitioned to prevent using more memory
        than the CL device provides. The calculation of the maximum ray count is based on cl_device.global_mem_size
        and sizes of the buffers required for calculation and overhead.

        Args:
            light_source (LightSource): Source of the light rays.
            meshes (List[OpticalElements]): Meshes in the scene.
            trace_iterations (int): Number of iterations to trace. Defaults to 100.
            trace_until_dissipated (float): Threshold for ray dissipation. Defaults to 0.99.
            max_ray_len (float): Maximum length of a ray. Defaults to 1e3.
            ior_env (float): Index of refraction of the environment. Defaults to 1.0.
        """

        # Step #8. Allocate device memory and move input data from the host to the device memory.
        # rays
        max_ray_len = np.float32(max_ray_len)
        ior_env = np.float32(ior_env)
        print("Initializing variables and converting scene data.")

        # Prepare lightsource buffers
        ray_count = 0
        rays_origin = np.array([], dtype=np.float32)
        rays_dir = np.array([], dtype=np.float32)
        rays_power = np.array([], dtype=np.float32)

        for light in light_source:
            ray_count += light.ray_count

            if rays_origin.size == 0:
                rays_origin = np.float32(light.rays_origin)
                rays_dir = np.float32(light.rays_dir)
                rays_power = np.float32(light.rays_power)
            else:
                rays_origin = np.append(rays_origin, light.rays_origin, axis=0)
                rays_dir = np.append(rays_dir, light.rays_dir, axis=0)
                rays_power = np.append(rays_power, light.rays_power, axis=0)

        ray_count = np.int32(ray_count)

        # Calculate the input power
        input_power = sum(np.sort(rays_power))
        rays_pow = np.array(rays_power, dtype=np.float32)
        rays_meas = np.zeros(ray_count, dtype=np.int32)
        # Set initial value -2 to indicate that the rays have been emitted and are not currently inside any mesh
        rays_current_mid = np.full(ray_count, -2, dtype=np.int32)

        # MESH does not change => only needs to be set up at start of tracer
        mesh_count = np.int32(len(meshes))
        mesh_mat_type = np.zeros(mesh_count, dtype=np.int32)
        mesh_ior = np.zeros(mesh_count, dtype=np.float32)
        mesh_refl = np.zeros(mesh_count, dtype=np.float32)
        mesh_diss = np.zeros(mesh_count, dtype=np.float32)

        # Convert meshes to linear buffer with mesh id for tracer to be able to iterate over
        m_v0, m_v1, m_v2, m_id = [], [], [], []
        for i, mesh in enumerate(meshes):
            mesh_mat = mesh.get_material_buf()

            mesh_mat_type[i] = np.int32(mesh_mat.get("type"))
            mesh_ior[i] = np.float32(mesh_mat.get("index_of_refraction"))
            mesh_refl[i] = np.float32(mesh_mat.get("reflectivity"))
            mesh_diss[i] = np.float32(mesh_mat.get("dissipation"))

            tribuf = mesh.tribuf()
            m_id_tmp = np.full(len(tribuf[0]), i, dtype=np.int32)

            m_v0.extend(tribuf[0])
            m_v1.extend(tribuf[1])
            m_v2.extend(tribuf[2])
            m_id.extend(m_id_tmp)

        m_v0, m_v1, m_v2 = map(lambda x: np.array(x, dtype=np.float32), [m_v0, m_v1, m_v2])
        m_id = np.array(m_id, dtype=np.int32)
        tri_count = np.int32(len(m_v0))
        self.tri_count = tri_count

        print(f"Triangle count: {tri_count}")
        print(f"Mesh count: {mesh_count}")

        # Store geometry for other functions to access
        self.meshes = meshes
        self.geometry = (m_v0, m_v1, m_v2)

        # Transfer geometry data to device memory
        m_v0_buf = cl_array.to_device(self.queue, m_v0)
        m_v1_buf = cl_array.to_device(self.queue, m_v1)
        m_v2_buf = cl_array.to_device(self.queue, m_v2)
        m_id_buf = cl_array.to_device(self.queue, m_id)

        m_typ_buf = cl_array.to_device(self.queue, mesh_mat_type)
        m_ior_buf = cl_array.to_device(self.queue, mesh_ior)
        m_refl_buf = cl_array.to_device(self.queue, mesh_refl)
        m_diss_buf = cl_array.to_device(self.queue, mesh_diss)

        # Sizes of float and int types in bytes
        sFloat = 4
        sFloat3 = sFloat * 4
        sInt = 4

        # Sizes of various variables in memory
        ray_var_sizes = {"r_orig": sFloat3, "r_dst": sFloat3, "r_dir": sFloat3, "r_pow": sFloat, "r_meas": sInt,
                         "r_ent": sInt,  # input ray
                         "rr_orig": sFloat3, "rr_dir": sFloat3, "rt_pow": sFloat, "rt_meas": sInt,  # reflected ray
                         "rt_orig": sFloat3, "rt_dir": sFloat3, "rt_pow": sFloat, "rt_meas": sInt,
                         # transmitted ray
                         "r_prev_isect_mid": sInt, "r_isect_mid": sInt, "r_isect_midx": sInt, "r_n1_mid": sInt,
                         "r_n2_mid": sInt,  # intersection material bufs
                         "isect_count": mesh_count * sInt, "isect_midx_hits": mesh_count * sInt,
                         "isect_m_minRayLen": mesh_count * sFloat
                         # intersect intermediate results before postprocessing
                         }
        mesh_var_sizes = {"v0": tri_count * sFloat3, "v1": tri_count * sFloat3, "v2": tri_count * sFloat3,
                          "m_id": tri_count * sInt,  # vertex buffers
                          "m_typ": mesh_count * sInt, "m_ior": mesh_count * sFloat, "m_refl": mesh_count * sFloat,
                          "m_diss": mesh_count * sFloat  # material buffers
                          }

        # Calculate memory requirements
        space_req_per_ray = sum(ray_var_sizes.values())
        space_req_meshes = sum(mesh_var_sizes.values())
        global_mem_size = self.device.global_mem_size
        global_mem_overhead_est = global_mem_size * 0.15
        max_ray_count = 50000

        # Print memory requirement details
        print(f"Available space on CL dev:   {global_mem_size / 1024 ** 2} MB")
        print(f"Space required per ray:      {space_req_per_ray} Bytes")
        print(f"Space required for all rays: {space_req_per_ray * ray_count / 1024 ** 2} MB")
        print(f"Space required for geometry: {space_req_meshes / 1024 ** 2} MB")
        print(f"Total space required:        {(space_req_meshes + space_req_per_ray * ray_count) / 1024 ** 2} MB")
        print(f"Maximum permitted ray count: {max_ray_count}")

        # SET UP CL DEVICE TEMP CALCULATION AND RESULT BUFFERS

        # Depending on the resources of the CL device, partition the input rays
        # Set up the partitioning variables

        # Find a clever way to do partitioning.
        # If ray_count is smaller, only that number of rays will be processed by setting the global id size.
        # If ray_count is larger than the maximum number of rays that can fit into global memory, do partitioning.
        part_ray_count = max(min(ray_count, max_ray_count), 50000)

        # Set up the internal buffers. They don't need initializing and will be used by the kernels as needed.
        # This saves setup time.

        # Define buffer names and their dimensions
        buffer_names = ['r_dest', 'rr_origin', 'rr_dir', 'rr_pow', 'rr_meas',
                        'rt_origin', 'rt_dir', 'rt_pow', 'rt_meas', 'r_entering',
                        'r_isect_m_id', 'r_isect_m_idx', 'r_n1_m_id', 'r_n2_m_id']

        r_dest_buf = cl_array.zeros(self.queue, (part_ray_count, 4), dtype=np.float32)
        rr_origin_buf = cl_array.zeros(self.queue, (part_ray_count, 4), dtype=np.float32)
        rr_dir_buf = cl_array.zeros(self.queue, (part_ray_count, 4), dtype=np.float32)
        rr_pow_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.float32)
        rr_meas_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.int32)
        rt_origin_buf = cl_array.zeros(self.queue, (part_ray_count, 4), dtype=np.float32)
        rt_dir_buf = cl_array.zeros(self.queue, (part_ray_count, 4), dtype=np.float32)
        rt_pow_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.float32)
        rt_meas_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.int32)

        r_entering_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.int32)
        r_isect_m_id_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.int32)
        r_isect_m_idx_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.int32)
        r_n1_m_id_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.int32)
        r_n2_m_id_buf = cl_array.zeros(self.queue, (part_ray_count, 1), dtype=np.int32)

        isects_count_buf = cl_array.zeros(self.queue, (part_ray_count, mesh_count), dtype=np.int32)
        ray_isect_mesh_idx_tmp_buf = cl_array.zeros(self.queue, (part_ray_count, mesh_count), dtype=np.int32)

        # create results variable and iterate
        self.results = []
        for t_iter in np.arange(trace_iterations):
            print(" ")
            print("ITERATION", t_iter + 1, "with ", ray_count, "rays.")
            print("================================================================")

            NParts = np.int32(np.ceil(np.float32(ray_count) / np.float32(
                part_ray_count)))  # amount of partitioning required to process all rays
            if NParts > 1:
                print("Too many rays for single pass. Partitioning rays into ", NParts, " groups.")

            # setup result buffers
            rays_dest = np.empty_like(rays_origin).astype(np.float32)
            rrays_origin = np.empty_like(rays_origin).astype(np.float32)
            rrays_dir = np.empty_like(rays_origin).astype(np.float32)
            rrays_meas = np.empty_like(rays_meas).astype(np.float32)
            rrays_pow = np.empty_like(rays_meas).astype(np.float32)
            trays_origin = np.empty_like(rays_origin).astype(np.float32)
            trays_dir = np.empty_like(rays_origin).astype(np.float32)
            trays_meas = np.empty_like(rays_meas).astype(np.float32)
            trays_pow = np.empty_like(rays_meas).astype(np.float32)

            for part in np.arange(0, NParts, 1):
                if NParts > 1:
                    print("")
                    print("ITERATION", t_iter + 1, "partition", part + 1, "of", NParts, "ray partitions.")
                    print("----------------------------------------------------------------")

                isect_min_ray_len_buf = cl_array.zeros(self.queue, (part_ray_count, mesh_count),
                                                       dtype=np.float32) + max_ray_len

                # partitioning indices
                minidx = part * part_ray_count
                maxidx = min((part + 1) * part_ray_count - 1,
                             ray_count - 1) + 1  # +1 because array[x:y] accesses elements x through y-1

                part_ray_count_this_iter = maxidx - minidx  # rays that will be processed in this iteration

                # CREATE DEVICE INPUT DATA
                print("Seting us up the buffers on the CL device.")
                mf = cl.mem_flags
                time1 = time()
                # ray bufs
                r_origin_buf = cl_array.to_device(self.queue, rays_origin[minidx:maxidx, :])  # needs copy
                r_dir_buf = cl_array.to_device(self.queue, rays_dir[minidx:maxidx, :])  # needs copy
                r_pow_buf = cl_array.to_device(self.queue, rays_pow[minidx:maxidx])  # needs copy
                r_meas_buf = cl_array.to_device(self.queue, rays_meas[minidx:maxidx])  # needs copy
                r_prev_isect_m_id_buf = cl_array.to_device(self.queue,
                                                           rays_current_mid[minidx:maxidx])  # needs copy
                # INTERSECT RAYS WITH SCENE
                GIDs = rays_meas[
                       minidx:maxidx].shape  # (part_ray_count_this_iter,1) # max global ids set to number of input rays
                print("Starting intersect parallel ray kernel.")
                event = self.prg.intersect(self.queue, GIDs, None, r_origin_buf.data,
                                           r_dir_buf.data, r_dest_buf.data, r_entering_buf.data,
                                           r_isect_m_id_buf.data, r_isect_m_idx_buf.data,
                                           m_v0_buf.data, m_v1_buf.data, m_v2_buf.data, m_id_buf.data,
                                           isect_min_ray_len_buf.data, isects_count_buf.data,
                                           ray_isect_mesh_idx_tmp_buf.data, np.int32(mesh_count),
                                           np.int32(tri_count), np.int32(ray_count), np.float32(max_ray_len))
                event.wait()
                time2 = time()
                t_intersect = time2 - time1
                print("Intersection execution time:   ", time2 - time1, "s")

                # POSTPROCESS INTERSECT RESULTS
                print("Running intersect postprocessing kernel.")
                time1 = time()
                event2 = self.prg.intersect_postproc(self.queue, GIDs, None, r_origin_buf.data,
                                                     r_dir_buf.data, r_dest_buf.data, r_prev_isect_m_id_buf.data,
                                                     r_n1_m_id_buf.data, r_n2_m_id_buf.data, r_entering_buf.data,
                                                     r_isect_m_id_buf.data, r_isect_m_idx_buf.data,
                                                     m_v0_buf.data, m_v1_buf.data, m_v2_buf.data, m_id_buf.data,
                                                     m_typ_buf.data,
                                                     isect_min_ray_len_buf.data, isects_count_buf.data,
                                                     ray_isect_mesh_idx_tmp_buf.data, mesh_count, ray_count,
                                                     max_ray_len)
                event2.wait()
                time2 = time()
                t_postproc = time2 - time1
                print("Intersect postprocessing time: ", time2 - time1, "s")

                # REFLECT AND REFRACT INTERSECTED RAYS
                print("Running Fresnell kernel.")
                time1 = time()
                event3 = self.prg.reflect_refract_rays(self.queue, GIDs, None, r_origin_buf.data, r_dest_buf.data,
                                                       r_dir_buf.data, r_pow_buf.data, r_meas_buf.data,
                                                       r_entering_buf.data,
                                                       r_n1_m_id_buf.data, r_n2_m_id_buf.data,
                                                       rr_origin_buf.data, rr_dir_buf.data, rr_pow_buf.data,
                                                       rr_meas_buf.data,
                                                       rt_origin_buf.data, rt_dir_buf.data, rt_pow_buf.data,
                                                       rt_meas_buf.data,
                                                       r_isect_m_id_buf.data, r_isect_m_idx_buf.data, m_v0_buf.data,
                                                       m_v1_buf.data,
                                                       m_v2_buf.data, m_id_buf.data, m_typ_buf.data, m_ior_buf.data,
                                                       m_refl_buf.data,
                                                       m_diss_buf.data, ior_env, mesh_count, ray_count, max_ray_len)
                event3.wait()
                time2 = time()
                t_fresnell = time2 - time1
                print("Fresnell processing time:      ", time2 - time1, "s")
                print("Performance:                   ",
                      (np.float64(part_ray_count_this_iter) * np.float64(tri_count)) / np.float64(t_fresnell + t_intersect + t_postproc), "RI/s")

                # FETCH RESULTS FROM CL DEV
                print("Fetching results from device.")
                time1 = time()
                rrays_origin[minidx:maxidx, :] = rr_origin_buf.get(self.queue)[0:part_ray_count_this_iter, :]
                rrays_dir[minidx:maxidx, :] = rr_dir_buf.get(self.queue)[0:part_ray_count_this_iter, :]
                rrays_meas[minidx:maxidx] = rr_meas_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
                rrays_pow[minidx:maxidx] = rr_pow_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
                trays_origin[minidx:maxidx, :] = rt_origin_buf.get(self.queue)[0:part_ray_count_this_iter, :]
                trays_dir[minidx:maxidx, :] = rt_dir_buf.get(self.queue)[0:part_ray_count_this_iter, :]
                trays_meas[minidx:maxidx] = rt_meas_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]
                trays_pow[minidx:maxidx] = rt_pow_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]

                rays_current_mid[minidx:maxidx] = r_isect_m_id_buf.get(self.queue).flatten()[
                                                  0:part_ray_count_this_iter]

                rays_dest[minidx:maxidx, :] = r_dest_buf.get(self.queue)[0:part_ray_count_this_iter, :]
                rays_pow[minidx:maxidx] = r_pow_buf.get(self.queue)[0:part_ray_count_this_iter]
                rays_meas[minidx:maxidx] = r_meas_buf.get(self.queue).flatten()[0:part_ray_count_this_iter]

                time2 = time()
                print("Fetching results took          ", time2 - time1, "s.")

            # APPEND RESULTS OF THIS ITERATION TO OVERALL RESULTS
            print("Assembling results")
            self.results.append((rays_origin, rays_dest, rays_pow, rays_meas))

            # ASSEMBLE INPUT RAYS FOR NEXT ITERATION OR QUIT LOOP IF NO RAYS ARE LEFT
            print("Setting up for next iter.")

            # remove measured and terminated rays for next iteration
            time1 = time()
            idx = np.where(np.concatenate((rrays_meas, trays_meas), axis=0) == 0)[
                0]  # filter index for unmeasured/unterminated result rays
            rays_origin = np.append(rrays_origin, trays_origin, axis=0).astype(np.float32)[idx]
            rays_dir = np.append(rrays_dir, trays_dir, axis=0).astype(np.float32)[idx]
            rays_pow = np.append(rrays_pow, trays_pow, axis=0).astype(np.float32)[idx]
            rays_meas = np.append(rrays_meas, trays_meas, axis=0).astype(np.int32)[idx]

            power_in_scene = sum(np.sort(rays_pow))
            rays_current_mid = np.append(rays_current_mid, rays_current_mid, axis=0).astype(np.int32)[idx]

            ray_count = np.int32(len(rays_origin))

            time2 = time()
            t_host_pproc = time2 - time1
            print("Host side data pruning:        ", t_host_pproc, "s")

            print("Power left in scene:           ", power_in_scene / input_power * 100.0)
            if power_in_scene < (1.0 - trace_until_dissipated) * input_power:
                print("****************************************************************")
                print("Scene power is below", trace_until_dissipated * 100.0,
                      "% of input power. Terminating trace.")
                print("****************************************************************")
                break

            if ray_count == 0:
                print("No rays left to trace. Terminating tracer.")
                break

        return self.results
    def get_measured_rays(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Filters the results to return only those rays that hit the measurement surface.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            A tuple containing positions and powers of rays that hit the measurement surface.
            Returns (None, None) if no rays hit the measurement surface.
        """

        # Initialize position and power variables
        pos, pwr = None, None

        # Iterate through results
        for k, (rays_origin, rays_dest, r_pow, r_meas) in enumerate(self.results):

            # Find indices of rays that hit the measurement surface
            idx = np.where(r_meas >= .9)[0]

            # Get destination and power of measured rays
            rays_dest_m = rays_dest[idx]
            rays_pow_m = r_pow[idx]

            # For the first iteration, assign rays_dest_m and rays_pow_m to pos and pwr
            if k == 0:
                pos = rays_dest_m
                pwr = rays_pow_m

            # For subsequent iterations, concatenate new measured rays to existing ones
            else:
                pos = np.concatenate((pos, rays_dest_m), axis=0)
                pwr = np.concatenate((pwr.flatten(), rays_pow_m.flatten()), axis=0)

        return pos, pwr

    def get_binned_data(self, limits: Tuple[Tuple[float, float], Tuple[float, float]] = ((-10, 10), (-10, 10)),
                        points: int = 500) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Bin data for visualization with a specified number of points within given limits.
        Defaults to 500 points within (-10, 10) for both dimensions.

        Parameters
        ----------
        limits : Tuple[Tuple[float, float], Tuple[float, float]], optional
            The boundaries for the two dimensions, by default ((-10, 10), (-10, 10))
        points : int, optional
            The number of points to be used for binning, by default 500

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            A tuple containing 2D histogram of positions weighted by power,
            x-coordinates and y-coordinates of the bins.
            Returns None if no rays hit the measurement surface.
        """

        # Fetch measured results
        pos, pwr = self.get_measured_rays()

        if pos is not None and pwr is not None:
            # Compute 2D histogram of positions, weighted by power
            H, x_coord, y_coord = np.histogram2d(x=pos[:, 0], y=pos[:, 1], bins=points, range=limits, weights=pwr)

            # Store the results and return them
            self.hist_data = (H, x_coord, y_coord)
            return self.hist_data

        return None

    def plot_elevation_histogram(self, points: int = 500, pole: List[float] = [0, 0, 1, 0]) -> None:
        """
        Collect only the elevation of all rays and plot a histogram of them.

        Parameters
        ----------
        points : int, optional
            The number of points to be used for binning, by default 500
        pole : List[float], optional
            The pole in Cartesian coordinates, by default [0, 0, 1, 0]
        """

        # Fetch measured results
        pos, pwr = self.get_measured_rays()

        # Normalize positions and flatten power
        pos0 = np.array(np.divide(pos, np.matrix(np.linalg.norm(pos, axis=1)).T))
        pwr = np.float64(pwr).flatten()

        # Calculate the angle between each position vector and the pole vector (the elevation)
        elevation = np.arccos(np.dot(pos0, pole)).flatten()

        # Calculate 1D histogram over all elevations, weighting by power
        H, x = np.histogram(elevation, bins=points, weights=pwr)

        # Calculate bin centers and normalize histogram
        x = (x[0:-1] + x[1:]) / 2.0
        dx = x[1] - x[0]
        H = H / (np.sin(x) * dx)

        # Plot the histogram
        plt.plot(x * 180.0 / np.pi, H)
        plt.title("Elevation Histogram")
        plt.xlabel("Elevation (Deg)")
        plt.ylabel("Intensity")
        plt.savefig("./elevation_power_distribution.pdf")
        plt.show()

    def get_beam_width_half_power(self, points: int = 500, pole: List[float] = [0, 0, 1, 0]) -> None:
        """
        Collect only the elevation of all rays and calculate the beam width at half power.

        Parameters
        ----------
        points : int, optional
            The number of points to be used for binning, by default 500
        pole : List[float], optional
            The pole in Cartesian coordinates, by default [0, 0, 1, 0]
        """

        # Fetch measured results
        pos, pwr0 = self.get_measured_rays()

        # Normalize positions
        pos0 = np.array(np.divide(pos, np.matrix(np.linalg.norm(pos, axis=1)).T))

        # Convert power to float64
        pwr0 = np.float64(pwr0)

        # Calculate the angle between each position vector and the pole vector (the elevation)
        elevation0 = np.arccos(np.dot(pos0, pole))

        # Define a lambda function to sort indices
        SIDX = lambda s: sorted(range(len(s)), key=lambda k: s[k])

        # Get the sorted indices
        sort_idx = SIDX(elevation0)

        # Sort elevation and power by indices
        elevation = elevation0[sort_idx]
        pwr = pwr0[sort_idx]

        # Calculate cumulative sum of power
        pwr_cumsum = np.cumsum(pwr)

        # Calculate half max power
        pwr_hmax = pwr_cumsum[-1] / 2.0

        # Find the index closest to half max power
        pwr_hm_idx = np.where(np.absolute(pwr_cumsum - pwr_hmax) == min(np.absolute(pwr_cumsum - pwr_hmax)))

        # Calculate half max elevation and sum of power at half max
        elevation_hmax = elevation[pwr_hm_idx]
        pwr_sum_hmax = pwr_cumsum[pwr_hm_idx]

        # Print results
        print("\nThroughput results:")
        print("===================")
        print("Total measured power: ", pwr_cumsum[-1])
        print("Beam halfpower angle: ", elevation_hmax / np.pi * 180.0, "Deg")
        print("Halfwidth throughput: ", pwr_sum_hmax / pwr_cumsum[-1] * 100.0, "%")

    def get_beam_HWHM(self, points: int = 500, pole: List[float] = [0, 0, 1, 0]) -> None:
        """
        Collect only the elevation of all rays and calculate the half-width half-maximum (HWHM) of the beam.

        Parameters
        ----------
        points : int, optional
            The number of points to be used for binning, by default 500
        pole : List[float], optional
            The pole in Cartesian coordinates, by default [0, 0, 1, 0]
        """

        # Fetch measured results
        pos, pwr0 = self.get_measured_rays()

        # Normalize positions
        pos0 = np.array(np.divide(pos, np.matrix(np.linalg.norm(pos, axis=1)).T))

        # Convert power to float64 and calculate the angle between each position vector and the pole vector (the elevation)
        pwr0 = np.float64(pwr0)
        elevation0 = np.arccos(np.dot(pos0, pole))

        # Duplicate elevation for later use
        elevation1 = np.copy(elevation0)

        # Define a lambda function to sort indices
        SIDX = lambda s: sorted(range(len(s)), key=lambda k: s[k])

        # Get the sorted indices and sort elevation and intensity by them
        sort_idx = SIDX(elevation0)
        elevation = elevation0[sort_idx].flatten()
        intensity = pwr0[sort_idx].flatten()

        # Calculate histogram of elevations, weighted by intensities
        H, x = np.histogram(a=elevation, bins=points, weights=intensity)

        # Get the middle points of the bins and normalize H by sin(x)
        x = (x[0:-1] + x[1:]) / 2.0
        H = (H.T / np.sin(x)).flatten()

        # Find the index of half max power
        pwr_hmax = max(H) / 2.0
        pwr_hm_idx = np.where(np.absolute(H - pwr_hmax) == min(np.absolute(H - pwr_hmax)))[0]

        # Calculate half max elevation and sum of power at half max
        elevation_hmax = x[pwr_hm_idx]
        pwr_sum_hmax = np.sum(pwr0[np.where(elevation1 < elevation_hmax)])

        # Print results
        print("\nThroughput results HWHM:")
        print("========================")
        print("Total measured power: ", np.sum(pwr0))
        print("Beam HWHM angle: ", elevation_hmax / np.pi * 180.0, "Deg")
        print("HWHM throughput: ", pwr_sum_hmax / np.sum(pwr0) * 100.0, "%")

    def get_binned_data_stereographic(self,
                                      limits: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1, 1), (-1, 1)),
                                      points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stereographically project measured ray endpoints and bin them on the CL DEV.
        This is a lot faster when you have loads of data. Binning is done with 'points'
        number of points within limits=((xmin, xmax), (ymin, ymax)).

        Parameters
        ----------
        limits : Tuple[Tuple[float, float], Tuple[float, float]], optional
            The limits for binning in the format ((xmin, xmax), (ymin, ymax)), by default ((-1, 1), (-1, 1))
        points : int, optional
            The number of points to be used for binning, by default 500

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The histogram data and the x and y coordinates of the bins
        """

        # Fetch measured results
        pos0, pwr0 = self.get_measured_rays()

        # Convert data to device arrays
        pos0_dev = cl_array.to_device(self.queue, pos0.astype(np.float32))
        x_dev = cl_array.zeros(self.queue, pwr0.shape, dtype=np.float32)
        y_dev = cl_array.zeros(self.queue, pwr0.shape, dtype=np.float32)
        pwr0_dev = cl_array.to_device(self.queue, pwr0.astype(np.float32))
        pwr_dev = cl_array.zeros(self.queue, pwr0.shape, dtype=np.float32)
        pivot = cl_array.to_device(self.queue, np.array([0, 0, 0, 0], dtype=np.float32))

        # Start timing
        time1 = time()

        # Define rotation matrix and perform stereographic projection
        R_dev = cl_array.to_device(self.queue,
                                   np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).astype(
                                       np.float32))
        evt = self.prg.stereograph_project(self.queue, pwr0.shape, None, pos0_dev.data, pwr0_dev.data, R_dev.data,
                                           pivot.data, x_dev.data, y_dev.data, pwr_dev.data)

        # Wait for event to finish
        evt.wait()

        # Get data from device
        x = x_dev.get()
        y = y_dev.get()
        pwr = np.float64(pwr_dev.get())

        # End timing
        time2 = time()

        # Correct power for bin size
        dx = np.float64(limits[0][1] - limits[0][0]) / np.float64(points)
        dy = np.float64(limits[1][1] - limits[1][0]) / np.float64(points)
        pwr = pwr / (dx * dy)

        # Compute 2D histogram of positions, weighted by power
        H, x_coord, y_coord = np.histogram2d(x=x.flatten(), y=y.flatten(), bins=points, range=limits,
                                             weights=pwr.flatten())

        # Store the results and return them
        self.hist_data = (H, x_coord, y_coord)

        return self.hist_data

    def get_binned_data_angular(self, limits: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1, 1), (-1, 1)),
                                points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Azimuth/elevation map measured ray endpoints to a circle and bin them on the CL DEV.
        This linearly maps elevation to the circle's radius and azimuth to phi.
        Nice for cross-section plots of directivity. Binning is done with 'points' number
        of points within limits=((xmin, xmax), (ymin, ymax)).

        Parameters
        ----------
        limits : Tuple[Tuple[float, float], Tuple[float, float]], optional
            The limits for binning in the format ((xmin, xmax), (ymin, ymax)), by default ((-1, 1), (-1, 1))
        points : int, optional
            The number of points to be used for binning, by default 500

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The histogram data and the x and y coordinates of the bins
        """

        # Fetch measured results
        pos0, pwr0 = self.get_measured_rays()

        # Convert data to device arrays
        pos0_dev = cl_array.to_device(self.queue, pos0.astype(np.float32))
        x_dev = cl_array.zeros(self.queue, pwr0.shape, dtype=np.float32)
        y_dev = cl_array.zeros(self.queue, pwr0.shape, dtype=np.float32)
        pwr0_dev = cl_array.to_device(self.queue, pwr0.astype(np.float32))
        pwr_dev = cl_array.zeros(self.queue, pwr0.shape, dtype=np.float32)
        pivot = cl_array.to_device(self.queue, np.array([0, 0, 0, 0], dtype=np.float32))

        # Start timing
        time1 = time()

        # Define rotation matrix and perform angular projection
        R_dev = cl_array.to_device(self.queue,
                                   np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]).astype(
                                       np.float32))
        evt = self.prg.angular_project(self.queue, pwr0.shape, None, pos0_dev.data, pwr0_dev.data, R_dev.data,
                                       pivot.data, x_dev.data, y_dev.data, pwr_dev.data)

        # Wait for event to finish
        evt.wait()

        # Get data from device
        x = x_dev.get()
        y = y_dev.get()
        pwr = np.float64(pwr_dev.get())

        # End timing
        time2 = time()

        # Correct power for bin size
        dx = np.float64(limits[0][1] - limits[0][0]) / np.float64(points)
        dy = np.float64(limits[1][1] - limits[1][0]) / np.float64(points)
        pwr = pwr / (dx * dy)

        # Compute 2D histogram of positions, weighted by power
        H, x_coord, y_coord = np.histogram2d(x=x.flatten(), y=y.flatten(), bins=points, range=limits,
                                             weights=pwr.flatten())

        # Store the results
        self.hist_data = (H, x_coord, y_coord)

        # Return the histogram data and coordinates
        return self.hist_data

    def plot_binned_data(self, limits: Tuple[Tuple[float, float], Tuple[float, float]] = ((-10, 10), (-10, 10)),
                         points: int = 500, use_3d: bool = True, use_angular: bool = False,
                         hist_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> None:
        """
        Plot binned data in 2D or 3D.

        Parameters
        ----------
        limits : Tuple[Tuple[float, float], Tuple[float, float]], optional
            The range of data to be plotted, by default ((-10, 10), (-10, 10)).
        points : int, optional
            The number of data points, by default 500.
        use_3d : bool, optional
            Flag to indicate if 3D plot should be used, by default True.
        use_angular : bool, optional
            Flag to indicate if angular data should be used, by default False.
        hist_data : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], optional
            Histogram data to be plotted. If None, data is obtained using other methods, by default None.
        """

        # Calculate data if none is provided
        if hist_data is None:
            if use_angular:
                self.get_binned_data_angular(limits=limits, points=points)
                efact = 1.0
            else:
                pos, pwr = self.get_measured_rays()
                H, x_coord, y_coord = np.histogram2d(
                    x=np.array(pos[:, 0].flatten()),
                    y=np.array(pos[:, 1].flatten()),
                    bins=points,
                    range=limits,
                    density=False,
                    weights=np.array(np.float64(pwr).flatten())
                )
                self.hist_data = (H.astype(np.float64), x_coord, y_coord)
                efact = 1.0
        else:
            efact = 90.0 if use_angular else 1.0
            self.hist_data = hist_data

        H, xedges, yedges = self.hist_data
        xedges *= efact
        yedges *= efact

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Create a directory to save the figures
        Path("./figures").mkdir(parents=True, exist_ok=True)

        # Use 3D plot
        if use_3d:
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
            surf = ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.savefig(Path("./figures/binned_3D_data.pdf"))
            plt.show()
        # Use 2D plot
        else:
            plt.imshow(np.log10(H), extent=extent, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.savefig(Path("./figures/binned_2D_data.pdf"))
            plt.show()

    def pickle_results(self) -> Optional[Path]:
        """
        Save results and meshes to a pickle file.

        Returns:
            Optional[Path]: Path to the created pickle file if successful, None otherwise.
        """
        try:
            # Pickle results and meshes
            res_str = pickle.dumps((self.results, self.meshes), 1)

            # Define filename with a timestamp
            timestring = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
            fname = Path(f"./{timestring}-tracer_results.txt")

            # Open file and write data
            with fname.open("wb") as f:
                f.write(res_str)

            print(f"Pickled results and meshes to {fname}")
            return fname

        except Exception as e:
            print(f"Pickling results failed: {str(e)}")
            return None

    def load_pickle_results(self, path: Union[Path, str]) -> None:
        """
        Load pickled results from a specified file.

        Args:
            path (Union[Path, str]): Path to the pickle file.
        """
        path = Path(path)
        try:
            # Open and read data from file
            with path.open("rb") as f:
                res_str = f.read()

            # Unpickle data
            self.results, self.meshes = pickle.loads(res_str)

            print(f"Unpickled results and meshes from {path}")

        except Exception as e:
            print(f"Unpickling results failed: {str(e)}")

    def save_traced_scene(self, dxf_file: Union[Path, str]) -> None:
        """
        Writes the geometry and traced rays to a DXF file.

        Note: This method may not be efficient with more than a few hundred rays.
        More than 1000 rays could result in cluttered visuals.

        Args:
            dxf_file (Union[Path, str]): Name or path of the DXF file to be saved.
        """
        # Convert to Path object for consistency
        dxf_file = Path(dxf_file)
        # Create a new DXF drawing with the given file name
        drawing = dxf.drawing(str(dxf_file))

        # Add layers for rays and geometry with specified colors
        drawing.add_layer('Rays', color=3)
        drawing.add_layer('Geometry', color=5)

        print("Writing results to DXF file.")
        for res in self.results:
            rays_origin, rays_dest = res

            # Draw rays to DXF file
            for r0, rd in zip(rays_origin, rays_dest):
                # Create a 3D line from the ray origin to destination
                line3d = dxf.line(r0[0:3], rd[0:3], layer="Rays")
                # Add the line to the drawing
                drawing.add(line3d)

        # Draw facets
        m_v0, m_v1, m_v2 = self.geometry
        for t0, t1, t2 in zip(m_v0, m_v1, m_v2):
            # Create a 3D face from the vertices
            face3d = dxf.face3d([t0[0:3], t1[0:3], t2[0:3]], layer="Geometry")
            # Add the face to the drawing
            drawing.add(face3d)

        # Save the drawing to the DXF file
        drawing.save()

    def show(self, draw_wireframes: bool = False, ray_power_render_threshold_percentile: float = 0.9) -> None:
        """
        Visualizes rays using Open3D.

        Args:
            draw_wireframes: Whether to draw wireframes or not.
            ray_power_render_threshold_percentile: Threshold percentile for ray power rendering.
        """
        # Create a coordinate frame
        geometry = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])]

        # Prepare data for LineSet
        rays_origin, rays_dest, rays_pow = [], [], []
        for res in self.results:
            pow_tmp = res[2].ravel()
            idx = pow_tmp >= np.percentile(pow_tmp, ray_power_render_threshold_percentile)
            rays_origin.append(res[0][idx])
            rays_dest.append(res[1][idx])
            rays_pow.append(pow_tmp[idx])

        # Concatenate arrays
        rays_pow = np.concatenate(rays_pow)
        rays_origin = np.vstack(rays_origin)
        rays_dest = np.vstack(rays_dest)

        # Prepare points and lines for LineSet
        points = np.concatenate([rays_origin, rays_dest])
        lines = [[i, i + len(rays_origin)] for i in range(len(rays_origin))]

        # Create LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))  # Exclude the fourth coordinate
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Set color depending on ray power
        normalized_pow = (rays_pow - rays_pow.min()) / (rays_pow.max() - rays_pow.min())
        colors = np.zeros((len(rays_origin), 3))
        colors[:, 0] = 1
        colors[:, 1] = 1 - normalized_pow

        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometry.append(line_set)

        # Prepare data for TriangleMesh
        for m in self.meshes:
            geometry.append(m.get_open3d_mesh(return_wireframe=draw_wireframes))

        # Visualize
        o3d.visualization.draw_geometries(geometry)
