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
os.environ['EGL_PLATFORM'] = 'surfaceless'  # Ubuntu 20.04+
os.environ['OPEN3D_CPU_RENDERING'] = 'true'  # Ubuntu 18.04
import open3d as o3d

import numpy as np
from dxfwrite import DXFEngine as dxf
from typing import Callable, Tuple

from utils.Pose import Pose
from utils.rodrigues_from_vectors import calculate_rotation_vector


class LightSource:
    def __init__(
            self,
            center: np.ndarray = np.array([0, 0, 0, 0], dtype=np.float32),
            direction: Tuple[int, int, int] = (0, 0, 1),
            directivity: Callable[[int, int], np.ndarray] = lambda x, y: np.cos(y),
            power: int = 1,
            ray_count: int = 500
    ):
        """
        Class constructor for LightSource.

        :param center: Center coordinates. Defaults to an array of zeroes.
        :type center: np.ndarray
        :param direction: Direction of light source. Defaults to (0,0,1).
        :type direction: Tuple[int, int, int]
        :param directivity: Function of phi and theta measured from direction.
                            Defaults to cosine function.
        :type directivity: Callable[[int, int], np.ndarray]
        :param power: Total power of light source (sum over all rays). Defaults to 1.
        :type power: int
        :param ray_count: Total ray count when asked for a set of rays. Defaults to 500.
        :type ray_count: int
        """
        self.center = center
        self.direction = direction
        self.directivity = directivity
        self.power = power
        self.ray_count = ray_count
        self.random_rays()

    def grid_rays(self) -> None:
        """
        Generate N equidistant rays that sample a hemisphere.

        The function updates the ray_count property to be a perfect square.
        It generates spherical coordinates and converts them to Cartesian coordinates.
        The computed rays are transformed by the function compute_and_transform_rays.
        """
        N: int = int(np.sqrt(self.ray_count))
        self.ray_count = N ** 2

        # Generate spherical coordinates
        phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, N), np.linspace(0, np.pi / 2, int(N)))

        # Convert to Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        self.compute_and_transform_rays(x, y, z)

    def compute_and_transform_rays(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """
        Compute and transform rays given Cartesian coordinates.

        The method computes a pose transform and combines x, y, z coordinates to form rays.
        It also calculates rays' power and ensures homogeneous coordinates.

        Args:
            x: Numpy ndarray representing the x component of the Cartesian coordinates.
            y: Numpy ndarray representing the y component of the Cartesian coordinates.
            z: Numpy ndarray representing the z component of the Cartesian coordinates.
        """
        # Compute pose transform
        r = calculate_rotation_vector(np.array([0, 0, 1]), np.array(self.direction))
        P = Pose.from_axis_angle(r, self.center)

        # Combine x, y, and z components to form the rays
        self.rays_dir = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1) @ P.r.T
        self.rays_origin = P.transform(np.zeros_like(self.rays_dir))
        self.rays_dest = self.rays_origin + self.rays_dir

        # Calculate rays' power
        self.rays_power = np.array(
            self.directivity(
                np.arctan2(y, x),
                np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
            ).ravel(),
            dtype=np.float32
        )

        self.rays_power *= self.power / np.sum(self.rays_power)

        self.ensure_homogeneous_coordinates()
    def random_rays(self) -> None:
        """
        Setup rays in a random pattern.

        The function applies the method of sphere picking to generate random rays.
        The rays are then transformed using the method compute_and_transform_rays.
        """
        print("Setting up rays in random pattern")

        # Sphere picking
        u = np.random.rand(self.ray_count, 1)
        v = np.random.rand(self.ray_count, 1)

        elevation = np.arccos(u)  # only rays in +z are desired. Otherwise use 2*u-1
        azimuth = 2.0 * np.pi * v

        x = np.sin(elevation) * np.cos(azimuth)
        y = np.sin(elevation) * np.sin(azimuth)
        z = np.cos(elevation)

        # Compute pose transform
        self.compute_and_transform_rays(x, y, z)

    def random_collimated_rays(self, diameter: float = 1.0) -> None:
        """
        Setup collimated rays in a random pattern.

        The method generates random rays with a given diameter, computes a pose transform,
        and combines x, y, z coordinates to form rays. It also calculates rays' power and ensures homogeneous coordinates.

        Args:
            diameter: A float indicating the diameter of the rays. Default is 1.0.
        """
        print("Setting up collimated rays in random pattern")
        x = (np.random.rand(self.ray_count, 1) - 0.5) * diameter
        y = (np.random.rand(self.ray_count, 1) - 0.5) * diameter
        z0 = np.zeros((self.ray_count, 1))
        z1 = np.ones((self.ray_count, 1))

        # Compute pose transform
        r = calculate_rotation_vector(np.array([0, 0, 1]), np.array(self.direction))
        P = Pose.from_axis_angle(r, self.center)

        # Combine x, y, and z components to form the rays
        self.rays_origin = P.transform(np.stack((x.ravel(), y.ravel(), z0.ravel()), axis=-1))
        self.rays_dest = P.transform(np.stack((x.ravel(), y.ravel(), z1.ravel()), axis=-1))
        self.rays_dir = self.rays_dest - self.rays_origin

        # Calculate rays' power
        self.rays_power = np.ones_like(x)
        self.rays_power *= self.power / np.sum(self.rays_power)

        self.ensure_homogeneous_coordinates()

    def ensure_homogeneous_coordinates(self) -> None:
        """
        Ensure that the rays' coordinates are homogeneous.

        This method checks the shape of rays' origin, destination, and direction.
        If they are not in homogeneous coordinates (4D), it adds an additional dimension
        filled with ones, effectively converting them into homogeneous coordinates.
        """
        # Check if rays_origin is 3D and convert to 4D if necessary
        if self.rays_origin is not None and self.rays_origin.shape[1] == 3:
            self.rays_origin = np.hstack((self.rays_origin, np.ones((self.rays_origin.shape[0], 1))))

        # Check if rays_dest is 3D and convert to 4D if necessary
        if self.rays_dest is not None and self.rays_dest.shape[1] == 3:
            self.rays_dest = np.hstack((self.rays_dest, np.ones((self.rays_dest.shape[0], 1))))

        # Check if rays_dir is 3D and convert to 4D if necessary
        if self.rays_dir is not None and self.rays_dir.shape[1] == 3:
            self.rays_dir = np.hstack((self.rays_dir, np.ones((self.rays_dir.shape[0], 1))))

    def save_dxf(self, dxf_file: str) -> None:
        """
        Save rays to a DXF file.

        This method generates a DXF drawing with a layer for rays,
        then writes the rays' origin and direction to the drawing.

        Args:
            dxf_file: A string representing the file path to save the DXF drawing.
        """
        # Initialize a new DXF drawing
        drawing = dxf.drawing(dxf_file)

        # Add a layer for rays to the drawing
        drawing.add_layer('Rays', color=3)

        print("Writing rays to DXF file.")

        # Write each ray to the DXF file
        for (r0, rd) in zip(self.rays_origin, self.rays_origin + self.rays_dir):
            # Add a 3D face representing the ray to the drawing
            drawing.add(dxf.face3d([r0[0:3],
                                    rd[0:3],
                                    rd[0:3]], layer="Rays"))

        # Save the DXF drawing
        drawing.save()
    def show(self) -> None:
        """
        Visualize rays using open3d.

        The method prepares data for LineSet using the rays' origin and direction,
        and visualizes the rays using open3d's draw_geometries function.
        """
        # Prepare data for LineSet
        red_color = np.array([1.0, 0.0, 0.0])
        line_set = self.rays_to_lineset(self.rays_dir, self.rays_origin, red_color)

        # Visualize rays
        o3d.visualization.draw_geometries([
            line_set,
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        ])

    def rays_to_lineset(self, rays_dir: np.ndarray, rays_origin: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
        """
        Convert rays to a LineSet for visualization in open3d.

        This method takes the direction and origin of rays and a color as input,
        and creates a LineSet object for visualization.

        Args:
            rays_dir: Numpy array representing the direction of the rays.
            rays_origin: Numpy array representing the origin of the rays.
            color: Numpy array representing the color of the rays.

        Returns:
            An open3d.geometry.LineSet object for visualization.
        """
        # Create points array by concatenating rays_origin and rays_origin + rays_dir
        points = np.concatenate([rays_origin, rays_origin + rays_dir])

        # Create lines array by creating pairs of indices for each ray
        lines = [[i, i + len(rays_origin)] for i in range(len(rays_origin))]

        # Create LineSet and populate it with points and lines
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))  # Exclude the fourth coordinate
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Create colors array by repeating the input color for each ray and assign to LineSet
        colors = np.tile(color, (len(rays_origin), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

if __name__ == "__main__":
    # Instantiate a LightSource object with given parameters
    ls0 = LightSource(center=np.array([-2, 0, 0], dtype=np.float32),
                      direction=(0.0, 0.0, 1.0),
                      directivity=lambda x, y: 1.0 + np.cos(y),
                      power=1000.,
                      ray_count=100)
    ls1 = LightSource(center=np.array([0, 0, 0], dtype=np.float32),
                      direction=(0.0, 0.0, 1.0),
                      directivity=lambda x, y: 1.0 + np.cos(y),
                      power=1000.,
                      ray_count=100)
    ls2 = LightSource(center=np.array([2, 0, 0], dtype=np.float32),
                      direction=(0.0, 0.0, 1.0),
                      directivity=lambda x, y: 1.0 + np.cos(y),
                      power=1000.,
                      ray_count=100)

    ls0.grid_rays()
    ls1.random_rays()
    ls2.random_collimated_rays()

    scene = [ls0.rays_to_lineset(rays_dir=ls0.rays_dir,rays_origin=ls0.rays_origin,color=np.array([1,0,0])),
             ls1.rays_to_lineset(rays_dir=ls1.rays_dir,rays_origin=ls1.rays_origin,color=np.array([1,0,0])),
             ls2.rays_to_lineset(rays_dir=ls2.rays_dir,rays_origin=ls2.rays_origin,color=np.array([1,0,0])),
             o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])]

    o3d.visualization.draw_geometries(scene)

