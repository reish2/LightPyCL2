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

import open3d as o3d

import numpy as np
from dxfwrite import DXFEngine as dxf
import triangle
from utils.Pose import Pose
from typing import Optional, Dict, Union, Tuple, List
from numpy.typing import ArrayLike

class GeoObject:
    """
    The GeoObject class provides structure to store meshes and perform simple geometric operations
    such as translation and rotation. It also provides conversion methods required by the tracer
    kernel and DXF output.
    """

    def __init__(
            self,
            verts: np.ndarray,
            tris: np.ndarray,
            mat_type: str = "refractive",
            index_of_refraction: float = 1.0,
            reflectivity: float = 1.0,
            dissipation: float = 0.0,
            anti_refraction_coating_index_of_refraction: float = 1.0,
            anti_reflection_coating_thickness: float = 0.0,
            anisotropy: Optional[np.ndarray] = None
    ) -> None:
        """
        Initializes the GeoObject instance.

        Parameters:
        verts (np.ndarray): The vertices of the GeoObject.
        tris (np.ndarray): The triangles of the GeoObject.
        mat_type (str): The material type of the GeoObject. Defaults to "refractive".
        index_of_refraction (float): The Index Of Refraction of the GeoObject. Defaults to 1.0.
        reflectivity (float): The reflectivity of the GeoObject. Defaults to 1.0.
        dissipation (float): The dissipation of the GeoObject in 1/m. Defaults to 0.0.
        anti_refraction_coating_index_of_refraction (float): The Anti-Reflective Index Of Refraction of the GeoObject. Defaults to 1.0.
        anti_reflection_coating_thickness (float): The Anti-Reflective thickness of the GeoObject. Defaults to 0.0.
        anisotropy (np.ndarray): The anisotropy of the GeoObject. Defaults to None.

        Returns:
        None
        """
        # Material types mapping
        self.mat_types = {
            "refractive": 0,
            "mirror": 1,
            "terminator": 2,
            "measure": 3,
            "refractive_anisotropic": 4
        }

        # Material types mapping
        self.mat_colors = {
            "refractive": np.array([0,0.5,1]),
            "mirror": np.array([0,1,1]),
            "terminator": np.array([1,0,1]),
            "measure": np.array([1,0.33,1]),
            "refractive_anisotropic": np.array([0.5,0,1])
        }

        # Initializing the instance variables
        self.mat_type = mat_type
        self.index_of_refraction = index_of_refraction
        self.reflectivity = reflectivity
        self.dissipation = dissipation  # in 1/m
        self.anti_refraction_coating_index_of_refraction = anti_refraction_coating_index_of_refraction  # not in use yet
        self.anti_reflection_coating_thickness = anti_reflection_coating_thickness  # not in use yet
        self.anisotropy = anisotropy  # not in use yet
        self.vertices = verts
        self.triangles = tris

        # Setting the material
        self.set_material(mat_type, index_of_refraction, reflectivity, dissipation, anti_refraction_coating_index_of_refraction, anti_reflection_coating_thickness, anisotropy)

    def set_material(
            self,
            mat_type: str = "refractive",
            index_of_refraction: float = 1.0,
            reflectivity: float = 1.0,
            dissipation: float = 0.0,
            anti_refraction_coating_index_of_refraction: float = 1.0,
            anti_reflection_coating_thickness: float = 0.0,
            anisotropy: Optional[np.ndarray] = None
    ) -> None:
        """
        Sets the material of the GeoObject instance.

        Parameters:
        mat_type (str): The material type of the GeoObject. Defaults to "refractive".
        index_of_refraction (float): The Index Of Refraction of the GeoObject. Defaults to 1.0.
        reflectivity (float): The reflectivity of the GeoObject. Defaults to 1.0.
        dissipation (float): The dissipation of the GeoObject in 1/m. Defaults to 0.0.
        anti_refraction_coating_index_of_refraction (float): The Anti-Reflective coating Index Of Refraction of the GeoObject. Defaults to 1.0.
        anti_reflection_coating_thickness (float): The Anti-Reflective coating thickness of the GeoObject. Defaults to 0.0.
        anisotropy (np.ndarray): The anisotropy of the GeoObject. Defaults to None.

        Returns:
        None
        """
        # Check if the provided material type is known
        if mat_type in self.mat_types:
            self.mat_type = mat_type
        else:
            # Set the material type to default if the provided one is unknown
            default = "refractive"
            self.mat_type = default
            print(f"Warning: material {mat_type} unknown. Setting material as {default}.")

        # Set the material properties based on the material type
        if self.mat_type == "refractive":
            self.index_of_refraction = index_of_refraction
            self.dissipation = dissipation
            self.anti_refraction_coating_index_of_refraction = anti_refraction_coating_index_of_refraction
            self.anti_reflection_coating_thickness = anti_reflection_coating_thickness
        elif self.mat_type == "mirror":
            self.reflectivity = reflectivity
        elif self.mat_type == "refractive_anisotropic":
            print("Warning: anisotropic materials are not yet supported.")

    def get_material_buf(self) -> Dict[str, Union[int, float]]:
        """
        Returns a dictionary of material parameters for the tracer code.

        Returns:
        dict: A dictionary containing the material type, Index Of Refraction,
        reflectivity and dissipation of the GeoObject.
        """
        return {
            "type": self.mat_types.get(self.mat_type),
            "index_of_refraction": self.index_of_refraction,
            "reflectivity": self.reflectivity,
            "dissipation": self.dissipation
        }

    def transform(self, P: Pose) -> None:
        """
        Applies a transformation to the vertices of the GeoObject using
        a Pose object.

        Args:
        P (Pose): A Pose object representing a transformation.

        Returns:
        None
        """
        self.vertices = P.transform(self.vertices)

    def tribuf(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Generate 3 buffers for PyOpenCL tracer code.
        Each buffer contains one vertex of a triangle.
        Therefore, m_v0[i], m_v1[i] and m_v2[i] form a triangle.

        Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike]: Three arrays, each containing
        the vertices of a triangle.
        """
        m_v0 = self.vertices[self.triangles[:, 0], :]
        m_v1 = self.vertices[self.triangles[:, 1], :]
        m_v2 = self.vertices[self.triangles[:, 2], :]
        return m_v0, m_v1, m_v2

    def append(self, verts: ArrayLike, tris: ArrayLike) -> None:
        """
        Append mesh data to existing GeoObject by appending vertices to
        self.vertices and index correcting triangle indices.

        Args:
        verts (ArrayLike): The vertices to be appended.
        tris (ArrayLike): The triangle indices to be appended.
        """
        self.triangles = np.append(self.triangles,
                                   np.array(tris).astype(np.int32) + len(self.vertices),
                                   axis=0)
        self.vertices = np.append(self.vertices, verts, axis=0)

    def write_dxf(self, dxf_file: str) -> None:
        """
        Write geometry to a DXF file using the dxfwrite library.

        Args:
        dxf_file (str): The path to the output DXF file.
        """
        drawing = dxf.drawing(dxf_file)
        drawing.add_layer('0', color=2)

        for tri in self.triangles:
            drawing.add(dxf.face3d([self.vertices[tri[0]][0:3],
                                    self.vertices[tri[1]][0:3],
                                    self.vertices[tri[2]][0:3]], layer="0"))
        drawing.save()

    def get_open3d_mesh(self, return_wireframe=False) -> Union[
        o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
        """
        Returns an Open3D TriangleMesh or LineSet object representation of the geometric object.
        This mesh is rendereable with Open3D using `o3d.visualization.draw_geometries([mesh])`.

        Parameters:
            color (numpy.array): A color for the mesh.
            return_wireframe (bool): A flag that specifies if a wireframe should be returned.

        Returns:
            o3d.geometry.TriangleMesh or o3d.geometry.LineSet: The Open3D TriangleMesh or LineSet object.
        """
        # Prepare data for TriangleMesh or LineSet
        m_v0 = self.vertices[self.triangles[:, 0], :]
        m_v1 = self.vertices[self.triangles[:, 1], :]
        m_v2 = self.vertices[self.triangles[:, 2], :]

        # Concatenate all vertices
        vertices = np.concatenate([m_v0[:, :3], m_v1[:, :3], m_v2[:, :3]])

        # color
        color = self.mat_colors[self.mat_type]

        if return_wireframe:
            # Arrange the vertices to create lines
            triangles_l1 = np.stack((np.arange(len(m_v0)), len(m_v0) + np.arange(len(m_v0)))).transpose((1, 0))
            triangles_l2 = np.stack((len(m_v0) + np.arange(len(m_v0)), 2 * len(m_v0) + np.arange(len(m_v0)))).transpose(
                (1, 0))
            triangles_l3 = np.stack((2 * len(m_v0) + np.arange(len(m_v0)), np.arange(len(m_v0)))).transpose((1, 0))
            triangles3 = np.vstack((triangles_l1, triangles_l2, triangles_l3))

            # Create LineSet for wireframe
            wireframe_mesh = o3d.geometry.LineSet()
            wireframe_mesh.points = o3d.utility.Vector3dVector(vertices)  # Exclude the fourth coordinate
            wireframe_mesh.lines = o3d.utility.Vector2iVector(triangles3)

            # Set color
            colors = np.tile(color, (3 * len(m_v0), 1))
            wireframe_mesh.colors = o3d.utility.Vector3dVector(colors)

            return wireframe_mesh
        else:
            # Arrange the vertices to create two-faced triangles :)
            triangles1 = np.stack((np.arange(len(m_v0)), len(m_v0) + np.arange(len(m_v0)),
                                   2 * len(m_v0) + np.arange(len(m_v0)))).transpose((1, 0))
            triangles2 = np.stack((2 * len(m_v0) + np.arange(len(m_v0)), len(m_v0) + np.arange(len(m_v0)),
                                   np.arange(len(m_v0)))).transpose((1, 0))
            triangles = np.vstack((triangles1, triangles2))

            # Create TriangleMesh
            triangle_mesh = o3d.geometry.TriangleMesh()
            triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            triangle_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            triangle_mesh.paint_uniform_color(color)  # Set color to blue
            triangle_mesh.compute_vertex_normals()

            return triangle_mesh

    def show(self) -> None:
        """
        Visualizes rays using Open3D.
        """

        # Create TriangleMesh
        triangle_mesh = self.get_open3d_mesh()

        # Create LineSet
        wireframe_mesh = self.get_open3d_mesh()

        # Visualize
        o3d.visualization.draw_geometries([triangle_mesh,
                                           o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15,
                                                                                             origin=[0, 0, 0])])


class OpticalElements:
    """
    The OpticalElements class provides generator methods that create mesh objects from basic
    parameters of the corresponding optical elements. Optical element generators return a GeoObject
    with vertices, triangle indices and index of refraction properties.
    """

    def __init__(self, mesh_angular_resolution=2*np.pi/300):
        self.mesh_angular_resolution = mesh_angular_resolution

    @staticmethod
    def cube(center: np.ndarray, size: float) -> GeoObject:
        """
        Generates a cube geometry given a center point and a size.

        Args:
            center (np.ndarray): Center coordinates of the cube.
            size (float): Size of the cube.

        Returns:
            GeoObject: Cube mesh as a GeoObject.
        """
        # Define vertices of the cube
        verts = np.array([[-1, -1, -1, 1],
                          [1, -1, -1, 1],
                          [-1, 1, -1, 1],
                          [1, 1, -1, 1],
                          [-1, -1, 1, 1],
                          [1, -1, 1, 1],
                          [-1, 1, 1, 1],
                          [1, 1, 1, 1]],
                         dtype=np.float32) / 2.0 * size + center

        # Define triangles of the cube
        triangles = np.array([[0, 1, 2], [2, 3, 1],  # bottom
                              [4, 5, 6], [6, 7, 5],  # top
                              [0, 1, 4], [4, 5, 1],  # rear
                              [2, 3, 6], [6, 7, 3],  # front
                              [0, 2, 4], [4, 6, 2],  # left
                              [1, 3, 5], [5, 7, 3]],  # right
                             dtype=int)

        # Create GeoObject mesh of the cube
        mesh = GeoObject(verts, triangles)
        return mesh
    def spherical_lens_nofoc(self, r1: float, r2: float, x1: float, x2: float, d: float, d2: Optional[float] = None,
                             sign1_arcsin: float = 1.0, sign2_arcsin: float = 1.0) -> GeoObject:
        """
        Generates a mesh for a spherical lens without focus.

        Args:
            r1 (float): Radius of the anterior lens.
            r2 (float): Radius of the posterior cornea.
            x1 (float): x-coordinate for the center of the anterior lens.
            x2 (float): x-coordinate for the center of the posterior cornea.
            d (float): Distance from the origin to the anterior lens.
            d2 (Optional[float]): Distance from the origin to the posterior cornea. If None, d is used.
            sign1_arcsin (float): Sign for the arcsin of the anterior lens. Default is 1.0.
            sign2_arcsin (float): Sign for the arcsin of the posterior cornea. Default is 1.0.

        Returns:
            GeoObject: Mesh for the spherical lens as a GeoObject.
        """
        N_min = 50
        d2 = d if d2 is None else d2
        z1 = r1 + x1  # circle center of anterior lens
        z2 = r2 + x2  # circle center of posterior cornea
        dphi1 = np.pi / 2.0 if sign1_arcsin < 0 else 0.0
        dphi2 = np.pi / 2.0 if sign2_arcsin < 0 else 0.0

        N1 = max(N_min,int(np.pi / (2*self.mesh_angular_resolution)))
        phi1 = np.linspace(0.0, np.absolute(np.arcsin(d / r1)) + dphi1, N1)
        N2 = max(N_min, int(np.pi / (2*self.mesh_angular_resolution)))
        phi2 = np.linspace(np.absolute(np.arcsin(d2 / r2)) + dphi2, 0.0, N2)
        xc_1 = z1 - r1 * np.cos(phi1)
        yc_1 = np.abs(r1 * np.sin(phi1))
        xc_2 = z2 - r2 * np.cos(phi2)
        yc_2 = np.abs(r2 * np.sin(phi2))
        xc = np.append(xc_1, xc_2)
        yc = np.append(yc_1, yc_2)

        z = np.zeros_like(xc)
        a = np.ones_like(xc)
        xyz = np.vstack((xc, yc, z, a)).T

        N3 = max(N_min, int(2*np.pi / self.mesh_angular_resolution))
        mesh = self.revolve_curve(xyz, direction=np.array([1, 0, 0]), ang=2.0*np.pi, ang_pts=N3)
        mesh.transform(Pose.from_axis_angle(np.array([0, 1, 0])*-np.pi / 2.0, np.array([0,0,0])))
        return mesh

    def sphere(self, center: np.ndarray, radius: float) -> GeoObject:
        """
        Generates a mesh for a sphere.

        Args:
            center (np.ndarray): 1D array representing the coordinates of the center of the sphere.
            radius (float): Radius of the sphere.

        Returns:
            GeoObject: Mesh for the sphere as a GeoObject.
        """
        N_min = 72
        N1 = max(N_min, int(2*np.pi / self.mesh_angular_resolution))
        phi = np.linspace(0.0, 2.0 * np.pi, N1)
        x = np.cos(phi) * radius
        y = np.sin(phi) * radius
        z = np.zeros_like(x)
        a = np.ones_like(x)
        xyz = np.vstack((x, y, z, a)).T

        N2 = max(N_min, int(np.pi / self.mesh_angular_resolution))
        mesh = self.revolve_curve(xyz, direction=np.array([1, 0, 0]), ang=np.pi, ang_pts=N2 + 1)
        mesh.transform(Pose.from_axis_angle(np.array([0, 0, 0]), np.array(center)))
        return mesh

    def hemisphere(self, center: np.ndarray, radius: float) -> GeoObject:
        """
        Generates a mesh for a hemisphere.

        Args:
            center (np.ndarray): 1D array representing the coordinates of the center of the hemisphere.
            radius (float): Radius of the hemisphere.

        Returns:
            GeoObject: Mesh for the hemisphere as a GeoObject.
        """
        N_min = 72
        N1 = max(N_min, int(np.pi / (2*self.mesh_angular_resolution)))
        phi = np.linspace(0.0, np.pi / 2, N1)
        x = np.cos(phi) * radius
        z = np.sin(phi) * radius
        y = np.zeros_like(x)
        a = np.ones_like(x)
        xyz = np.vstack((x, y, z, a)).T

        N2 = max(N_min, int(2*np.pi / self.mesh_angular_resolution))
        mesh = self.revolve_curve(xyz, direction=np.array([0, 0, 1]), ang=2 * np.pi, ang_pts=N2 + 1)
        mesh.transform(Pose.from_axis_angle(np.array([0, 0, 0]), np.array(center)))
        return mesh

    def parabolic_mirror(self, focus: tuple = (0, 0, 0), focal_length: float = 5.0,
                         diameter: float = 20.0, reflectivity: float = 0.98) -> GeoObject:
        """
        Generates a mesh for a parabolic mirror.

        Args:
            focus (tuple): Tuple representing the coordinates of the focus of the parabola.
            focal_length (float): Focal length of the parabolic mirror.
            diameter (float): Diameter of the parabolic mirror.
            reflectivity (float): Reflectivity of the parabolic mirror.

        Returns:
            GeoObject: Mesh for the parabolic mirror as a GeoObject.
        """
        M_min = 200
        M1 = max(M_min, int(np.pi / (2 * self.mesh_angular_resolution)))
        yn = np.linspace(0.0, diameter / 2.0, M1)
        xn = yn ** 2 / (4.0 * focal_length) - focal_length

        x = focus[0] + xn
        y = focus[1] + yn

        z = np.zeros_like(x)
        a = np.ones_like(x)
        xyz = np.vstack((x, y, z, a)).T

        N_min = 72
        N1 = max(N_min, int(2*np.pi / (self.mesh_angular_resolution)))
        mesh = self.revolve_curve(xyz, direction=np.array([1, 0, 0]), ang=2 * np.pi, ang_pts=N1 + 1)
        mesh.set_material(mat_type="mirror", reflectivity=reflectivity)
        return mesh

    def revolve_curve(self, curve3d: np.array, direction: np.array = np.array([1, 0, 0]),
                      ang: float = 2 * np.pi, ang_pts: int = 36) -> GeoObject:
        """
        Revolves a curve around an axis to create a surface.

        Args:
            curve3d (np.array): Array of 3D points forming the curve to be revolved.
            direction (np.array): Array representing the axis of revolution.
            ang (float): Angle over which to revolve the curve (default is 2pi, for full revolution).
            ang_pts (int): The number of points in the angle over which the curve is to be revolved.

        Returns:
            GeoObject: A mesh representing the revolved curve as a GeoObject.
        """
        tris = []
        angs = np.linspace(0.0, ang, ang_pts)
        curves = []
        for k, ang in enumerate(angs):
            P = Pose.from_axis_angle(direction * ang, np.array([0, 0, 0]))
            curves.append(P.transform(curve3d))

            # base_idx generates an array of sequential numbers from 0 to one less than the length of the curve.
            # It's used to create indexes for vertices that will be used in creating triangles.
            base_idx = np.arange(len(curve3d) - 1)
            v1 = (base_idx + 0) + ((k + 0) % ang_pts) * (len(base_idx) + 1)
            v2 = (base_idx + 1) + ((k + 0) % ang_pts) * (len(base_idx) + 1)
            v3 = (base_idx + 0) + ((k + 1) % ang_pts) * (len(base_idx) + 1)
            v4 = (base_idx + 1) + ((k + 1) % ang_pts) * (len(base_idx) + 1)
            tris.append(np.vstack((v1, v2, v3, v3, v4, v2)).T)

        verts = np.vstack(curves)
        tris = np.vstack(tris).reshape((-1, 3))

        return GeoObject(verts, tris)

    def extrude_by_vector(self, curve: np.ndarray, vector: np.ndarray, capped: bool = True) -> GeoObject:
        """
        Extrudes a 2D curve along a specified vector, optionally capping the ends.

        Args:
            curve (np.ndarray): Array of 2D points forming the curve to be extruded.
            vector (np.ndarray): The direction and magnitude of the extrusion.
            capped (bool): Whether to cap the ends of the extrusion.

        Returns:
            GeoObject: A mesh representing the extruded curve as a GeoObject.
        """

        # Create extrusion surface with open ends
        # Initialize the z-coordinate as zero and 'a' as ones for each point in the curve
        z0 = 0.0
        z = np.full((len(curve), 1), z0)
        a = np.ones((len(curve), 1))

        # Define the base curve (v0) and the cap curve (v1) of the extrusion
        v0 = np.hstack((curve[:, :2], z, a))
        v1 = v0.copy()
        v1[:, :3] += vector[:3]

        # Combine the vertices of the base and cap curves
        verts = np.vstack((v0, v1))

        # Generate an array of sequential numbers from 0 to one less than the length of the curve
        base_idx = np.arange(len(curve) - 1)

        # Calculate indices for vertices that will be used to create triangles
        v1 = base_idx
        v2 = base_idx + 1
        v3 = v1 + len(base_idx) + 1
        v4 = v2 + len(base_idx) + 1

        # Stack triangle vertices to create a new triangle surface
        tris = np.vstack((np.stack((v1, v2, v3), axis=1), np.stack((v3, v4, v2), axis=1)))

        # Create a GeoObject from the vertices and triangles
        gobj = GeoObject(verts, tris)

        # If the 'capped' argument is True, add end caps to close the extrusion
        if capped:
            trisurf = self.curve_to_mesh(curve[:, :2])
            cap1 = trisurf.vertices.copy()
            cap2 = cap1.copy()
            cap2[:, :3] += vector[:3]
            gobj.append(cap1, trisurf.triangles.copy())
            gobj.append(cap2, trisurf.triangles.copy())

        return gobj

    def curve_to_mesh(self, curve: np.ndarray) -> GeoObject:
        """
        Transforms a curve into a triangular mesh.

        Args:
            curve (np.ndarray): Array of 2D points forming the curve to be transformed.

        Returns:
            GeoObject: A mesh representing the curve as a GeoObject.
        """

        # Number of points in the curve
        M = len(curve)

        # Initialize segments as zeros with two columns
        segs = np.zeros((M, 2), dtype=np.int32)

        # Fill the first column with sequential numbers and the second column with sequential numbers shifted by one
        segs[:, 0] = np.arange(M)
        segs[:, 1] = segs[:, 0] + 1
        segs[-1, -1] = 0  # Link the last point to the first

        # Create the face object with vertices and segments
        face = {"vertices": curve.astype(np.float32), "segments": segs}

        # Triangulate the face with a triangle library
        tri = triangle.triangulate(face, 'pq10')

        # Get 2D vertices and triangles from the triangulation result
        verts_2d = tri["vertices"]
        triangles_2d = tri["triangles"]

        # Initialize 4D vertices and 3D triangles with zeros
        verts = np.zeros((len(verts_2d), 4), dtype=np.float32)
        triangles = np.zeros((len(triangles_2d), 3), dtype=np.int32)

        # Fill 4D vertices and 3D triangles with the values from 2D counterparts
        verts[:, :2] = verts_2d
        triangles[:, :] = triangles_2d
        verts[:, 3] = 1  # Set homogeneous coordinate to 1

        # Return a GeoObject with the vertices and triangles
        return GeoObject(verts=verts, tris=triangles)

    def lens_spherical_biconcave(self,
                                 focus: np.array,
                                 r1: float,
                                 r2: float,
                                 diameter: float,
                                 IOR: float) -> GeoObject:
        """
        Creates a biconcave spherical lens.

        Args:
            focus (np.array): The distance from the lens at which light from a point on the axis will be focused.
            r1 (float): Radius of the first spherical surface of the lens.
            r2 (float): Radius of the second spherical surface of the lens.
            diameter (float): The diameter of the lens.
            IOR (float): Index of Refraction of the lens material.

        Returns:
            Mesh: A Mesh object representing the lens.
        """

        # Generate a spherical lens curve
        xyz = self.lens_spherical_2r(focus, r1, r2, diameter, 1, IOR)

        # Revolve the curve around the x-axis to create the lens surface
        mesh = self.revolve_curve(xyz,
                                  direction=np.array([1, 0, 0]),
                                  ang=2*np.pi,
                                  ang_pts=36)

        # Set the material properties of the lens
        mesh.set_material(mat_type="refractive",
                          index_of_refraction=IOR)

        return mesh

    def lens_spherical_2r(self,
                          focus: np.array,
                          r1: float,
                          r2: float,
                          diameter: float,
                          lens_sign: int,
                          n: int) -> np.array:
        """
        Generates a spherical lens curve.

        Args:
            focus (np.array): The focus point of the lens.
            r1 (float): Radius of the first spherical surface of the lens.
            r2 (float): Radius of the second spherical surface of the lens.
            diameter (float): The diameter of the lens.
            lens_sign (int): The sign of the lens.
            n (int): A parameter related to the lens characteristics.

        Returns:
            np.array: The array representing the lens curve.
        """
        N_min = 72
        N1 = max(N_min, int(np.pi / (2 * self.mesh_angular_resolution)))
        fx0, fy0 = focus[0], focus[1]

        phi_r1 = np.arcsin(diameter / r1)
        phi_r2 = np.arcsin(diameter / r2)

        d = np.abs(r1 - r1 * np.cos(phi_r1)) + np.abs(r2 - r2 * np.cos(phi_r2))

        f = np.abs(1 / ((n - 1) * (1 / r1 - 1 / r2 + (n - 1) * d / (n * r1 * r2))))

        q = (f - r1)  # distance from f to center of r1 circle

        r1x0 = fx0 + q
        r1y0 = fy0
        r2x0 = fx0 + q + r1 - lens_sign * d + r2
        r2y0 = fy0

        phi1 = np.linspace(np.pi - np.abs(phi_r2), np.pi, N1)
        xy1 = np.vstack((r2x0 + r2 * np.cos(phi1), r2y0 + r2 * np.sin(phi1))).T

        phi2 = np.linspace(0, np.abs(phi_r1), N1)
        xy2 = np.vstack((r1x0 + r1 * np.cos(phi2), r1y0 + r1 * np.sin(phi2))).T

        xy3 = xy1[0, :].T

        xy = np.vstack((xy1, xy2, xy3))

        xyza = np.hstack((xy, np.zeros((len(xy), 1)), np.ones((len(xy), 1))))

        return xyza


    def curve_circle(self, radius: float):
        """
        Generates a mesh for a sphere.

        Args:
            center (np.ndarray): 1D array representing the coordinates of the center of the sphere.
            radius (float): Radius of the sphere.

        Returns:
            GeoObject: Mesh for the sphere as a GeoObject.
        """
        N_min = 72
        N1 = max(N_min, int(2*np.pi / ( self.mesh_angular_resolution)))
        phi = np.linspace(0.0, 2.0 * np.pi, N1)
        x = np.cos(phi) * radius
        y = np.sin(phi) * radius
        z = np.zeros_like(x)
        a = np.ones_like(x)
        xyz = np.vstack((x, y, z, a)).T

        return xyz

    def setup_eye_elements(self, eye_spec: Dict[str, float]) -> List[GeoObject]:
        """
        Set up the eye elements (retina, cornea, lens, aqueous humour, vitreous humour).

        Parameters
        ----------
        eye_spec : dict
            Dictionary of eye specifications (data from: "Optics of the Human Eye" by W. N. Charman)

        Returns
        -------
        list
            List of eye element meshes.
        """

        retina = self.hemisphere(center=[0, 0, eye_spec['d_r'] / 2.0], radius=-eye_spec['r_r'] * (1.0 - 1e-3))
        retina.set_material(mat_type="measure")
        meshes = [retina]

        cornea = self.spherical_lens_nofoc(r1=eye_spec['r_ac'], r2=eye_spec['r_pc'], x1=eye_spec['d_ac'], x2=eye_spec['d_pc'], d=eye_spec['r_cornea'])
        cornea.set_material(mat_type="refractive", index_of_refraction=eye_spec['IOR_c'])
        meshes.append(cornea)

        lens = self.spherical_lens_nofoc(r1=eye_spec['r_al'], r2=eye_spec['r_pl'], x1=eye_spec['d_al'], x2=eye_spec['d_pl'], d=eye_spec['r_lens'])
        lens.set_material(mat_type="refractive", index_of_refraction=eye_spec['IOR_l'])
        meshes.append(lens)

        aqu_humour = self.spherical_lens_nofoc(r1=eye_spec['r_pc'], r2=eye_spec['r_al'],
                                               x1=eye_spec['d_pc'] * (1.0 + 1e-6), x2=eye_spec['d_al'] * (1.0 - 1e-6),
                                               d=eye_spec['r_cornea'], d2=eye_spec['r_lens'])
        aqu_humour.set_material(mat_type="refractive", index_of_refraction=eye_spec['IOR_ah'])
        meshes.append(aqu_humour)

        vit_humour = self.spherical_lens_nofoc(r1=eye_spec['r_pl'], r2=eye_spec['r_r'] * (1.0 + 1e-6),
                                               x1=eye_spec['d_pl'] * (1.0 + 1e-6), x2=eye_spec['d_r'],
                                               d=eye_spec['r_lens'], d2=eye_spec['r_lens'], sign2_arcsin=-1.0)
        vit_humour.set_material(mat_type="refractive", index_of_refraction=eye_spec['IOR_vh'])
        meshes.append(vit_humour)

        return meshes

if __name__ == '__main__':
    print("Generating test geometry")

    # Create OpticalElements object
    oe = OpticalElements(mesh_angular_resolution=2*np.pi/50)
    draw_wireframes = False

    # Create list to store the meshes
    meshes = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])]

    # Create parabolic mirror mesh and append to meshes list
    parabolic_mirror_mesh = oe.parabolic_mirror(focus=(0, 0, 0), focal_length=1.0, diameter=2.0, reflectivity=0.98)
    meshes.append(parabolic_mirror_mesh.get_open3d_mesh(return_wireframe=draw_wireframes))

    # Create sphere mesh and append to meshes list
    sphere_mesh = oe.sphere(center=[3, 0, 0], radius=1.0)
    meshes.append(sphere_mesh.get_open3d_mesh(return_wireframe=draw_wireframes))

    # Create hemisphere mesh and append to meshes list
    hemisphere_mesh = oe.hemisphere(center=[-3, 0, 0], radius=1)
    meshes.append(hemisphere_mesh.get_open3d_mesh(return_wireframe=draw_wireframes))

    # Create spherical lens mesh and append to meshes list
    lens_mesh = oe.spherical_lens_nofoc(r1=5, r2=10, x1=0, x2=0.5, d=1)
    lens_mesh.transform(Pose.from_axis_angle(np.array([0,0,0]), np.array([-3,3,0])))
    meshes.append(lens_mesh.get_open3d_mesh(return_wireframe=draw_wireframes))

    # Create uncapped extruded mesh and append to meshes list
    mesh_extruded_uncapped = oe.extrude_by_vector(oe.curve_circle(radius=1.0), np.array([0, 1, 1]), False)
    mesh_extruded_uncapped.transform(Pose.from_axis_angle(np.array([0,0,0]), np.array([3,3,0])))
    meshes.append(mesh_extruded_uncapped.get_open3d_mesh(return_wireframe=draw_wireframes))

    # Create capped extruded mesh and append to meshes list
    mesh_extruded_capped = oe.extrude_by_vector(oe.curve_circle(radius=1.0), np.array([0, -1, 1]), True)
    mesh_extruded_capped.transform(Pose.from_axis_angle(np.array([0, 0, 0]), np.array([0, 3, 0])))
    meshes.append(mesh_extruded_capped.get_open3d_mesh(return_wireframe=draw_wireframes))

    # Create biconcave spherical lens mesh and append to meshes list
    lens_spherical_biconcave_mesh = oe.lens_spherical_biconcave(focus=(0, 0, 0), r1=2., r2=3., diameter=1.0, IOR=2.5)
    lens_spherical_biconcave_mesh.transform(Pose.from_axis_angle(np.array([0, 0, 0]), np.array([-3, 6, 0])))
    meshes.append(lens_spherical_biconcave_mesh.get_open3d_mesh(return_wireframe=draw_wireframes))

    # Create a human eye model
    # eye specifications (data from: "Optics of the Human Eye" by W. N. Charman)
    eye_spec = {
        "r_cornea": 5.0*1e-1,
        "r_lens": 5.0*1e-1,
        "r_ac": 7.8*1e-1,
        "d_ac": 0.0*1e-1,
        "r_pc": 6.5*1e-1,
        "d_pc": 0.55*1e-1,
        "r_al": 10.2*1e-1,
        "d_al": 3.6*1e-1,
        "r_pl": -6.0*1e-1,
        "d_pl": 7.6*1e-1,
        "r_r": -12.1*1e-1,
        "d_r": 24.2*1e-1,
        "IOR_c": 1.3771,
        "IOR_l": 1.4200,
        "IOR_ah": 1.3374,
        "IOR_vh": 1.336
    }

    # setup meshes (eye elements)
    eye_mesh = oe.setup_eye_elements(eye_spec)
    [e.transform(Pose.from_axis_angle(np.array([0, 0, 0]), np.array([-3, 6, 0]))) for e in eye_mesh]
    [meshes.append(e.get_open3d_mesh(return_wireframe=draw_wireframes)) for e in eye_mesh]

    # Draw all meshes
    o3d.visualization.draw_geometries(meshes)
