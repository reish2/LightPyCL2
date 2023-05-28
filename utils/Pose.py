"""
Copyright (c) 2015-2022 IVISO GmbH

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import numpy as np


class Pose:
    def __init__(self, r: np.ndarray, t: np.ndarray) -> None:
        """
        Initialize Pose object with rotation matrix and translation vector.

        :param r: Rotation matrix (3x3)
        :type r: np.ndarray
        :param t: Translation vector
        :type t: np.ndarray
        """
        assert r.shape == (3, 3), "Rotation matrix must be 3x3"
        self.r = r
        self.t = t

    @classmethod
    def from_axis_angle(cls, r: np.ndarray, t: np.ndarray) -> 'Pose':
        """
        Create Pose object from axis (as a 3D vector) and angle.

        :param r: Axis of rotation (3D vector)
        :type r: np.ndarray
        :param t: Angle of rotation
        :type t: np.ndarray
        :return: Pose object
        :rtype: Pose
        """
        assert r.shape == (3,), "Rotation axis must be a 3D vector"
        return cls(cv2.Rodrigues(r.astype(np.float64))[0].astype(np.float64), t.astype(np.float64))

    @property
    def I(self) -> 'Pose':
        """
        Inverse of the pose - when applied transforms the pose back to the origin.

        :return: Inverted pose
        :rtype: Pose
        """
        return Pose(self.r.T, - (self.r.T @ self.t))

    def __matmul__(self, other: 'Pose') -> 'Pose':
        """
        Operator overload for matrix multiplication, applies the poses one after another.

        :param other: Another pose to apply after this one
        :type other: Pose
        :return: Resulting pose after applying the two poses
        :rtype: Pose
        """
        return Pose(self.r @ other.r, self.r @ other.t + self.t)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Applies the Pose to a set of points.

        Args:
            points: Set of points represented by an Nx3 numpy array.

        Returns:
            Transformed set of points.
        """
        if points.shape[1] == 3:
            return points @ self.r.T + self.t
        if points.shape[1] == 4:
            T = np.eye(4)
            T[:3,:3] = self.r
            T[:3,3] = self.t
            return points @ T.T
        else:
            print("Error: input array shape is not Nx3 or Nx4")
            raise ValueError