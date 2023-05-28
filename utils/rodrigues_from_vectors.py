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

from utils.Pose import Pose


def calculate_rotation_vector(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Computes a rotation vector that rotates one vector into another.

    :param vec1: Initial vector
    :param vec2: Vector to rotate into
    :return: Computed rotation vector
    """

    # Normalize the input vectors
    vec1 = vec1.astype(np.float64) / np.linalg.norm(vec1.astype(np.float64))
    vec2 = vec2.astype(np.float64) / np.linalg.norm(vec2.astype(np.float64))

    # Handle the case when the vectors are in the same or opposite direction
    if np.allclose(vec1, vec2):
        # No rotation needed
        return np.array([0.0, 0.0, 0.0])

    elif np.allclose(vec1, -vec2):
        # 180 degree rotation around an orthogonal vector
        orthogonal_vec = np.array([-vec1[1], vec1[0], 0])
        if np.linalg.norm(orthogonal_vec) == 0:
            orthogonal_vec = np.array([0, -vec1[2], vec1[1]])
        orthogonal_vec /= np.linalg.norm(orthogonal_vec)
        return np.pi * orthogonal_vec

    # Calculate the cross and dot products
    cross_product = np.cross(vec1, vec2)
    dot_product = np.dot(vec1, vec2)

    # Calculate the angle
    theta = np.arccos(dot_product)

    # Calculate the rotation axis
    k = cross_product / np.linalg.norm(cross_product)

    # Calculate the rotation vector
    return theta * k


def main() -> None:
    """
    Test function to verify rotation_vector_rod
    """

    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    r = calculate_rotation_vector(vec1, vec2)
    P = Pose.from_axis_angle(r, np.array([0, 0, 0]))
    print(P.r @ vec1 - vec2)

    vec1 = np.array([1, 0, 0])
    vec2 = np.array([1, 0, 0])
    r = calculate_rotation_vector(vec1, vec2)
    P = Pose.from_axis_angle(r, np.array([0, 0, 0]))
    print(P.r @ vec1 - vec2)

    vec1 = np.random.rand(3)
    vec2 = -vec1
    r = calculate_rotation_vector(vec1, vec2)
    P = Pose.from_axis_angle(r, np.array([0, 0, 0]))
    print(P.r @ vec1 - vec2)


if __name__ == "__main__":
    main()
