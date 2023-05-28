// Copyright (C) 2023 Reish2
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//Notes:
// - use one kernel to calculate intersections and generate new rays
// - call from python for each iteration => geometry remains in GPU memory, rays are trimmed
//   => worst case rays need to be partitioned when they become too many.
//	=> estimation for 9 bounces of 10k rays = 1024 * 10krays = 10Mrays => currently ~1GB VRAM required
//   => iteration count remains low (few read/write cycles)
// - kernel for postprocessing (calculate angles for each measured ray)
//	or binning kernel (sorting/summing)

/**
mesh_count 		= 4
ray_count		= 1e4
iterations		= 9 => 1024 ray_multiplyer

rays_origin 		= 4xfloat32 = 16
rays_dir 		= 16
rays_dest		= 16
ray_entering		= 4
ray_isect_mesh_id	= 4
ray_isect_mesh_idx	= 4
isect_min_ray_len	= 4 * mesh_count = 16
isects_count		= 4 * mesh_count = 16
ray_isect_mesh_idx_tmp	= 4 * mesh_count = 16
TOTAL			= 108B
=> estimated memory usage for worst case of 1e4 rays and 9 iterations = 1.03 GB <= FOR RAYS AND BUFFERS ONLY

**/

/**
 * Function to intersect ray with triangle.
 * Algorithm based on: Tomas Möller and Ben Trumbore, "Fast, Minimum Storage Ray-Triangle Intersection", Journal of Graphics Tools, vol. 2, no. 1, pp. 21-28, 1997.
 *
 * @param O: Origin of the ray.
 * @param D: Direction of the ray.
 * @param V0: First vertex of the triangle.
 * @param V1: Second vertex of the triangle.
 * @param V2: Third vertex of the triangle.
 * @param t: Output parameter to get the intersection point along the ray.
 * @param u: Output parameter to get the barycentric coordinate.
 * @param v: Output parameter to get the barycentric coordinate.
 *
 * @return 1 if there is an intersection, otherwise 0.
 *
 * The Ray is defined by R(t) = O + t*D
 * The Triangle is defined by T(u,v) = (1-u-v)*V0 + u*V1 + v*V2
 * where u >= 0, v >= 0, and u+v <= 1
 *
 * The intersection of the Ray and Triangle is found by solving R(t) = T(u,v) for t, u, v.
 */
int intersect_triangle(float3 O, float3 D, float3 V0, float3 V1, float3 V2, float *t, float *u, float *v)
{
    float3 E1, E2, T, P, Q;
    float DEN, iDEN;
    const float EPSILON_NUM = 0.000001;

    // Calculate triangle edge vectors. They define the triangle plane.
    E1 = V1 - V0;
    E2 = V2 - V0;

    P = cross(D, E2);
    DEN  = dot(P, E1);

    // If DEN = 0, the ray is parallel to the triangle, no intersection.
    if (DEN > -EPSILON_NUM && DEN < EPSILON_NUM)
        return 0;

    iDEN = 1.0 / DEN;
    T = O - V0;

    // Check barycentric coordinates.
    *u = dot(P, T) * iDEN;
    if (*u < 0.0 || *u > 1.0)
        return 0;

    Q = cross(T, E1);

    // Check barycentric coordinates.
    *v = dot(Q, D) * iDEN;
    if (*v < 0.0 || *u + *v > 1.0)
        return 0;

    // All conditions for u and v are met? Now we can calculate t.
    *t = dot(Q, E2) * iDEN;

    return 1;
}


/**
 * Postprocessing step for intersection results.
 * The intersection routine collects the closest intersection for every mesh.
 * This function answers the questions: Which intersection is the closest? Is the ray entering or leaving it?
 * Where is the ray's destination?
 *
 * @param rays_origin: Origin of the rays.
 * @param rays_dir: Direction of the rays.
 * @param rays_dest: Destination of the rays.
 * @param rays_prev_isect_mesh_id: Previous intersection mesh ID of the rays.
 * @param rays_n1_mesh_id: Refractive index n1 of the rays.
 * @param rays_n2_mesh_id: Refractive index n2 of the rays.
 * @param ray_entering: Ray entering status.
 * @param ray_isect_mesh_id: Intersection mesh ID of the rays.
 * @param ray_isect_mesh_idx: Intersection mesh index of the rays.
 * @param mesh_v0: First vertex of the mesh.
 * @param mesh_v1: Second vertex of the mesh.
 * @param mesh_v2: Third vertex of the mesh.
 * @param mesh_id: Mesh ID.
 * @param mesh_mat_type: Mesh material type.
 * @param isect_min_ray_len: Minimum intersection ray length.
 * @param isects_count: Count of intersections.
 * @param ray_isect_mesh_idx_tmp: Temporary intersection mesh index of the rays.
 * @param mesh_count: Total mesh count.
 * @param ray_count: Total ray count.
 * @param max_ray_len: Maximum ray length.
 *
 * The function performs the following steps:
 * 1. It calculates rays_dest and ray_entering.
 * 2. Iterates over all meshes (not triangles) to find the closest intersected mesh.
 * 3. Determines whether the ray is entering or exiting the mesh.
 * 4. Figures out which mesh id the ray is going to propagate in next.
 */
__kernel void intersect_postproc(__global const float3 *rays_origin, __global const float3 *rays_dir, __global float3 *rays_dest,
		__global int *rays_prev_isect_mesh_id, __global int *rays_n1_mesh_id, __global int *rays_n2_mesh_id,
		__global int *ray_entering, __global int *ray_isect_mesh_id, __global int *ray_isect_mesh_idx, __global const float3 *mesh_v0,
		__global const float3 *mesh_v1, __global const float3 *mesh_v2, __global const int *mesh_id, __global const int *mesh_mat_type, __global float *isect_min_ray_len,
		__global int *isects_count, __global int *ray_isect_mesh_idx_tmp, int mesh_count, int ray_count, float max_ray_len)
{
    // Initiate postprocessing to calculate rays_dest and ray_entering.
    // EPSILON serves as the smallest possible length scale for error margins, optimized for float32's ~7 digits precision.
    // The value of EPSILON is adjusted according to the scene's current length scale, here determined by max_ray_len.
    const float EPSILON 	= 0.000001*max_ray_len;

    // Obtain the ray index for parallel postprocessing.
    int rid = get_global_id(0);

    // Create a buffer for isect_min_ray_len, initialized with maximum ray length.
    float t_tmp = max_ray_len;

    // Initialize id and index of the closest intersection as -1, signaling no intersection found yet.
    int isect_mesh_id_tmp = -1;
    int isect_mesh_idx_tmp = -1;

    // Initialize an intersection distance buffer with the maximum ray length.
    float t_min = max_ray_len;

    // n1_id and n2_id represent the refractive indices n1 and n2. Initialization as -1 denotes that ray is in the environment (IOR_env).
    int n1_id = -1;
    int n2_id = -1;

    // Get the previous intersect mesh id.
    // A value of -2 implies ray has not yet interacted with a material.
    // A value of -1 suggests ray is located outside any mesh.
    // A non-negative value indicates ray is within a mesh of corresponding id.
    int prev_isect = rays_prev_isect_mesh_id[rid];

    // Traverse all meshes to find the one with the closest intersection.
    for(int j=0; j<mesh_count; j++) {
        // Update t_tmp with the minimum intersection ray length for the given mesh.
        t_tmp=isect_min_ray_len[mesh_count*rid+j];

        // If a closer intersection is found, update the buffers for minimum intersection distance, mesh id, and mesh index.
        if(t_tmp<t_min) {
            t_min = t_tmp;
            isect_mesh_id_tmp = j;
            isect_mesh_idx_tmp = ray_isect_mesh_idx_tmp[mesh_count*rid+j];
        }
    }

    // Check if there's at least one intersection (denoted by at least one non-negative mesh id).
    // If so, we need to determine whether the ray is entering or exiting the mesh.
    // If the count of intersections with a closed (solid) mesh is even, it signifies that the ray is entering the mesh.
    // Conversely, an odd count of intersections suggests the ray is exiting the mesh.
    int entering = 0;
    if(isect_mesh_id_tmp >= 0) {
		// Compute the entering status by using the remainder of the intersection count.
        // If this remainder is even, the ray is entering the mesh; if odd, it's exiting.
        entering = 1 - (isects_count[mesh_count * rid + isect_mesh_id_tmp] % 2);
        ray_entering[rid] = entering;

        // If the previous intersection is -2, the ray has just been emitted from the source.
        if(prev_isect == -2) {
            if(entering == 1) {
                // The ray has started in a vacuum.
                n1_id = -1;
                n2_id = isect_mesh_id_tmp;
            } else {
                // In this case, n1_id is known but n2_id still needs to be determined later.
                n1_id = isect_mesh_id_tmp;
                n2_id = -1;
            }
        } else if(prev_isect == -1) {
            // If the previous intersection is -1, the ray is about to enter a new mesh.
            if(entering == 1) {
                // If the ray is entering, n1_id should reflect the material of the previous traversal, while n2_id represents the current mesh.
                n1_id = -1;
                n2_id = isect_mesh_id_tmp;
            } else {
                // If the ray is exiting, it's impossible for refractive media.
                // However, for completeness, we still assign material IDs.
                n1_id = isect_mesh_id_tmp;
                n2_id = -1;
            }
        } else {
            // If the previous intersection was with a material (not -1 or -2), then the ray was traveling in a material.
            // If the ray is entering a new mesh, assign the material of the previous intersection to n1_id, and the new mesh to n2_id.
            if(entering == 1) {
                n1_id = prev_isect;
                n2_id = isect_mesh_id_tmp;
            } else {
                // If the ray is exiting the mesh, assign the mesh to n1_id. The next intersecting mesh (n2_id) will be determined later.
                n1_id = isect_mesh_id_tmp;
                n2_id = -1;
            }
        }
        // Next, determine the mesh ID that the ray will propagate through next. This only needs to be calculated if an intersection has been found.
        // This task involves differentiating between two situations:
        // 1. Intersections occurring within the interval [t_min, t_min+EPSILON]. In this scenario, only the final intersection is important.
        // 	Note: As LightPyCL currently doesn't consider interference effects, only the final intersect's entering/exiting status matters.
        // 	If the final intersection is entering, the n2_id mesh is determined.
        // 	If it's exiting, the next intersecting mesh outside of the defined interval is determined in case 2.
        // 2. Identify the closest exiting intersection within the interval [t_min+EPSILON, max_ray_len]. This mesh will define n2_id.
        // 	If no such mesh exists, n2_id is assigned the refractive index of the environment.
        // NOTE: This process only works if we disregard all materials but refractive/dissipative. Therefore, mesh_mat_type == 0 must be true.

        // Here, t_minmin and t_maxmin are the minimum and maximum ray lengths in the interval [t_minmin, t_minmin+EPSILON] respectively.
        // t_minmax is the ray length of the closest exiting intersection in the interval [t_min+EPSILON, max_ray_len].
        float t_minmin = t_min;
        float t_maxmin = t_min;
        float t_minmax = max_ray_len;

        // These variables will hold the entering status and indices of the meshes for the intersections within the intervals defined above.
        int   maxmin_entering = 0;
        int   maxmin_idx = -1;
        int   minmax_idx = -1;

                // Loop over all meshes to evaluate intersections and track those involving refractive/dissipative media.
        for(int j = 0; j < mesh_count; j++) {
            // Only consider intersections with refractive (type 0) and dissipative (type 4) media.
            if(mesh_mat_type[j] == 0 || mesh_mat_type[j] == 4) {
                // Retrieve the minimum intersection ray length for the given mesh.
                t_tmp = isect_min_ray_len[mesh_count*rid+j];

                // Determine the entering status for the j-th intersection.
                entering = 1 - (isects_count[mesh_count*rid+j] % 2);

                // If the intersection is within the interval [t_minmin, t_minmin+EPSILON] and greater than the current maximum,
                // update the maximum ray length (t_maxmin) in this interval, along with the corresponding mesh index (maxmin_idx)
                // and its entering status (maxmin_entering).
                if(t_tmp <= t_minmin + EPSILON && t_tmp >= t_maxmin) {
                    t_maxmin = t_tmp;
                    maxmin_idx = j;
                    maxmin_entering = entering;
                }

                // Identify the smallest exiting intersection within the interval [t_min+EPSILON, max_ray_len].
                // If an intersection is within this interval, is exiting (entering == 0), and is less than the current minimum,
                // update the minimum ray length (t_minmax) in this interval and the corresponding mesh index (minmax_idx).
                if(t_tmp > t_minmin + EPSILON && entering == 0 && t_tmp <= t_minmax) {
                    t_minmax = t_tmp;
                    minmax_idx = j;
                }
            }
        }

        // After evaluating intersections within both intervals, we can determine the next material (n2) based on different scenarios.

        // Case 1: The ray is entering an adjacent mesh.
        if(maxmin_entering == 1) {
            // Update minimum intersection distance (t_min) and the ID of the next material (n2_id) to the intersected mesh index.
            t_min = t_maxmin;
            n2_id = maxmin_idx;
        } else { // Case 2: The ray is exiting one or many adjacent meshes.
            // If there is an intersection with a nearby mesh (in the first interval), we need to update the minimum intersection distance (t_min).
            if(maxmin_idx >= 0) {
                t_min = t_maxmin;
            }
            // If there are no intersections within the first interval, the previously set values are still valid.

            // Determine the next material the ray will traverse.
            // Case 2.1: There's a mesh within the ray's path that hasn't been exited yet. This mesh will be the next material.
            if(minmax_idx >= 0) {
                n2_id = minmax_idx;
            }
            // Case 2.2: The ray isn't exiting any materials within its path. In this case, the ray will be entering the environment,
            // thus leaving n2_id at its initial value of -1.
        }
    }

    // Store the final results into the global memory. This includes the IDs of the current and next materials the ray is in (n1_id and n2_id),
    // the new destination point of the ray, and the IDs of the mesh and the triangle where the ray intersects.

    // Set the ID of the current material the ray is in.
    rays_n1_mesh_id[rid] = n1_id;

    // Set the ID of the next material the ray will enter.
    rays_n2_mesh_id[rid] = n2_id;

    // Calculate and set the new destination point of the ray, which is the ray's origin plus the direction vector scaled by the minimum intersection distance.
    rays_dest[rid] = rays_origin[rid] + rays_dir[rid] * t_min;

    // Set the ID of the mesh where the ray intersects.
    ray_isect_mesh_id[rid] = isect_mesh_id_tmp;

    // Set the index of the intersected triangle within the mesh where the ray intersects.
    ray_isect_mesh_idx[rid] = isect_mesh_idx_tmp;
}

/**
 * @brief Intersect rays with triangles in parallel.
 *
 * This function takes a set of rays and checks for intersections with a set of triangles. Each ray will be iterated
 * over all triangles, and any found intersections will be recorded and updated in the relevant buffers. Given the
 * randomness of ray's origin and direction, the intersection process is performed with a brute force approach.
 *
 * @param rays_origin Array containing the origins of all rays.
 * @param rays_dir Array containing the directions of all rays.
 * @param rays_dest Array containing the destination points of all rays.
 * @param ray_entering Array indicating if each ray is entering or exiting a medium.
 * @param ray_isect_mesh_id Array storing the ID of the mesh that each ray intersects with.
 * @param ray_isect_mesh_idx Array storing the index of the mesh that each ray intersects with.
 * @param mesh_v0 Array of the first vertex of each triangle in the mesh.
 * @param mesh_v1 Array of the second vertex of each triangle in the mesh.
 * @param mesh_v2 Array of the third vertex of each triangle in the mesh.
 * @param mesh_id Array of mesh IDs for each triangle.
 * @param isect_min_ray_len Array storing the minimum length of the ray before it intersects with a triangle.
 * @param isects_count Array storing the count of intersections for each mesh ID.
 * @param ray_isect_mesh_idx_tmp Temporary array storing the indices of the intersected mesh.
 * @param mesh_count Total count of meshes.
 * @param tri_count Total count of triangles.
 * @param ray_count Total count of rays.
 * @param max_ray_len The maximum possible length of a ray.
 *
 * @return void
 */
__kernel void intersect(__global const float3 *rays_origin, __global const float3 *rays_dir, __global float3 *rays_dest,
                        __global int *ray_entering, __global int *ray_isect_mesh_id, __global int *ray_isect_mesh_idx,
                        __global const float3 *mesh_v0, __global const float3 *mesh_v1, __global const float3 *mesh_v2,
                        __global const int *mesh_id, __global float *isect_min_ray_len, __global int *isects_count,
                        __global int *ray_isect_mesh_idx_tmp, int mesh_count, int tri_count, int ray_count, float max_ray_len)
{
    // Define an EPSILON value to manage precision errors. It depends on the current scale of the scene, represented by max_ray_len.
    const float EPSILON = 0.000001*max_ray_len;

    // Fetch the index of the ray, its origin and direction.
    int rid = get_global_id(0);
    float3 ray_origin = rays_origin[rid];
    float3 ray_dir = rays_dir[rid];

    // Initialize intersection variables for each ray.
    int isects_count_tmp = 0;
    float isect_min_ray_len_tmp = max_ray_len;
    int ray_isect_mesh_idx_ltmp = -1;

    // Placeholder variables for triangle intersection.
    float t, u, v;
    int isect = 0;
    int idx_tmp = 0;

    // Iterate over each triangle to check for intersections.
    for(int i = 0; i < tri_count; i++) {
        idx_tmp = mesh_count*rid + mesh_id[i];

        // Upon moving to a new mesh, update intersection counters in global memory and reset temporary buffers.
        if(i>0 && mesh_id[i-1] != mesh_id[i]) {
            isects_count[idx_tmp-1] = isects_count_tmp;
            isect_min_ray_len[idx_tmp-1] = isect_min_ray_len_tmp;
            ray_isect_mesh_idx_tmp[idx_tmp-1] = ray_isect_mesh_idx_ltmp;
            isects_count_tmp = 0;
            isect_min_ray_len_tmp = max_ray_len;
            ray_isect_mesh_idx_ltmp = -1;
        }

        // Perform intersection between ray and triangle.
        isect = intersect_triangle(ray_origin, ray_dir, mesh_v0[i], mesh_v1[i], mesh_v2[i], &t, &u, &v);

        // If intersection is found and it is closer to ray origin than the previous intersections, update intersection distance and counters.
        if(isect && t > EPSILON){
            if(t < isect_min_ray_len_tmp) {
                isect_min_ray_len_tmp = t;
                ray_isect_mesh_idx_ltmp = i;
            }
            isects_count_tmp += 1;
        }
    }

    // Ensure results for the last mesh are stored in global memory.
    isects_count[idx_tmp] = isects_count_tmp;
    isect_min_ray_len[idx_tmp] = isect_min_ray_len_tmp;
    ray_isect_mesh_idx_tmp[idx_tmp] = ray_isect_mesh_idx_ltmp;
}

/**
 * Function to reflect and refract incoming rays according to Snell's law.
 *
 * @param in_ray_dest The destination of the incoming ray
 * @param in_ray_dir The direction of the incoming ray
 * @param in_ray_power The power of the incoming ray
 *
 * @param ray_reflect_origin To be filled with the origin of the reflected ray
 * @param ray_reflect_dir To be filled with the direction of the reflected ray
 * @param ray_reflect_power To be filled with the power of the reflected ray
 * @param ray_reflect_measured To be filled with the measurement flag of the reflected ray
 *
 * @param ray_refract_origin To be filled with the origin of the refracted ray
 * @param ray_refract_dir To be filled with the direction of the refracted ray
 * @param ray_refract_power To be filled with the power of the refracted ray
 * @param ray_refract_measured To be filled with the measurement flag of the refracted ray
 *
 * @param surf_normal_in The incoming surface normal
 * @param n1 The refractive index of the first medium
 * @param n2 The refractive index of the second medium
 *
 * @return 0 if the ray was reflected or refracted successfully, 1 if an error occurred
 */
int reflect_refract(float3 in_ray_dest, float3 in_ray_dir, float in_ray_power,
                    float3 *ray_reflect_origin, float3 *ray_reflect_dir, float *ray_reflect_power, int *ray_reflect_measured,
                    float3 *ray_refract_origin, float3 *ray_refract_dir, float *ray_refract_power, int *ray_refract_measured,
                    float3 surf_normal_in, float n1, float n2) {
    // Initializing variables and computing the ratio of refractive indices
    float3 surf_normal = surf_normal_in;
    float r = n1/n2;
    float TIR_check;

    // Adjust the surface normal vector to always point towards the incoming ray
    float cosT1 = -dot(surf_normal,in_ray_dir);
    if(cosT1 < 0.0){
        surf_normal = -surf_normal_in;
        cosT1 = -dot(surf_normal,in_ray_dir);
    }

    // Check for Total Internal Reflection (TIR)
    TIR_check = 1.0f - pown(r,2) * (1.0f - pown(cosT1,2));

    // Normal refraction: No TIR occurs
    if (TIR_check >= 0.0f) {
        float cosT2 = sqrt(TIR_check);
        float Rs = pown(fabs((n1*cosT1 - n2*cosT2)/(n1*cosT1 + n2*cosT2)),2); // s polarized
        float Rp = pown(fabs((n1*cosT2 - n2*cosT1)/(n1*cosT2 + n2*cosT1)),2); // p polarized
        float reflect_power = in_ray_power * (Rs+Rp)/2.0;

        // Update properties of reflected and refracted rays
        *ray_reflect_dir = in_ray_dir + 2.0f * cosT1 * surf_normal;
        *ray_reflect_origin = in_ray_dest;
        *ray_reflect_power = reflect_power;
        *ray_reflect_measured = 0;

        *ray_refract_dir = in_ray_dir * r + (r * cosT1 - cosT2) * surf_normal;
        *ray_refract_origin = in_ray_dest;
        *ray_refract_power = in_ray_power - reflect_power;
        *ray_refract_measured = 0;

        return 0;
    }

    // Total Internal Reflection occurs: Propagate all power to reflection
    if(TIR_check < 0.0f) {
        *ray_reflect_dir = in_ray_dir + 2.0f * cosT1 * surf_normal;
        *ray_reflect_origin = in_ray_dest;
        *ray_reflect_power = in_ray_power;
        *ray_reflect_measured = 0;

        *ray_refract_dir = (float3)(0,0,0);
        *ray_refract_origin = in_ray_dest;
        *ray_refract_power = 0.0f;
        *ray_refract_measured = -1;

        return 0;
    }

    // Return error code if the code execution reaches this point
    return 1;
}


/**
 * OpenCL Kernel function that reflects and refracts rays, checks if the ray should be terminated or measured, and sets result buffers accordingly.
 *
 * @param in_rays_origin Global input buffer for the origins of incoming rays.
 * @param in_rays_dest Global input buffer for the destinations of incoming rays.
 * @param in_rays_dir Global input buffer for the directions of incoming rays.
 * @param in_rays_power Global input buffer for the power of incoming rays.
 * @param in_rays_measured Global input buffer indicating whether each incoming ray has been measured.
 * @param in_ray_entering Global input buffer specifying if each ray is entering the scene.
 * @param rays_n1_mesh_id Global buffer storing mesh IDs indicating the first medium for refraction for each ray.
 * @param rays_n2_mesh_id Global buffer storing mesh IDs indicating the second medium for refraction for each ray.
 * @param rays_reflect_origin Global output buffer to be filled with the origin of the reflected rays.
 * @param rays_reflect_dir Global output buffer to be filled with the direction of the reflected rays.
 * @param rays_reflect_power Global output buffer to be filled with the power of the reflected rays.
 * @param rays_reflect_measured Global output buffer to be filled with measurement flags for the reflected rays.
 * @param rays_refract_origin Global output buffer to be filled with the origin of the refracted rays.
 * @param rays_refract_dir Global output buffer to be filled with the direction of the refracted rays.
 * @param rays_refract_power Global output buffer to be filled with the power of the refracted rays.
 * @param rays_refract_measured Global output buffer to be filled with measurement flags for the refracted rays.
 * @param ray_isect_mesh_id Global output buffer to be filled with the mesh IDs of intersection points for the rays.
 * @param ray_isect_mesh_idx Global output buffer to be filled with the mesh indices of intersection points for the rays.
 * @param mesh_v0 Global input buffer containing the first vertices of meshes.
 * @param mesh_v1 Global input buffer containing the second vertices of meshes.
 * @param mesh_v2 Global input buffer containing the third vertices of meshes.
 * @param mesh_id Global input buffer containing the IDs of meshes.
 * @param mesh_mat_type Global input buffer containing the material types of meshes.
 * @param mesh_ior Global input buffer containing the Index of Refraction (IOR) of meshes.
 * @param mesh_refl Global input buffer containing the reflectivity of meshes.
 * @param mesh_diss Global input buffer containing the dissipation of meshes.
 * @param IOR_env Index of Refraction of the environment.
 * @param mesh_count The total number of meshes.
 * @param ray_count The total number of rays.
 * @param max_ray_len The maximum length that a ray can have.
 */
__kernel void reflect_refract_rays(__global const float3 *in_rays_origin, __global const float3 *in_rays_dest, __global const float3 *in_rays_dir,
                                   __global float *in_rays_power, __global int *in_rays_measured, __global const int *in_ray_entering,
                                   __global int *rays_n1_mesh_id, __global int *rays_n2_mesh_id,
                                   __global float3 *rays_reflect_origin, __global float3 *rays_reflect_dir,
                                   __global float *rays_reflect_power, __global int *rays_reflect_measured,
                                   __global float3 *rays_refract_origin, __global float3 *rays_refract_dir,
                                   __global float *rays_refract_power, __global int *rays_refract_measured,
                                   __global int *ray_isect_mesh_id, __global int *ray_isect_mesh_idx,
                                   __global const float3 *mesh_v0, __global const float3 *mesh_v1, __global const float3 *mesh_v2,
                                   __global const int *mesh_id, __global const int *mesh_mat_type, __global const float *mesh_ior,
                                   __global const float *mesh_refl, __global const float *mesh_diss, float IOR_env, int mesh_count, int ray_count, float max_ray_len)
{
	// EPSILON for length comparisons is dependent on the current length scale of the scene,
    // given that float32 precision is approximately 7 digits.
    // max_ray_len is a suitable measure for this context.
    const float EPSILON = 0.000001f * max_ray_len;

    // EPSILON_NUM for non-length comparisons can be smaller, as values close to 0 are relevant.
    const float EPSILON_NUM = 0.000001f;

    // Get the global ID for this instance of the kernel.
    int rid = get_global_id(0);

    // Get the mesh ID of the intersected object for the current ray.
    int rmid = ray_isect_mesh_id[rid];

    // Set default values to terminate ray, assuming no intersection has occurred.
    int mesh_mat = 2; // Type of mesh material: 0 refractive, 1 mirror, 2 terminate, 3 measure, 4 anisotropic refractive
    float R_mesh = 0.0f; // Mesh reflectivity

    // If a ray intersection has occurred, fetch material parameters from the intersected mesh.
    if(rmid >= 0) {
        mesh_mat = mesh_mat_type[rmid];
        R_mesh = mesh_refl[rmid];
    }

    // Calculate dissipation and determine the Index of Refraction (IOR) of the material from which the ray is originating.
    int r_orig_mid = rays_n1_mesh_id[rid]; // n1 mesh ID (origin material)
    int r_dest_mid = rays_n2_mesh_id[rid]; // n2 mesh ID (destination material)

    // Assume IOR of environment (typically air) if not otherwise specified.
    float IOR_in_ray = IOR_env; // IOR of incoming ray's medium
    float IOR_n2_ray = IOR_env; // IOR of destination medium

    float in_ray_pow = in_rays_power[rid]; // Power of incoming ray
    float3 in_ray_dest = in_rays_dest[rid]; // Destination of incoming ray

    // If the ray originates from a material (not air), fetch the IOR and apply dissipation if necessary.
    if(r_orig_mid >= 0) {
        IOR_in_ray = mesh_ior[r_orig_mid];

        // If the originating material is dissipative, adjust the power of the incoming ray.
        if(mesh_mat_type[r_orig_mid] == 0 && mesh_diss[r_orig_mid] > EPSILON_NUM) {
            float ray_len = length(in_ray_dest - in_rays_origin[rid]);
            in_ray_pow = in_ray_pow * exp(-mesh_diss[r_orig_mid] * ray_len);
            in_rays_power[rid] = in_ray_pow;
        }
    } else {
        // If the ray does not originate from a material, assume the IOR of the environment.
        IOR_in_ray = IOR_env;
    }

    // If the ray's destination is a material (not air), fetch the corresponding IOR.
    if(r_dest_mid >= 0) {
        IOR_n2_ray = mesh_ior[r_dest_mid];
    }


	// If ray has not been terminated by a measurement or termination surface,
    // and intersects with a refractive or mirror material, then generate reflect and refract beams.
    int irm = in_rays_measured[rid];
    if(irm == 0 && rmid >= 0 && (mesh_mat == 0 || mesh_mat == 1)) {
        // Get the vertex positions of the intersected triangle
        int m_idx = ray_isect_mesh_idx[rid];
        float3 v0 = mesh_v0[m_idx];
        float3 v1 = mesh_v1[m_idx];
        float3 v2 = mesh_v2[m_idx];

        // Calculate normalized surface normal
        float3 surf_normal = normalize(cross(v1 - v0, v2 - v1));

        // Declare variables to store results of reflection and refraction
        float3 r_reflect_origin, r_reflect_dir, r_refract_origin, r_refract_dir;
        float r_reflect_power, r_refract_power;
        int r_reflect_measured, r_refract_measured;

        // Perform reflection and refraction
        reflect_refract(in_ray_dest, in_rays_dir[rid], in_ray_pow,
                        &r_reflect_origin, &r_reflect_dir, &r_reflect_power, &r_reflect_measured,
                        &r_refract_origin, &r_refract_dir, &r_refract_power, &r_refract_measured,
                        surf_normal, IOR_in_ray, IOR_n2_ray);

        // If the material is not a mirror, assign both reflection and refraction results
        if(mesh_mat == 0) {
            rays_reflect_origin[rid] = r_reflect_origin;
            rays_reflect_dir[rid] = r_reflect_dir;
            rays_reflect_power[rid] = r_reflect_power;
            rays_reflect_measured[rid] = r_reflect_measured;

            rays_refract_origin[rid] = r_refract_origin;
            rays_refract_dir[rid] = r_refract_dir;
            rays_refract_power[rid] = r_refract_power;
            rays_refract_measured[rid] = r_refract_measured;
        } else { // If the material is a mirror, assign reflection result and nullify refraction result
            rays_reflect_origin[rid] = r_reflect_origin;
            rays_reflect_dir[rid] = r_reflect_dir;
            rays_reflect_power[rid] = in_ray_pow * R_mesh; // Incorporate mirror losses
            rays_reflect_measured[rid] = r_reflect_measured;

            rays_refract_origin[rid] = r_refract_origin;
            rays_refract_dir[rid] = (float3)(0,0,0);
            rays_refract_power[rid] = 0.0f;
            rays_refract_measured[rid] = -1;
        }
    } else { // If the rays are measured, terminated or go nowhere, terminate reflection and refraction without calculation
        rays_reflect_origin[rid] = in_rays_dest[rid];
        rays_reflect_dir[rid] = (float3)(0,0,0);
        rays_reflect_power[rid] = 0.0f;
        rays_reflect_measured[rid] = -1;

        rays_refract_origin[rid] = in_rays_dest[rid];
        rays_refract_dir[rid] = (float3)(0,0,0);
        rays_refract_power[rid] = 0.0f;
        rays_refract_measured[rid] = -1;

        // If the beam hit termination surface or goes nowhere, mark as measured and terminated
        if(mesh_mat == 2 || rmid < 0) {
            in_rays_measured[rid] = -1;
        }
        // If the beam hit measurement surface, mark as measured
        if(mesh_mat == 3 && rmid >=0 ) {
            in_rays_measured[rid] = 1;
        }
    }

}

/**
 * @brief Rotate an array of vectors around a given pivot using a rotation matrix.
 *
 * This function applies a rotation matrix to each vector in the input array, subtracting
 * a pivot vector before the rotation and adding it back afterwards. The results are written
 * to an output array.
 *
 * @param vecs The input array of vectors to be rotated.
 * @param rot_mtx The rotation matrix to apply to each vector.
 * @param pivot The pivot vector to subtract before rotation and add back afterwards.
 * @param vecs_rot The output array where the rotated vectors are stored.
 */
__kernel void rotate_vectors(
    __global const float3 *vecs,    // input array of vectors
    __global const float3 *rot_mtx, // rotation matrix
    __global float3 *pivot,         // pivot vector
    __global float3 *vecs_rot       // output array for rotated vectors
) {
    // Get global ID for current work item
    int gid = get_global_id(0);

    // Get pivot vector (we assume it's the same for all vectors)
    float3 piv = pivot[0];

    // Subtract pivot from current vector before rotation
    float3 vec = vecs[gid] - piv;

    // Apply rotation matrix and add pivot back to the vector
    // The rotated vector is saved to the output array
    vecs_rot[gid] = (float3)(
        dot(vec, rot_mtx[0]),
        dot(vec, rot_mtx[1]),
        dot(vec, rot_mtx[2])
    ) + piv;
}



/**
 * @brief Apply a stereographic projection to an array of vectors located on a hemisphere surface.
 *
 * This function takes an array of vectors and performs a stereographic projection on them. Each vector
 * is first translated by subtracting a pivot point. Then, the vectors are rotated using a rotation matrix.
 * Finally, the stereographic projection is performed and the resulting x and y coordinates, as well as
 * corrected powers, are saved to separate arrays.
 *
 * @param vecs The input array of vectors to be projected.
 * @param pwrs The input array of power values associated with each vector.
 * @param rot_mtx The rotation matrix to be applied to each vector.
 * @param pivot The pivot point to be subtracted from each vector.
 * @param x The output array where the x-coordinates of the projected vectors are stored.
 * @param y The output array where the y-coordinates of the projected vectors are stored.
 * @param pwrs_cor The output array where the corrected power values are stored.
 */
__kernel void stereographic_projection(
    __global const float3 *vecs,     // input array of vectors
    __global const float *pwrs,      // input array of power values
    __global const float3 *rot_mtx,  // rotation matrix
    __global float3 *pivot,          // pivot point
    __global float *x,               // output array for x-coordinates
    __global float *y,               // output array for y-coordinates
    __global float *pwrs_cor         // output array for corrected powers
) {
    // Get global ID for current work item
    int gid = get_global_id(0);

    // Get pivot vector (we assume it's the same for all vectors)
    float3 piv = pivot[0];

    // Subtract pivot from current vector
    float3 vec = vecs[gid] - piv;

    // Get associated power for current vector
    float  pwr = pwrs[gid];

    // Apply rotation matrix to the vector and add pivot back
    float u = dot(vec, rot_mtx[0]) + piv.x;
    float v = dot(vec, rot_mtx[1]) + piv.y;
    float w = dot(vec, rot_mtx[2]) + piv.z;

    // Calculate the magnitude of the rotated vector
    float l = sqrt(pown(u,2)+pown(v,2)+pown(w,2));

    // Perform stereographic projection
    float xt = u / (l + w);
    float yt = v / (l + w);

    // Calculate the correction factor for power
    float A  = 4.0 / pown((1.0 + pown(xt,2) + pown(yt,2)), 2);

    // Store results to output arrays
    x[gid] = xt;
    y[gid] = yt;
    pwrs_cor[gid] = pwr / A;
}


/**
 * @brief Apply an azimuth/elevation mapping to vectors on a sphere's surface, projecting them to a circle.
 *
 * This function takes an array of vectors and performs an azimuth/elevation mapping on them. Each vector
 * is first translated by subtracting a pivot point. Then, the vectors are rotated using a rotation matrix.
 * Afterward, the mapping is applied and the resulting x and y coordinates, as well as corrected powers,
 * are saved to separate arrays.
 *
 * @param vecs The input array of vectors to be mapped.
 * @param pwrs The input array of power values associated with each vector.
 * @param rot_mtx The rotation matrix to be applied to each vector.
 * @param pivot The pivot point to be subtracted from each vector.
 * @param x The output array where the x-coordinates of the mapped vectors are stored.
 * @param y The output array where the y-coordinates of the mapped vectors are stored.
 * @param pwrs_cor The output array where the corrected power values are stored.
 */
__kernel void angular_project(
    __global const float3 *vecs,     // input array of vectors
    __global const float *pwrs,      // input array of power values
    __global const float3 *rot_mtx,  // rotation matrix
    __global float3 *pivot,          // pivot point
    __global float *x,               // output array for x-coordinates
    __global float *y,               // output array for y-coordinates
    __global float *pwrs_cor         // output array for corrected powers
) {
    const float EPSILON = 0.000001;
    int gid = get_global_id(0);

    float3 pivot_point = pivot[0];
    float3 vec = vecs[gid] - pivot_point;
    float  power = pwrs[gid];

    float u = dot(vec, rot_mtx[0]) + pivot_point.x;
    float v = dot(vec, rot_mtx[1]) + pivot_point.y;
    float w = dot(vec, rot_mtx[2]) + pivot_point.z;

    float length = sqrt(pown(u,2)+pown(v,2)+pown(w,2));
    float3 vecr = (float3)(u,v,w) / length;

    float cos_theta = dot((float3)(0,0,1), vecr);
    float phi  = atan2(v, u);

    float radius = acos(cos_theta);
    float x_transformed = radius * cos(phi);
    float y_transformed = radius * sin(phi);

    float area_factor  = 1.0;
    if(radius > EPSILON) //prevent calculation errors of sin(x)/x for x~0
    {
        area_factor  = sin(radius) / radius;
    }

    x[gid] = x_transformed;
    y[gid] = y_transformed;
    pwrs_cor[gid] = power / area_factor;
}