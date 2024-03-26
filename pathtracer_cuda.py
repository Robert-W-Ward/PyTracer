import numpy as np
import time
import numba
import math
from numba import cuda, float32, int32, boolean
from numba.experimental import jitclass

# Define constant values for material types
LAMBERTIAN = 0
METAL = 1
DIELECTRIC = 2

# Define a structured array for spheres
sphere_dtype = np.dtype([
    ('center', np.float32, (3,)),  # Specify shape as a tuple
    ('radius', np.float32),
    ('material_type', np.int32),
    ('material_albedo', np.float32, (3,)),  # Specify shape as a tuple
    ('material_ref_idx', np.float32),
    ('material_fuzz', np.float32)
])

# Define a structured array for point lights
point_light_dtype = np.dtype([
    ('position', np.float32, (3,)),
    ('color', np.float32, (3,)),
    ('intensity', np.float32)
])

# Define Numba CUDA kernels
@cuda.jit(device=True)
def ray_color_kernel(ray_origin,ray_dir, world, point_lights, area_lights, depth):
    if depth <= 0:
        return cuda.local.array(3, dtype=numba.float32)

    hit_record = None
    closest_so_far = np.inf

    for sphere in world:
        temp_rec = sphere.hit(ray_origin,ray_dir, 0.001, closest_so_far)
        if temp_rec:
            hit_record = temp_rec
            closest_so_far = temp_rec[0]

    if hit_record:
        emitted_color = cuda.local.array(3, dtype=numba.float32)

        for light in point_lights:
            to_light = light.position - hit_record[1]
            distance_to_light = cuda.local.array(1, dtype=numba.float32)
            distance_to_light[0] = np.linalg.norm(to_light)
            to_light /= distance_to_light[0]

            if is_in_shadow(hit_record[1], to_light, distance_to_light[0], world):
                continue

            cos_theta = max(np.dot(hit_record[2], to_light), 0.0)
            emitted_color += hit_record[4].albedo * light.intensity * cos_theta

        for light in area_lights:
            light_sample = light.sample_point()
            to_light = light_sample - hit_record[1]
            distance_to_light = cuda.local.array(1, dtype=numba.float32)
            distance_to_light[0] = np.linalg.norm(to_light)
            to_light /= distance_to_light[0]

            if is_in_shadow(hit_record[1], to_light, distance_to_light[0], world):
                continue

            cos_theta = max(np.dot(hit_record[2], to_light), 0.0)
            light_intensity = light.color / (distance_to_light[0] ** 2)
            emitted_color += hit_record[4].albedo * light_intensity * cos_theta

        scattered, attenuation = hit_record[4].scatter(ray_origin,ray_dir, hit_record)
        if scattered is None:
            return emitted_color
        return emitted_color + attenuation * ray_color_kernel(scattered, world, point_lights, area_lights, depth - 1)

    return cuda.local.array([0.1, 0.1, 0.1], dtype=numba.float32)

@cuda.jit
def render_kernel(world, point_lights, area_lights, aspect_ratio, image_width,image_height, origin,lower_left_corner,horizontal,vertical, samples_per_pixel, max_depth, img):
    i, j = cuda.grid(2)
    if i < image_width and j < image_height:
        pixel_color = cuda.local.array(3, dtype=numba.float32)

        for s in range(samples_per_pixel):
            u = (i + np.random.random()) / (image_width - 1)
            v = (j + np.random.random()) / (image_height - 1)

            ray_dir = cuda.local.array(3,dtype=numba.float32)
                        
            for k in range(3):
                ray_dir[k] = (lower_left_corner[k] + u * horizontal[k] + v * vertical[k]) - origin[k]

            # Directly create the ray's origin and direction for use in ray_color_kernel
            ray_origin = cuda.local.array(3, dtype=numba.float32)
            for k in range(3):
                ray_origin[k] = origin[k]  # The origin remains constant for all rays from the camera


            color_sample = ray_color_kernel(ray_origin,ray_dir, world, point_lights, area_lights, max_depth)

            if s > 3:
                p = max(color_sample)
                if np.random.random() > p:
                    break
                # color_sample /= p

            pixel_color += color_sample

        img[j, i] = np.sqrt(pixel_color / samples_per_pixel)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cuda.jit(device=True)
def sphere_hit(center, radius, ray_origin, ray_direction, t_min, t_max):
    oc = (ray_origin[0] - center[0], ray_origin[1] - center[1], ray_origin[2] - center[2])
    a = dot(ray_direction, ray_direction)
    half_b = dot(oc, ray_direction)
    c = dot(oc, oc) - radius * radius
    discriminant = half_b * half_b - a * c
    if discriminant > 0:
        sqrt_discriminant = math.sqrt(discriminant)
        root = (-half_b - sqrt_discriminant) / a
        if t_min < root < t_max:
            hit_point = (ray_origin[0] + root * ray_direction[0], ray_origin[1] + root * ray_direction[1], ray_origin[2] + root * ray_direction[2])
            normal = ((hit_point[0] - center[0]) / radius, (hit_point[1] - center[1]) / radius, (hit_point[2] - center[2]) / radius)
            return True, root, hit_point, normal
        root = (-half_b + sqrt_discriminant) / a
        if t_min < root < t_max:
            hit_point = (ray_origin[0] + root * ray_direction[0], ray_origin[1] + root * ray_direction[1], ray_origin[2] + root * ray_direction[2])
            normal = ((hit_point[0] - center[0]) / radius, (hit_point[1] - center[1]) / radius, (hit_point[2] - center[2]) / radius)
            return True, root, hit_point, normal
    return False, 0, (0, 0, 0), (0, 0, 0)

@cuda.jit(device=True)
def is_in_shadow(origin, direction, max_dist, spheres):
    for i in range(len(spheres)):
        center, radius = spheres[i][:3], spheres[i][3]
        hit, _, _, _ = sphere_hit(center, radius, origin, direction, 0.001, max_dist)
        if hit:
            return True
    return False

@cuda.jit(device=True)
def random_unit_vector(state):
    # CUDA device function for generating a random unit vector
    while True:
        p = 2.0 * numba.cuda.random.xoroshiro128p_uniform_float32(state, 3) - 1.0
        norm_p = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
        if norm_p >= 1 or norm_p == 0:
            continue
        return p / norm_p

@cuda.jit(device=True)
def random_in_unit_sphere(state):
    # CUDA device function for generating a random vector inside a unit sphere
    while True:
        p = 2.0 * numba.cuda.random.xoroshiro128p_uniform_float32(state, 3) - 1.0
        if math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) >= 1:
            continue
        return p

@cuda.jit(device=True)
def reflect(v, n):
    # CUDA device function for reflection calculation
    vn = 2 * (v[0] * n[0] + v[1] * n[1] + v[2] * n[2])
    return v[0] - vn * n[0], v[1] - vn * n[1], v[2] - vn * n[2]

@cuda.jit(device=True)
def refract(uv, n, etai_over_etat):
    # CUDA device function for refraction calculation
    cos_theta = min(-(uv[0] * n[0] + uv[1] * n[1] + uv[2] * n[2]), 1.0)
    r_out_perp = etai_over_etat * (uv[0] + cos_theta * n[0], uv[1] + cos_theta * n[1], uv[2] + cos_theta * n[2])
    r_out_parallel = -math.sqrt(abs(1.0 - (r_out_perp[0] ** 2 + r_out_perp[1] ** 2 + r_out_perp[2] ** 2))) * n
    return r_out_perp[0] + r_out_parallel[0], r_out_perp[1] + r_out_parallel[1], r_out_perp[2] + r_out_parallel[2]

@cuda.jit(device=True)
def schlick(cosine, ref_idx):
    # CUDA device function for Schlick's approximation
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 ** 2
    return r0 + (1 - r0) * ((1 - cosine) ** 5)

@cuda.jit(device=True)
def unit_vector(v):
    # CUDA device function for calculating a unit vector
    norm_v = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return v[0] / norm_v, v[1] / norm_v, v[2] / norm_v


def main():
    aspect_ratio = 16.0 / 9.0
    image_width = 1280
    image_height = int(image_width / aspect_ratio)
    samples_per_pixel = 64
    max_depth = 10

    viewport_height = 2.0
    viewport_width = aspect_ratio * viewport_height
    focal_length = 1.0

    origin = np.array([0,0,0],dtype=np.float32)
    horizontal = np.array([viewport_width, 0, 0],dtype=np.float32)
    vertical = np.array([0, viewport_height, 0],dtype=np.float32)
    lower_left_corner = origin - horizontal / 2 - vertical / 2 - np.array([0, 0, focal_length],dtype=np.float32)

    img = np.zeros((image_height, image_width, 3),dtype=np.float32)

    world = np.array([
        ((0, 0, -1), 0.5, LAMBERTIAN, (0.1, 0.2, 0.5), 0.0, 0.0),
        ((0, 100.5, -1), 100, LAMBERTIAN, (0.8, 0.8, 0.0), 0.0, 0.0),
        ((1, 0, -1), 0.5, METAL, (0.8, 0.6, 0.2), 0.0, 0.3),
        ((-1, 0, -1), 0.5, DIELECTRIC, (0.0, 1.0, 0.0), 1.5, 0.0)
    ], dtype=sphere_dtype)

    # Create the point lights array
    point_lights = np.array([
        ((0, -3, 0), (1, 1, 1), 5.0)
    ], dtype=point_light_dtype)

    area_lights = [
        # Define area lights
    ]

    # Allocate memory on the GPU
    d_image = cuda.device_array((image_height, image_width, 3), dtype=np.float32)
    d_world = cuda.to_device(world)
    d_point_lights = cuda.to_device(point_lights)
    d_area_lights = cuda.to_device(area_lights)

    # Launch Numba CUDA kernels
    blocks_x = (image_width + 15) // 16
    blocks_y = (image_height + 15) // 16
    blocks = (blocks_x, blocks_y)
    threads = (16, 16)
    render_kernel[blocks, threads](d_world, d_point_lights, d_area_lights, aspect_ratio, image_width,image_height,origin,lower_left_corner,horizontal,vertical, samples_per_pixel, max_depth, d_image)

    # Transfer results back to the host
    img = d_image.copy_to_host()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # Save the image
    timestamp = str(time.time_ns())
    filename = f"image_{timestamp}.png"
    import cv2
    cv2.imwrite(filename,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()