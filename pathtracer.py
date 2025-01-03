import numpy as np
from multiprocessing import Pool
import cv2
import time

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

class Material:
    def scatter(self, ray, hit_record):
        raise NotImplementedError
class PointLight:
    def __init__(self, position, color,intensity):
        self.position = position
        self.color = color
        self.intensity = intensity
class AreaLight:
    def __init__(self, center, normal, up, width, height, color):
        self.center = center  # Center of the light area
        self.normal = normal  # Normal vector pointing out from the light surface
        self.up = up  # 'Up' vector to define orientation
        self.width = width
        self.height = height
        self.color = color  # Color (intensity) of the light
        # Calculate the light's right and up vectors for area sampling
        self.right = np.cross(self.normal, self.up)
        self.up = np.cross(self.right, self.normal)  # Ensure orthogonality

    def sample_point(self):
        """Sample a random point on the area light's surface."""
        u = np.random.uniform(-0.5, 0.5)
        v = np.random.uniform(-0.5, 0.5)
        return self.center + u * self.width * self.right + v * self.height * self.up

class Lambertian(Material):
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, ray, hit_record):
        _, p, normal, front_face, _ = hit_record
        scatter_direction = normal + random_unit_vector()
        scattered = Ray(p, scatter_direction)
        attenuation = self.albedo
        return scattered, attenuation

class Metal(Material):
    def __init__(self, albedo, fuzz):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1.0)

    def scatter(self, ray, hit_record):
        _, p, normal, front_face, _ = hit_record
        reflected = reflect(unit_vector(ray.direction), normal)
        scattered = Ray(p, reflected + self.fuzz * random_in_unit_sphere())
        # if np.dot(scattered.direction,normal)>0:
        attenuation = self.albedo
        return scattered, attenuation
        return None,None

class Dielectric(Material):
    def __init__(self,albedo, ref_idx):
        self.albedo = albedo
        self.ref_idx = ref_idx

    def scatter(self, ray, hit_record):
        attenuation = np.ones(3)
        _, p, normal, front_face, _ = hit_record
        etai_over_etat = (1.0 / self.ref_idx) if front_face else self.ref_idx

        unit_direction = unit_vector(ray.direction)
        cos_theta = min(np.dot(-unit_direction, normal), 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)

        if etai_over_etat * sin_theta > 1.0:
            reflected = reflect(unit_direction, normal)
            scattered = Ray(p, reflected)
        else:
            refracted = refract(unit_direction, normal, etai_over_etat)
            scattered = Ray(p, refracted)

        return scattered, attenuation

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, t_min, t_max):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant > 0:
            root = np.sqrt(discriminant)
            temp = (-b - root) / (2.0 * a)
            if t_min < temp < t_max:
                p = ray.origin + temp * ray.direction
                normal = (p - self.center) / self.radius
                front_face = np.dot(ray.direction, normal) < 0
                normal = normal if front_face else -normal
                return temp, p, normal, front_face, self.material

            temp = (-b + root) / (2.0 * a)
            if t_min < temp < t_max:
                p = ray.origin + temp * ray.direction
                normal = (p - self.center) / self.radius
                front_face = np.dot(ray.direction, normal) < 0
                normal = normal if front_face else -normal
                return temp, p, normal, front_face, self.material

        return None

def random_unit_vector():
    while True:
        p = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(p) >= 1:
            continue
        return p / np.linalg.norm(p)

def random_in_unit_sphere():
    while True:
        p = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(p) >= 1:
            continue
        return p

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

def refract(uv, n, etai_over_etat):
    cos_theta = min(np.dot(-uv, n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -np.sqrt(abs(1.0 - np.dot(r_out_perp, r_out_perp))) * n
    return r_out_perp + r_out_parallel
def schlick(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0**2
    return r0 + (1 - r0) * (1 - cosine)**5

def unit_vector(v):
    return v / np.linalg.norm(v)

def is_in_shadow(origin, direction, max_dist, world):
    shadow_ray = Ray(origin, direction)
    for obj in world:
        hit = obj.hit(shadow_ray, 0.001, max_dist)
        if hit:
            return True
    return False
def ray_color(ray, world, point_lights,area_lights, depth):
    if depth <= 0:
        return np.zeros(3)  
    
    hit_record = None
    closest_so_far = np.inf
    for sphere in world:
        temp_rec = sphere.hit(ray, 0.001, closest_so_far)
        if temp_rec:
            hit_record = temp_rec
            closest_so_far = temp_rec[0]

    if hit_record:
        emitted_color = np.zeros(3)
        for light in point_lights:
            to_light = light.position - hit_record[1]
            distance_to_light = np.linalg.norm(to_light)
            to_light /= distance_to_light

            if is_in_shadow(hit_record[1], to_light, distance_to_light, world):
                continue

            cos_theta = max(np.dot(hit_record[2], to_light), 0.0)
            emitted_color += hit_record[4].albedo * light.intensity * cos_theta
        for light in area_lights:
            light_sample = light.sample_point()  # Sample a point on the area light
            to_light = light_sample - hit_record[1]
            distance_to_light = np.linalg.norm(to_light)
            to_light /= distance_to_light

            if is_in_shadow(hit_record[1], to_light, distance_to_light, world):
                continue

            cos_theta = max(np.dot(hit_record[2], to_light), 0.0)
            light_intensity = light.color / (distance_to_light ** 2)
            emitted_color += hit_record[4].albedo * light_intensity * cos_theta
        scattered, attenuation = hit_record[4].scatter(ray, hit_record)
        if scattered is None:
            return emitted_color
        return emitted_color + attenuation * ray_color(scattered, world, point_lights,area_lights, depth - 1)
    return np.array([0.1, 0.1, 0.1])  
def render(world,point_lights,area_lights, aspect_ratio, image_width, samples_per_pixel, max_depth):
    image_height = int(image_width / aspect_ratio)

    viewport_height = 2.0
    viewport_width = aspect_ratio * viewport_height
    focal_length = 1.0

    origin = np.array([0,0,0])
    horizontal = np.array([viewport_width, 0, 0])
    vertical = np.array([0, viewport_height, 0])
    lower_left_corner = origin - horizontal / 2 - vertical / 2 - np.array([0, 0, focal_length])

    img = np.zeros((image_height, image_width, 3))

    for j in range(image_height):
        print(f"Scanlines remaining: {image_height - j}")
        for i in range(image_width):
            pixel_color = np.zeros(3)
            for s in range(samples_per_pixel):
                u = (i + np.random.random()) / (image_width - 1)
                v = (j + np.random.random()) / (image_height - 1)
                ray = Ray(origin, lower_left_corner + u * horizontal + v * vertical - origin)
                color_sample = ray_color(ray, world,point_lights,area_lights, max_depth)

                # Implement Russian Roulette here, with a better approach
                if s > 3:  
                    p = max(color_sample)  
                    if np.random.random() > p:
                        break  #
                    color_sample /= p 

                pixel_color += color_sample
            img[j, i] = np.sqrt(pixel_color / samples_per_pixel)

    return img

def main():
    aspect_ratio = 16.0 / 9.0
    image_width = 50
    samples_per_pixel = 64
    max_depth = 10

    world = [
        Sphere(np.asarray([0, 0, -1]), 0.5, Lambertian(np.asarray([0.1, 0.2, 0.5]))),
        Sphere(np.array([0, 100.5, -1]), 100, Lambertian(np.array([0.8, 0.8, 0.0]))),
        Sphere(np.asarray([1, 0, -1]), 0.5, Metal(np.asarray([0.8, 0.6, 0.2]), 0.3)),
        Sphere(np.asarray([-1, 0, -1]), 0.5, Dielectric(np.asarray([0.0,1.0,0.0]),1.5)),
    ]
    point_lights = [
        PointLight(np.array([0, -3, 0]), np.array([1, 1, 1]),5.0),  # White light above the scene
    ]
    area_lights = [
        AreaLight(np.array([0, -2, 0]), np.array([0, 1,0]), np.array([0, 0, 1]), 2, 2, np.array([5, 5, 5])),
    ]


    with Pool() as pool:
        img = pool.apply(render, (world,point_lights,area_lights, aspect_ratio, image_width, samples_per_pixel, max_depth))
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    timestamp = str(time.time_ns())
    filename = f"image_{timestamp}.png"
    cv2.imwrite(filename,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()