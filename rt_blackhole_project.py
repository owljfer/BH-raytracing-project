import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import numba
from numba import prange
import math
import time

#Global variables for mouse handling
prev_mouse_x, prev_mouse_y = 0.0, 0.0
mouse_pressed = False
last_mouse_move = 0.0


class Camera3D:
    def __init__(self):
        #Position
        self.position = np.array([40.0, 10.0, 0.0], dtype=np.float64)

        #Orientation to look at the black hole
        self.yaw = 3 * math.pi / 2
        self.pitch = 0.25

        #Camera properties
        self.fov = math.radians(60.0)
        self.near = 0.1
        self.far = 1000.0

        #Movement speed
        self.move_speed = 2.0
        self.rotate_speed = 0.002

        #Cached matrices
        self.view_matrix = None
        self.need_update = True

    def get_position(self):
        return self.position.copy()

    def get_forwards_vec(self):
        cos_pitch = math.cos(self.pitch)
        return np.array([
            cos_pitch * math.cos(self.yaw),
            math.sin(self.pitch),
            cos_pitch * math.sin(self.yaw),
        ], dtype=np.float64)

    def get_right_vec(self):
        return np.array([
            -math.sin(self.yaw),
            0.0,
            math.cos(self.yaw),
        ], dtype=np.float64)

    def get_up_vec(self):
        forwards = self.get_forwards_vec()
        right = self.get_right_vec()
        return np.cross(right, forwards)

    def move_forwards(self, distance):
        self.position += self.get_forwards_vec() * distance
        self.need_update = True

    def move_right(self, distance):
        self.position += self.get_right_vec() * distance
        self.need_update = True

    def move_up(self, distance):
        self.position += self.get_up_vec() * distance
        self.need_update = True

    def rotate(self, dyaw, dpitch):
        self.yaw += dyaw
        self.pitch += dpitch
        self.pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self.pitch))
        self.yaw = self.yaw % (2 * math.pi)
        self.need_update = True

    def get_view_matrix(self):
        if self.need_update or self.view_matrix is None:
            cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
            cos_pitch, sin_pitch = math.cos(self.pitch), math.sin(self.pitch)

            #Create view matrix - rotation only
            self.view_matrix = np.array([
                [cos_yaw, sin_pitch * sin_yaw, -cos_pitch * sin_yaw],
                [0.0, cos_pitch, sin_pitch],
                [sin_yaw, -sin_pitch * cos_yaw, cos_pitch * cos_yaw]
            ], dtype=np.float64)

            self.need_update = False

        return self.view_matrix.copy()

    def generate_ray_direction(self, pix_x, pix_y, screen_width, screen_height):
        aspect_ratio = screen_width / screen_height
        ndc_x = (2.0 * pix_x) / screen_width - 1.0
        ndc_y = 1.0 - (2.0 * pix_y) / screen_height
        ray_x = ndc_x * math.tan(self.fov / 2.0) * aspect_ratio
        ray_y = ndc_y * math.tan(self.fov / 2.0)
        ray_dir_cam = np.array([ray_x, ray_y, -1.0], dtype=np.float64)
        ray_dir_cam /= np.linalg.norm(ray_dir_cam)
        view_matrix = self.get_view_matrix()
        ray_dir_world = view_matrix @ ray_dir_cam
        return ray_dir_world


class RenderParameters:
    def __init__(self):
        #Black hole parameters
        self.r_s = 2.0  #Schwarzschild radius
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        #Disk parameters
        self.disk_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.disk_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)  #Normal to XZ plane
        self.disk_rin = 3.0  #Inner radius (outside photon sphere at 1.5*r_s)
        self.disk_rout = 20.0  #Larger outer radius for better visibility
        self.disk_height = 0.8  #Thicker disk for visibility
        self.disk_rotation_speed = 0.1  #Orbital velocity factor

        #Rendering parameters
        self.samples = 1  #Start with 1 sample for speed
        self.max_steps = 400
        self.step_size = 0.15
        self.show_stars = True
        self.show_disk = True
        self.shadow_scale = 2.5

        #Performance settings
        self.quality_preset = "fast"  # "fast", "balanced", "quality"
        self.need_update = True

    def apply_quality_preset(self):
        if self.quality_preset == "fast":
            self.samples = 1
            self.max_steps = 200
            self.step_size = 0.2
        elif self.quality_preset == "balanced":
            self.samples = 2
            self.max_steps = 400
            self.step_size = 0.15
        elif self.quality_preset == "quality":
            self.samples = 3
            self.max_steps = 800
            self.step_size = 0.1

@numba.jit(nopython=True, fastmath=True, cache=True)
def normalize_3D(vect):
    length = math.sqrt(vect[0] ** 2 + vect[1] ** 2 + vect[2] ** 2)
    if length > 1e-6:
        return vect / length
    return vect

@numba.jit(nopython=True, fastmath=True, cache=True)
def dot_3D(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@numba.jit(nopython=True, fastmath=True, cache=True)
def cross_3D(a, b):
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ], dtype=np.float64)

@numba.jit(nopython=True, fastmath=True, cache=True)
def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-10:
        return 0.0, 0.0, 0.0
    theta = math.acos(z / r) if r > 1e-10 else 0.0
    phi = math.atan2(y, x)
    return r, theta, phi

@numba.jit(nopython=True, fastmath=True, cache=True)
def spherical_to_cartesian(r, theta, phi):
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta
    return x, y, z

@numba.jit(nopython=True, fastmath=True, cache=True)
def compute_christoffel_symbols(r, theta, r_s):
    if r <= r_s * 1.001:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    f = 1.0 - r_s / r
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    #Key Christoffel symbols for Schwarzschild metric
    g_r_tt = r_s * f / (2.0 * r * r)
    g_r_rr = -r_s / (2.0 * r * r * f)
    g_r_theta_theta = -(r - r_s)
    g_r_phi_phi = -(r - r_s) * sin_theta * sin_theta
    g_theta_r_theta = 1.0 / r
    g_phi_r_phi = 1.0 / r
    g_phi_theta_phi = cos_theta / sin_theta if abs(sin_theta) > 1e-10 else 0.0

    return g_r_tt, g_r_rr, g_r_theta_theta, g_r_phi_phi, g_theta_r_theta, g_phi_r_phi, g_phi_theta_phi

@numba.jit(nopython=True, fastmath=True, cache=True)
def geodesic_equations_3D(pos, vel, r_s):
    t, r, theta, phi = pos
    dt_ds, dr_ds, dtheta_ds, dphi_ds = vel

    if r <= r_s * 1.001:
        return np.zeros(4, dtype=np.float64)

    g_r_tt, g_r_rr, g_r_theta_theta, g_r_phi_phi, g_theta_r_theta, g_phi_r_phi, g_phi_theta_phi = compute_christoffel_symbols(
        r, theta, r_s)

    #Time component
    d2t_ds2 = -2.0 * g_r_tt * dt_ds * dr_ds

    #Radial component
    d2r_ds2 = (-g_r_tt * dt_ds * dt_ds
               - g_r_rr * dr_ds * dr_ds
               - g_r_theta_theta * dtheta_ds * dtheta_ds
               - g_r_phi_phi * dphi_ds * dphi_ds)

    #Theta component
    d2theta_ds2 = (-2.0 * g_theta_r_theta * dr_ds * dtheta_ds
                   - g_phi_theta_phi * dphi_ds * dphi_ds)

    #Phi component
    d2phi_ds2 = (-2.0 * g_phi_r_phi * dr_ds * dphi_ds
                 - 2.0 * g_phi_theta_phi * dtheta_ds * dphi_ds)

    return np.array([d2t_ds2, d2r_ds2, d2theta_ds2, d2phi_ds2], dtype=np.float64)

@numba.jit(nopython=True, fastmath=True, cache=True)
def ray_disk_intersection_3D(origin, direction, params):
    center = np.array([params[0], params[1], params[2]])
    normal = np.array([params[3], params[4], params[5]])
    inner_radius = params[6]
    outer_radius = params[7]
    thickness = params[8]

    #Normalise normal vector
    norm_len = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
    if norm_len < 1e-10:
        return False, 0.0, 0.0, 0.0, 0.0
    normal = normal / norm_len

    #Check if ray is parallel to disk
    denom = dot_3D(direction, normal)
    if abs(denom) < 1e-6:
        return False, 0.0, 0.0, 0.0, 0.0

    #Calculate intersection
    to_center = center - origin
    t = dot_3D(to_center, normal) / denom

    #Check if intersection is behind the ray origin
    if t < 0.001:
        return False, 0.0, 0.0, 0.0, 0.0

    #Calculate hit point
    hit_point = origin + t * direction

    #Check if point is within disk radial bounds
    hit_vector = hit_point - center
    r_hit = math.sqrt(hit_vector[0] ** 2 + hit_vector[1] ** 2 + hit_vector[2] ** 2)
    if r_hit < inner_radius or r_hit > outer_radius:
        return False, 0.0, 0.0, 0.0, 0.0

    #Check if point is within disk thickness
    height_above = abs(dot_3D(hit_vector, normal))
    if height_above > thickness * 0.5:
        return False, 0.0, 0.0, 0.0, 0.0

    #Calculate azimuthal angle for disk
    projected = hit_vector - normal * dot_3D(hit_vector, normal)
    phi = math.atan2(projected[2], projected[0])
    if phi < 0.0:
        phi += 2.0 * math.pi

    return True, t, r_hit, phi, height_above

@numba.jit(nopython=True, fastmath=True, cache=True)
def cal_disk_temp(r, r_in, r_out):
    if r < r_in or r > r_out:
        return 0.0
    r_norm = (r - r_in) / (r_out - r_in)
    #Realistic temperature profile: T ∝ r^(-3/4)
    temp = 1.0 / (0.1 + r_norm ** 0.75)
    return min(temp, 10.0)

@numba.jit(nopython=True, fastmath=True, cache=True)
def blackbody_color(temp):
    temp_norm = min(temp / 5.0, 1.0)

    if temp_norm < 0.33:
        #Red-orange
        r = 1.0
        g = temp_norm * 3.0
        b = 0.0
    elif temp_norm < 0.66:
        #Orange-yellow
        factor = (temp_norm - 0.33) / 0.33
        r = 1.0
        g = 0.5 + factor * 0.5
        b = factor * 0.3
    else:
        #Yellow-white
        factor = (temp_norm - 0.66) / 0.34
        r = 1.0
        g = 1.0
        b = 0.3 + factor * 0.7

    return r, g, b

@numba.jit(nopython=True, fastmath=True, cache=True)
def calc_disk_vel(r, phi, rotation_speed):
    if r < 1e-10:
        return 0.0, 0.0, 0.0

    #Keplerian orbital velocity: v ∝ 1/sqrt(r)
    orbital_speed = rotation_speed / math.sqrt(r)

    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    #Velocity is tangential to the circle at radius r
    v_x = -orbital_speed * sin_phi
    v_y = 0.0
    v_z = orbital_speed * cos_phi

    return v_x, v_y, v_z

@numba.jit(nopython=True, fastmath=True, cache=True)
def doppler_shift(basecolor, vel, dir):
    v_los = dot_3D(vel, dir)  #Line-of-sight velocity

    beta = v_los * 0.1  #Velocity as fraction of c
    doppler_factor = 1.0 / (1.0 + beta)

    #Redshift/blueshift effect
    if beta > 0:  #Receding (redshift)
        r = basecolor[0] * (1.0 + abs(beta))
        g = basecolor[1] * doppler_factor
        b = basecolor[2] * doppler_factor
    else:  #Approaching (blueshift)
        r = basecolor[0] * doppler_factor
        g = basecolor[1] * doppler_factor
        b = basecolor[2] * (1.0 + abs(beta))

    return min(r, 1.0), min(g, 1.0), min(b, 1.0)

@numba.jit(nopython=True, fastmath=True, cache=True)
def calc_grav_redshift(r, r_s):
    if r <= r_s * 1.001:
        return 0.0

    #Gravitational redshift factor: sqrt(1 - 2M/r)
    sqrt_factor = math.sqrt(1.0 - r_s / r)
    if sqrt_factor < 1e-10:
        return 10.0

    redshift_factor = 1 / sqrt_factor
    return redshift_factor

@numba.jit(nopython=True, fastmath=True, cache=True)
def render_disk_pixel_3D(origin, direction, disk_params, render_params):

    #Unpack parameters
    r_s = render_params[0]
    rotation_speed = render_params[1]
    shadow_scale = render_params[2]

    hit, t_hit, r_hit, phi_hit, height_above = ray_disk_intersection_3D(origin, direction, disk_params)

    if not hit:
        return 0.0, 0.0, 0.0, 0.0

    #Calculate temperature
    temp = cal_disk_temp(r_hit, disk_params[6], disk_params[7])

    #Base color from temperature
    base_r, base_g, base_b = blackbody_color(temp)

    #Disk velocity for Doppler effects
    v_x, v_y, v_z = calc_disk_vel(r_hit, phi_hit, rotation_speed)
    vel = np.array([v_x, v_y, v_z])

    #Apply Doppler shift
    doppler_r, doppler_g, doppler_b = doppler_shift((base_r, base_g, base_b), vel, direction)

    #Apply gravitational redshift
    redshift_factor = calc_grav_redshift(r_hit, r_s)

    final_r = doppler_r * (1.0 + 0.1 * (redshift_factor - 1.0))
    final_g = doppler_g * (1.0 + 0.05 * (redshift_factor - 1.0))
    final_b = doppler_b * (1.0 + 0.1 * (redshift_factor - 1.0))

    brightness = temp * 0.3 / (1.0 + 0.1 * (redshift_factor - 1.0))

    #Add disk noise pattern
    noise_factor = 0.8 + 0.4 * (math.sin(phi_hit * 7.0 + r_hit * 2.0))
    brightness *= noise_factor

    #Artificially boost brightness for visibility
    brightness *= 1.5

    return (min(final_r * brightness, 1.0),
            min(final_g * brightness, 1.0),
            min(final_b * brightness, 1.0),
            min(brightness, 1.0))

@numba.jit(nopython=True, fastmath=True, cache=True)
def handle_background(direction, show_stars):
    if not show_stars:
        return np.array([0.01, 0.01, 0.03], dtype=np.float32)

    phi_bg = math.atan2(direction[1], direction[0])
    theta_bg = math.acos(max(-1.0, min(1.0, direction[2])))

    #Add offset to break up grid patterns
    sky_x = phi_bg * 573.0 + theta_bg * 1.234
    sky_y = theta_bg * 671.0 - phi_bg * 0.789

    #Multiple hash layers for better randomness
    hash1 = int(sky_x * 12347.0) ^ int(sky_y * 23456.0)
    hash2 = int(sky_x * 34567.0) ^ int(sky_y * 45678.0)
    hash3 = int(sky_x * 56789.0) ^ int(sky_y * 67890.0)

    #Combine hashes
    final_hash = (hash1 ^ (hash2 << 11) ^ (hash3 << 7)) & 0x7FFFFFFF
    rand_val = (final_hash % 10000) / 10000.0  #0.0 to 1.0

    #Realistic star magnitude distribution
    if rand_val < 0.0005:  # 0.05% - Bright stars
        magnitude = 0.8 + 0.2 * ((final_hash >> 8) % 100) / 100.0
        #Slight color temperature variation
        temp_var = ((final_hash >> 16) % 100) / 100.0
        if temp_var < 0.3:  #Cool stars
            return np.array([magnitude, magnitude * 0.8, magnitude * 0.6], dtype=np.float32)
        elif temp_var < 0.7:  #Sun-like stars
            return np.array([magnitude * 0.95, magnitude, magnitude * 0.9], dtype=np.float32)
        else:  #Hot stars
            return np.array([magnitude * 0.8, magnitude * 0.9, magnitude], dtype=np.float32)

    elif rand_val < 0.003:  #0.25%
        magnitude = 0.4 + 0.3 * ((final_hash >> 12) % 100) / 100.0
        return np.array([magnitude, magnitude * 0.95, magnitude * 0.9], dtype=np.float32)

    elif rand_val < 0.015:  #1.2%
        magnitude = 0.15 + 0.25 * ((final_hash >> 4) % 100) / 100.0
        return np.array([magnitude * 0.9, magnitude, magnitude * 0.8], dtype=np.float32)

    else:
        #Deep space background with subtle nebulosity
        nebula_r = 0.02 + 0.03 * (0.5 + 0.5 * math.sin(sky_x * 0.0123 + sky_y * 0.0234))
        nebula_g = 0.015 + 0.025 * (0.5 + 0.5 * math.sin(sky_x * 0.0345 - sky_y * 0.0156))
        nebula_b = 0.03 + 0.06 * (0.5 + 0.5 * math.sin(sky_x * 0.0078 + sky_y * 0.0456))

        return np.array([nebula_r, nebula_g, nebula_b], dtype=np.float32)

@numba.jit(nopython=True, fastmath=True, cache=True)
def trace_simplified_ray(origin, direction, r_s, disk_params, max_steps, step_size, show_disk, show_stars):
    if not show_disk:
        return handle_background(direction, show_stars)

    #Pre-calculate common values
    origin_to_bh = -origin
    proj = dot_3D(origin_to_bh, direction)
    closest_point = origin + proj * direction
    impact_distance = math.sqrt(closest_point[0] ** 2 + closest_point[1] ** 2 + closest_point[2] ** 2)

    #Shadow check first
    b_crit = 2.6 * r_s
    if impact_distance < b_crit:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    final_color = np.zeros(3, dtype=np.float32)

    #Primary disk intersection
    hit, t_hit, r_hit, phi_hit, height_above = ray_disk_intersection_3D(origin, direction, disk_params)

    if hit:
        render_params = np.array([r_s, 0.1, 2.0], dtype=np.float64)
        r, g, b, a = render_disk_pixel_3D(origin, direction, disk_params, render_params)
        final_color[0] += r
        final_color[1] += g
        final_color[2] += b

    #Enhanced lensing check
    b_lens_min = 2.6 * r_s  #Just outside the shadow
    b_lens_max = 6.0 * r_s  #Increased range for better lensing

    if b_lens_min < impact_distance < b_lens_max:
        #More accurate deflection angle for Schwarzschild metric
        #Deflection angle ≈ 4*M/b for large impact parameters
        deflection_angle = 4.0 * r_s / (2.0 * impact_distance)

        #Calculate perpendicular direction for deflection
        #Need to deflect toward the black hole
        to_bh = -closest_point
        to_bh_normalized = to_bh / math.sqrt(to_bh[0] ** 2 + to_bh[1] ** 2 + to_bh[2] ** 2)

        #Apply deflection by rotating the direction vector toward the black hole
        deflected_dir = direction + deflection_angle * to_bh_normalized
        deflected_dir = deflected_dir / math.sqrt(deflected_dir[0] ** 2 + deflected_dir[1] ** 2 + deflected_dir[2] ** 2)

        #Check if deflected ray hits the disk from a different angle
        hit_lensed, _, r_lensed, phi_lensed, _ = ray_disk_intersection_3D(closest_point, deflected_dir, disk_params)

        if hit_lensed:
            render_params = np.array([r_s, 0.1, 2.0], dtype=np.float64)
            r_l, g_l, b_l, a_l = render_disk_pixel_3D(closest_point, deflected_dir, disk_params, render_params)

            #Brightness depends on how close to the critical radius
            lensing_brightness = 0.5 * (b_lens_max - impact_distance) / (b_lens_max - b_lens_min)
            final_color[0] += lensing_brightness * r_l
            final_color[1] += lensing_brightness * g_l
            final_color[2] += lensing_brightness * b_l

        #Check for additional higher-order images by deflecting more
        if impact_distance < 3.5 * r_s:
            #Second-order deflection
            stronger_deflection = deflection_angle * 2.0
            deflected_dir2 = direction + stronger_deflection * to_bh_normalized
            deflected_dir2 = deflected_dir2 / math.sqrt(
                deflected_dir2[0] ** 2 + deflected_dir2[1] ** 2 + deflected_dir2[2] ** 2)

            hit_lensed2, _, r_lensed2, phi_lensed2, _ = ray_disk_intersection_3D(closest_point, deflected_dir2,
                                                                                 disk_params)

            if hit_lensed2:
                render_params = np.array([r_s, 0.1, 2.0], dtype=np.float64)
                r_l2, g_l2, b_l2, a_l2 = render_disk_pixel_3D(closest_point, deflected_dir2, disk_params, render_params)

                #Second-order image is much dimmer
                lensing_brightness2 = 0.2 * (3.5 * r_s - impact_distance) / (3.5 * r_s - b_lens_min)
                final_color[0] += lensing_brightness2 * r_l2
                final_color[1] += lensing_brightness2 * g_l2
                final_color[2] += lensing_brightness2 * b_l2

    #Return disk color if we found any, otherwise background
    if final_color[0] > 0 or final_color[1] > 0 or final_color[2] > 0:
        return np.array([min(final_color[0], 1.0), min(final_color[1], 1.0), min(final_color[2], 1.0)],
                        dtype=np.float32)

    return handle_background(direction, show_stars)

@numba.jit(nopython=True, fastmath=True, cache=True)
def compute_ray_direction(x, y, sx, sy, samples, width, height, fov, aspect_ratio, view_matrix):
    #Convert pixel coordinates with subpixel
    pixel_x = x + (sx + 0.5) / samples
    pixel_y = y + (sy + 0.5) / samples

    ndc_x = (2.0 * pixel_x / width - 1.0)
    ndc_y = 1.0 - (2.0 * pixel_y / height)

    #Calculate direction in camera space
    ray_x = ndc_x * math.tan(fov / 2.0) * aspect_ratio
    ray_y = ndc_y * math.tan(fov / 2.0)
    ray_z = -1.0

    #Transform to world space
    direction = np.zeros(3, dtype=np.float32)
    direction[0] = view_matrix[0, 0] * ray_x + view_matrix[0, 1] * ray_y + view_matrix[0, 2] * ray_z
    direction[1] = view_matrix[1, 0] * ray_x + view_matrix[1, 1] * ray_y + view_matrix[1, 2] * ray_z
    direction[2] = view_matrix[2, 0] * ray_x + view_matrix[2, 1] * ray_y + view_matrix[2, 2] * ray_z

    #Normalize
    length = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if length > 1e-10:
        direction = direction / length

    return direction

@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
def render_frame_optimised(camera_pos, view_matrix, params_array, width, height):
    image = np.zeros((height, width, 3), dtype=np.float32)

    #Extract parameters
    r_s = params_array[0]
    disk_center_x = params_array[1]
    disk_center_y = params_array[2]
    disk_center_z = params_array[3]
    disk_normal_x = params_array[4]
    disk_normal_y = params_array[5]
    disk_normal_z = params_array[6]
    disk_rin = params_array[7]
    disk_rout = params_array[8]
    disk_height = params_array[9]
    max_steps = int(params_array[10])
    step_size = params_array[11]
    shadow_scale = params_array[12]
    samples = int(params_array[13])
    show_disk = params_array[14] > 0.5
    show_stars = params_array[15] > 0.5
    fov = params_array[16]

    #Set up parameters
    disk_params = np.array([
        disk_center_x, disk_center_y, disk_center_z,
        disk_normal_x, disk_normal_y, disk_normal_z,
        disk_rin, disk_rout, disk_height
    ], dtype=np.float64)

    aspect_ratio = width / height
    actual_samples = max(1, samples)

    #Render all pixels in parallel
    for y in numba.prange(height):
        for x in range(width):
            pixel_color = np.zeros(3, dtype=np.float32)

            for sx in range(actual_samples):
                for sy in range(actual_samples):
                    #Compute ray direction with anti-aliasing
                    ray_dir = compute_ray_direction(
                        x, y, sx, sy, actual_samples,
                        width, height, fov, aspect_ratio, view_matrix
                    )

                    color = trace_simplified_ray(
                        camera_pos, ray_dir, r_s, disk_params,
                        max_steps, step_size, show_disk, show_stars
                    )

                    pixel_color += color

            #Average samples and apply tone mapping
            pixel_color /= (actual_samples * actual_samples)
            pixel_color = pixel_color / (1.0 + pixel_color)  #Simple Reinhard tone mapping

            image[y, x, 0] = max(0.0, min(1.0, pixel_color[0]))
            image[y, x, 1] = max(0.0, min(1.0, pixel_color[1]))
            image[y, x, 2] = max(0.0, min(1.0, pixel_color[2]))

    return image

def render_frame(camera, params, width, height):
    #Get camera parameters
    camera_pos = camera.get_position().astype(np.float32)
    view_matrix = camera.get_view_matrix().astype(np.float32)

    #Pack parameters
    params_array = np.array([
        params.r_s,  #0: Schwarzschild radius
        params.disk_center[0],  #1: Disk center x
        params.disk_center[1],  #2: Disk center y
        params.disk_center[2],  #3: Disk center z
        params.disk_normal[0],  #4: Disk normal x
        params.disk_normal[1],  #5: Disk normal y
        params.disk_normal[2],  #6: Disk normal z
        params.disk_rin,  #7: Disk inner radius
        params.disk_rout,  #8: Disk outer radius
        params.disk_height,  #9: Disk height
        params.max_steps,  #10: Max ray steps
        params.step_size,  #11: Step size
        params.shadow_scale,  #12: Shadow scale
        params.samples,  #13: Anti-aliasing samples
        float(params.show_disk),  #14: Show disk flag
        float(params.show_stars),  #15: Show stars flag
        camera.fov,  #16: Field of view
    ], dtype=np.float32)

    return render_frame_optimised(camera_pos, view_matrix, params_array, width, height)


#OpenGL setup code
def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise Exception(f"Shader compile error: {error}")
    return shader


def create_shader_program():
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec3 aPos;
    out vec2 TexCoords;

    void main()
    {
        gl_Position = vec4(aPos, 1.0);
        TexCoords = (aPos.xy + 1.0) / 2.0;
    }
    """

    FRAGMENT_SHADER = """
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoords;

    uniform sampler2D screenTexture;

    void main()
    {
        FragColor = texture(screenTexture, TexCoords);
    }
    """

    vertex_shader = compile_shader(GL_VERTEX_SHADER, VERTEX_SHADER)
    fragment_shader = compile_shader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise Exception("Shader link error")
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program


def print_instructions():
    print('\n3D Black Hole Ray Tracer Controls:')
    print(' WASD: Move camera forwards/backwards/left/right')
    print(' Q/E: Move camera up/down')
    print(' Mouse Drag: Rotate camera')
    print(' Z/X: Decrease/Increase field of view')
    print(' R/T: Decrease/Increase disk inner radius')
    print(' F/G: Decrease/Increase disk outer radius')
    print(' V/B: Decrease/Increase disk height')
    print(' -/=: Decrease/Increase anti-aliasing samples')
    print(' 1: Toggle stars on/off')
    print(' 2: Toggle disk on/off')
    print(' 0: Cycle quality presets (fast/balanced/quality)')
    print(' P: Print current parameters')
    print(' ESC: Exit\n')


def print_debug_info(camera, params):
    print('\nCurrent Parameters:')
    print(f' Camera Position: [{camera.position[0]:.1f}, {camera.position[1]:.1f}, {camera.position[2]:.1f}]')
    print(f' Camera Orientation: Yaw={math.degrees(camera.yaw):.1f}°, Pitch={math.degrees(camera.pitch):.1f}°')
    print(f' Field of View: {math.degrees(camera.fov):.1f}°')
    print(f' Quality Preset: {params.quality_preset}')
    print(f' Samples: {params.samples}x{params.samples}')
    print(f' Disk Inner Radius: {params.disk_rin:.1f}')
    print(f' Disk Outer Radius: {params.disk_rout:.1f}')
    print(f' Disk Height: {params.disk_height:.1f}')
    print(f' Stars: {"On" if params.show_stars else "Off"}, Disk: {"On" if params.show_disk else "Off"}\n')


def main():
    global prev_mouse_x, prev_mouse_y, mouse_pressed, last_mouse_move

    if not glfw.init():
        raise Exception("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    #Smaller window for faster rendering
    width, height = 480, 360
    window = glfw.create_window(width, height, "Black Hole Ray Tracer", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Window creation failed")

    glfw.make_context_current(window)

    #Create camera and render parameters
    camera = Camera3D()
    params = RenderParameters()

    #Set initial mouse position
    prev_mouse_x, prev_mouse_y = glfw.get_cursor_pos(window)

    def key_callback(window, key, scancode, action, mods):
        step = 0.5 if mods & glfw.MOD_SHIFT else 2.0
        if action == glfw.PRESS or action == glfw.REPEAT:
            #Camera movement
            if key == glfw.KEY_W:
                #Move forward toward black hole
                camera.position[0] -= step
                params.need_update = True
            elif key == glfw.KEY_S:
                #Move backward away from black hole
                camera.position[0] += step
                params.need_update = True
            elif key == glfw.KEY_A:
                #Move left
                camera.position[2] -= step
                params.need_update = True
            elif key == glfw.KEY_D:
                #Move right
                camera.position[2] += step
                params.need_update = True
            elif key == glfw.KEY_Q:
                #Move up
                camera.position[1] += step
                params.need_update = True
            elif key == glfw.KEY_E:
                #Move down
                camera.position[1] -= step
                params.need_update = True

            #Disk parameters
            elif key == glfw.KEY_R:
                params.disk_rin = max(2.5, params.disk_rin - 0.1)
                params.need_update = True
            elif key == glfw.KEY_T:
                params.disk_rin = min(params.disk_rout - 1.0, params.disk_rin + 0.1)
                params.need_update = True
            elif key == glfw.KEY_F:
                params.disk_rout = max(params.disk_rin + 1.0, params.disk_rout - 0.5)
                params.need_update = True
            elif key == glfw.KEY_G:
                params.disk_rout = min(50.0, params.disk_rout + 0.5)
                params.need_update = True
            elif key == glfw.KEY_V:
                params.disk_height = max(0.1, params.disk_height - 0.1)
                params.need_update = True
            elif key == glfw.KEY_B:
                params.disk_height = min(3.0, params.disk_height + 0.1)
                params.need_update = True

            #Quality settings
            elif key == glfw.KEY_MINUS:
                params.samples = max(1, params.samples - 1)
                params.need_update = True
            elif key == glfw.KEY_EQUAL:
                params.samples = min(4, params.samples + 1)
                params.need_update = True
            elif key == glfw.KEY_0:
                if params.quality_preset == "fast":
                    params.quality_preset = "balanced"
                elif params.quality_preset == "balanced":
                    params.quality_preset = "quality"
                else:
                    params.quality_preset = "fast"
                params.apply_quality_preset()
                print(f"Quality preset: {params.quality_preset}")
                params.need_update = True

            #Toggle features
            elif key == glfw.KEY_1:
                params.show_stars = not params.show_stars
                params.need_update = True
            elif key == glfw.KEY_2:
                params.show_disk = not params.show_disk
                params.need_update = True

            #Reset view
            elif key == glfw.KEY_SPACE:
                camera.__init__()
                params.__init__()
                params.need_update = True

            #Print parameters
            elif key == glfw.KEY_P:
                print_debug_info(camera, params)

    def mouse_button_callback(window, button, action, mods):
        global mouse_pressed
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse_pressed = action == glfw.PRESS

    def cursor_pos_callback(window, x_pos, y_pos):
        global prev_mouse_x, prev_mouse_y, last_mouse_move, mouse_pressed

        if mouse_pressed:
            dx = x_pos - prev_mouse_x
            dy = y_pos - prev_mouse_y

            if dx != 0 or dy != 0:
                camera.rotate(dx * 0.002, -dy * 0.002)
                params.need_update = True
                last_mouse_move = time.time()

        prev_mouse_x, prev_mouse_y = x_pos, y_pos

    def scroll_callback(window, x_offset, y_offset):
        camera.move_speed = max(0.5, min(10.0, camera.move_speed + y_offset * 0.1))
        print(f"Movement speed: {camera.move_speed:.1f}")

    #Register callbacks
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    #OpenGL setup
    shader_program = create_shader_program()

    #Create quad for rendering
    quad_vertices = np.array([
        -1.0, -1.0, 0.0,
        1.0, -1.0, 0.0,
        -1.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
    ], dtype=np.float32)

    quad_indices = np.array([
        0, 1, 2,
        2, 1, 3
    ], dtype=np.int32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * quad_vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    #Create texture
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    print_instructions()
    print('Generating initial image, please wait...')
    print('Compiling Numba functions on first run...')

    #Force initial render
    params.need_update = True

    #Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        #Check for ESC key
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        #Render if needed
        if params.need_update:
            print(f"Rendering... (samples: {params.samples}, quality: {params.quality_preset})")
            start_time = time.time()

            #Render frame
            image = render_frame(camera, params, width, height)

            #Convert from float to uint8
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

            end_time = time.time()
            print(f"Render completed in {end_time - start_time:.2f} seconds")

            #Update texture
            image_uint8 = np.flipud(image_uint8)  # Flip for OpenGL
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, image_uint8)

            params.need_update = False

        #Render to screen
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glUniform1i(glGetUniformLocation(shader_program, "screenTexture"), 0)

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == '__main__':
    main()