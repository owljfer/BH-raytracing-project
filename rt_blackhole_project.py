import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import numba
from numba import prange
import math

class render_parameters:
    #Contains all parameters for rendering
    def __init__(self):
        #BH parameters
        self.r_s = 2.0 #Schwarzschild radius
        self.r_cam = 40.0 #Camera distance
        self.shadow_scale = 2.0 #Shadow size multiplier

        #Disk parameters
        self.disk_rin = 3.0 #Outside photon sphere
        self.disk_rout = 15.0 #Large disk
        self.disk_height = 0.3 #Thin disk height

        #Camera setup
        self.fov = math.radians(45.0)
        self.camera_rotation_x = 0.0 #Camera rotation X
        self.camera_rotation_y = 0.0 #Camera rotation Y

        #Rendering parameters
        self.samples = 2 # Anti-aliasing samples (2x2)
        self.show_stars = True
        self.show_disk = True

        #Parameter changes tracking
        self.need_update = True

#Global parameters
params = render_parameters()
prev_mouse_x, prev_mouse_y = 0.0, 0.0
mouse_sensitivity = 0.002

#Function to handle keyboard input
def key_callback(window, key, scancode, action, mods):
    global params
    if action == glfw.PRESS or action == glfw.REPEAT:
        step = 0.1 if mods & glfw.MOD_SHIFT else 1.0

        #Distance controls
        if key == glfw.KEY_W:
            params.r_cam = max(5.0, params.r_cam - step)
            params.need_update = True
        elif key == glfw.KEY_S:
            params.r_cam = min(100.0, params.r_cam + step)
            params.need_update = True

        #Disk inner radius
        elif key == glfw.KEY_Q:
            params.disk_rin = max(2.5, params.disk_rin - step * 0.1)
            params.need_update = True
        elif key == glfw.KEY_E:
            params.disk_rin = min(params.disk_rout - 1.0, params.disk_rin + step * 0.1)
            params.need_update = True

        #Disk outer radius
        elif key == glfw.KEY_R:
            params.disk_rout = max(params.disk_height - 1.0, params.disk_rout + step * 0.5)
            params.need_update = True
        elif key == glfw.KEY_T:
            params.disk_rout = min(50.0, params.disk_rout - step * 0.5)
            params.need_update = True

        #Disk height
        elif key == glfw.KEY_G:
            params.disk_height = max(0.1, params.disk_height - step * 0.1)
            params.need_update = True
        elif key == glfw.KEY_H:
            params.disk_height = min(3.0, params.disk_height + step * 0.1)
            params.need_update = True

        #Shadow scale
        elif key == glfw.KEY_O:
            params.shadow_scale = max(1.0, params.shadow_scale - step * 0.1)
            params.need_update = True
        elif key == glfw.KEY_I:
            params.shadow_scale = min(5.0, params.shadow_scale + step * 0.1)
            params.need_update = True

        #Field of view
        elif key == glfw.KEY_Z:
            params.fov = max(math.radians(10.0), params.fov - step * 0.05)
            params.need_update = True
        elif key == glfw.KEY_X:
            params.fov = min(math.radians(120.0), params.fov + step * 0.05)

        #Anti-aliasing samples
        elif key == glfw.KEY_MINUS:
            params.samples = max(1, params.samples - 1)
            params.need_update = True
        elif key == glfw.KEY_EQUAL:
            params.samples = min(4, params.samples + 1)
            params.need_update = True

        #Toggle stars
        elif key == glfw.KEY_1:
            params.show_stars = not params.show_stars
            params.need_update = True

        #Toggle disk
        elif key == glfw.KEY_2:
            params.show_disk = not params.show_disk
            params.need_update = True

        #Reset view
        elif key == glfw.KEY_SPACE:
            params.__init__()
            params.need_update = True

        #Print parameters
        elif key == glfw.KEY_P:
            print_params()

#Function to handle mouse movement
def cursor_position_callback(window, x_pos, y_pos):
    global params, prev_mouse_x, prev_mouse_y, mouse_sensitivity

    if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
        dx = x_pos - prev_mouse_x
        dy = y_pos - prev_mouse_y

        params.camera_rotation_y += dx * mouse_sensitivity
        params.camera_rotation_x += dy * mouse_sensitivity

        #Clamp vertical rotation
        params.camera_rotation_x = max(-math.pi/2, min(math.pi/2, params.camera_rotation_x))
        params.need_update = True

    prev_mouse_x, prev_mouse_y = x_pos, y_pos

#Handle mouse scroll for zoom
def scroll_callback(window, x_offset, y_offset):
    global params

    zoom_factor = 1.1 if y_offset > 0 else 0.9
    params.r_cam = max(5.0, min(100.0, params.r_cam*zoom_factor))
    params.need_update = True

#Function to print current parameters
def print_params():
    print('\nCurrent Rendering Parameters:')
    print(f' Camera Distance: {params.r_cam:.2f}')
    print(f' Disk Inner Radius: {params.disk_rin:.2f}')
    print(f' Disk Outer Radius: {params.disk_rout:.2f}')
    print(f' Disk Height: {params.disk_height:.2f}')
    print(f' Shadow Scale: {params.shadow_scale:.2f}\n')
    print(f' Disk Height: {params.disk_height:.2f}')
    print(f' Field of View: {math.degrees(params.fov):.2f} degrees')
    print(f' Anti-Aliasing Samples: {params.samples}x{params.samples}')
    print(f' Show Stars: {"On" if params.show_stars else "Off"}')
    print(f' Show Disk: {"On" if params.show_disk else "Off"}')


#Instructions for controls
def print_instructions():
    print('\nControls:')
    print(' W/S: Move camera closer/farther')
    print(' Q/E: Decrease/Increase disk inner radius')
    print(' R/T: Increase/Decrease disk outer radius')
    print(' Z/X: Decrease/Increase field of view')
    print(' -/=: Decrease/Increase anti-aliasing samples')
    print(' 1: Toggle stars on/off')
    print(' 2: Toggle disk on/off')
    print(' SPACE: Reset view to default')
    print(' P: Print current parameters')
    print(' Mouse Drag: Rotate camera')
    print(' Mouse Scroll: Zoom in/out')

#Ray tracing function with Numba optimization and configurable parameters
@numba.jit(parallel=True)
def geodesic_raytrace(width, height, r_s, r_cam, disk_rin, disk_rout, disk_height, fov, samples, show_stars, show_disk, cam_rot_x, cam_rot_y, shadow_scale):
    #Pixels on screen
    image = np.zeros((height, width, 3), dtype=np.uint8)

    cos_x, sin_x = math.cos(cam_rot_x), math.sin(cam_rot_x)
    cos_y, sin_y = math.cos(cam_rot_y), math.sin(cam_rot_y)

    #Photon sphere and critical parameters
    r_photon = 1.5 * r_s  #Photon sphere

    #Dynamic shadow
    shadow_angular_size = math.atan(shadow_scale * r_s / r_cam) #Approximate angular size of black hole shadow

    #Scale capture
    capture_scale = math.sqrt(10/r_cam)  #Capture radius scaling based on camera distance
    effect_capture_r =  2.6 * r_s * capture_scale  #Effective capture radius for lensing effects

    #Loop for all pixels
    for y in prange(height):
        for x in range(width):
            #Convert pixel to angular coordinates with sub-pixel sampling for anti-aliasing
            total_r, total_g, total_b = 0.0, 0.0, 0.0

            for dx in range(samples):
                for dy in range(samples):
                    #Sub-pixels
                    px = x + (dx + 0.5) / samples - 0.5
                    py = y + (dy + 0.5) / samples - 0.5

                    #Angular coordinates
                    alpha = (2.0 * px / width - 1.0) * math.tan(fov / 2.0)
                    beta = (1.0 - 2.0 * py / height) * math.tan(fov / 2.0) * height / width

                    #Rotate according to camera orientation
                    alpha_rot = alpha * cos_y - r_cam * sin_y * 0.001
                    beta_rot = beta * cos_x + alpha * sin_x * 0.1

                    #Impact parameter
                    b = r_cam * math.sqrt(alpha_rot ** 2 + beta_rot ** 2)

                    #Singularity check
                    if b < 1e-6:
                        continue

                    angular_distance = math.sqrt(alpha_rot ** 2 + beta_rot ** 2)

                    # Physics-based capture
                    physics_capture = b < effect_capture_r

                    # Perspective-based capture
                    perspective_capture = angular_distance < shadow_angular_size * 0.5

                    if physics_capture or perspective_capture:
                        #Inside shadow
                        continue

                    #Check if photon is captured (for photon sphere)
                    if b < 2.6 * r_s:
                        continue

                    #Improved ray tracing
                    #Calculate deflection angle more accurately
                    u_max = 1.0 / (b * math.sqrt(1.0 - r_s * r_s / (4.0 * b * b)))
                    r_min = 1.0 / u_max

                    #Check if ray falls
                    if r_min <= r_s * 1.05:
                        continue

                    #More accurate deflection
                    if b > 3.0 * r_s:
                        deflection = 4.0 * r_s / b + 15.0 * math.pi * r_s ** 2 / (4.0 * b ** 2)  #Second-order correction
                    else:
                        deflection = 4.0 * r_s / b  #First-order only for close approaches

                    #Check disk intersection
                    hit_disk = False
                    if show_disk and disk_rin <= r_min <=  disk_rout:
                        #Check if ray passes through disk plane (accounting for thickness)
                        #Assume ray hits if closest approach is in disk range
                        disk_height_at_r = disk_height * (disk_rout - r_min) / (disk_rout - disk_rin)

                        #Probability of hitting based on disk thickness and ray angle
                        hit_prob = min(1.0, disk_height_at_r / abs(beta * r_cam))

                        if hit_prob > 0.3 or abs(beta) < disk_height_at_r / r_cam:
                            hit_disk = True
                            disk_r = r_min

                            #Temperature
                            disk_temp = 1.0 - (disk_r - disk_rin) / (disk_rout - disk_rin)
                            disk_temp = disk_temp ** 0.7  #Gradual temperature falloff

                            #Added Doppler effects
                            #Doppler shift approximation
                            velocity_factor = math.sqrt(r_s / disk_r)  #Orbital velocity
                            doppler_shift = 1.0 + 0.3 * velocity_factor * alpha  #Simplified relativistic Doppler

                            #Blackbody temperature colors
                            temp_effective = disk_temp * doppler_shift
                            temp_effective = max(0.1, min(1.0, temp_effective))

                            #Blackbody colors
                            if temp_effective > 0.9:
                                #Very hot - blue-white
                                r, g, b = 200, 220, 255
                            elif temp_effective > 0.7:
                                #Hot - white
                                r, g, b = 255, 255, 240
                            elif temp_effective > 0.5:
                                #Warm - yellow-white
                                r, g, b = 255, 240, 180
                            elif temp_effective > 0.3:
                                #Cool - orange
                                r, g, b = 255, 180, 80
                            else:
                                #Cold - red
                                r, g, b = 255, 100, 30

                            #Apply brightness and gravitational redshift
                            gravity_redshift = math.sqrt(1.0 - r_s / disk_r)
                            brightness = (0.3 + 0.7 * temp_effective) * gravity_redshift

                            #Add disk turbulence/noise
                            noise_seed = int((alpha * 1000 + beta * 1000 + disk_r * 100)) % 100
                            if noise_seed < 10:
                                brightness *= 1.3
                            elif noise_seed < 30:
                                brightness *= 0.8

                            total_r += r * brightness
                            total_g += g * brightness
                            total_b += b * brightness

                    if not hit_disk:
                        #Gravitational lensing
                        final_alpha = alpha_rot + deflection * math.cos(math.atan2(beta_rot, alpha_rot))
                        final_beta = beta_rot + deflection * math.sin(math.atan2(beta_rot, alpha_rot))

                        if show_stars:
                            #Starry background
                            phi_bg = math.atan2(final_beta, final_alpha)
                            r_bg = math.sqrt(final_alpha ** 2 + final_beta ** 2)

                            #Multiple layer background
                            #Large-scale structure
                            bg_r = 1 + int(5 * (0.5 + 0.5 * math.sin(phi_bg * 3 + r_bg * 5)))
                            bg_g = 1 + int(4 * (0.5 + 0.5 * math.cos(phi_bg * 4 - r_bg * 3)))
                            bg_b = 2 + int(10 * (0.5 + 0.5 * math.sin(phi_bg * 2 + r_bg * 7)))

                            #Medium structure)
                            nebula_r = int(15 * math.exp(-((final_alpha - 0.5) ** 2 + (final_beta - 0.3) ** 2) * 8))
                            nebula_b = int(12 * math.exp(-((final_alpha + 0.3) ** 2 + (final_beta - 0.5) ** 2) * 6))

                            bg_r += nebula_r
                            bg_b += nebula_b

                            #Stars
                            star_seed = int((phi_bg * 1000 + r_bg * 500)) % 300
                            if star_seed < 3:  #Bright
                                bg_r, bg_g, bg_b = 255, 255, 240
                            elif star_seed < 8:  #Medium
                                bg_r += 120
                                bg_g += 100
                                bg_b += 80
                            elif star_seed < 20:  #Dim
                                bg_r += 40
                                bg_g += 40
                                bg_b += 30

                            #Apply lensing brightness enhancement
                            lensing_factor = 1.0 + 0.5 * deflection / (4.0 * r_s / b) if b > r_s else 1.0

                            total_r += bg_r * lensing_factor
                            total_g += bg_g * lensing_factor
                            total_b += bg_b * lensing_factor

            #Average the samples and apply final processing
            avg_r = total_r / (samples * samples)
            avg_g = total_g / (samples * samples)
            avg_b = total_b / (samples * samples)

            #Tone mapping for HDR-like effect
            avg_r = 255 * (avg_r / (avg_r + 255))
            avg_g = 255 * (avg_g / (avg_g + 255))
            avg_b = 255 * (avg_b / (avg_b + 255))

            image[y, x, 0] = int(min(255, max(0, avg_r)))
            image[y, x, 1] = int(min(255, max(0, avg_g)))
            image[y, x, 2] = int(min(255, max(0, avg_b)))

    return image

def ray_sphere_intersect(origin, direction, center, radius):
    vect = center - origin
    closest_approach = np.dot(vect, direction)
    dsqr = np.dot(vect, vect) - closest_approach * closest_approach
    if dsqr > radius * radius:
        return False, None
    half_chord = np.sqrt(radius * radius - dsqr)
    d0 = closest_approach - half_chord
    d1 = closest_approach + half_chord
    if d0 < 0 and d1 < 0:
        return False, None
    d = d0 if d0 > 0 else d1
    return True, d


# OpenGL setup code remains the same
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


def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise Exception(f"Shader compile error: {error}")
    return shader


def create_shader_program():
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


def main():
    global params, prev_mouse_x, prev_mouse_y

    if not glfw.init():
        raise Exception("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    width, height = 640, 480
    window = glfw.create_window(width, height, "Geodesic Ray Tracer - Black Hole", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Window creation failed")

    glfw.make_context_current(window)

    # Set callbacks
    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Initialize previous mouse position
    prev_mouse_x, prev_mouse_y = glfw.get_cursor_pos(window)

    shader_program = create_shader_program()

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

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    print_instructions()
    print('Generating initial image, please wait...')

    import time

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Check for ESC key
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # Re-render if parameters changed
        if params.need_update:
            print(f"Rendering... (samples: {params.samples}, camera: {params.r_cam:.1f})")
            start_time = time.time()

            image = geodesic_raytrace(
                width, height,
                params.r_s, params.r_cam,
                params.disk_rin, params.disk_rout, params.disk_height,
                params.fov, params.samples,
                params.show_stars, params.show_disk,
                params.camera_rotation_x, params.camera_rotation_y,
                params.shadow_scale
            )

            end_time = time.time()
            print(f"Render completed in {end_time - start_time:.2f} seconds")

            # Update texture
            image = np.flipud(image)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, image)

            params.needs_update = False

        # Render frame
        glClearColor(0.1, 0.1, 0.1, 1.0)
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