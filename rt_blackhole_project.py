import glfw
from OpenGL.GL import *
import numpy as np
import ctypes
import numba
from numba import prange
import math


@numba.jit(parallel=True)
def geodesic_raytrace(width, height):
    #Pixels on screen
    image = np.zeros((height, width, 3), dtype=np.uint8)

    #Define BH variables (G=M=c=1)
    rs = 2.0  #Schwarzschild radius
    r_cam = 40.0  #Camera distance

    #Disk parameters
    disk_rin = 3.0  #Outside photon sphere
    disk_rout = 15.0  #Large disk
    disk_height = 0.3  #Thin disk height

    #Camera setup
    fov = math.radians(45.0)

    #Photon sphere and critical parameters
    r_photon = 1.5 * rs  #Photon sphere

    #Loop for all pixels
    for y in prange(height):
        for x in range(width):
            #Convert pixel to angular coordinates with sub-pixel sampling for anti-aliasing
            samples = 2  # 2x2 supersampling
            total_r, total_g, total_b = 0.0, 0.0, 0.0

            for dx in range(samples):
                for dy in range(samples):
                    #Sub-pixels
                    px = x + (dx + 0.5) / samples - 0.5
                    py = y + (dy + 0.5) / samples - 0.5

                    #Angular coordinates
                    alpha = (2.0 * px / width - 1.0) * math.tan(fov / 2.0)
                    beta = (1.0 - 2.0 * py / height) * math.tan(fov / 2.0) * height / width

                    #Impact parameter
                    b = r_cam * math.sqrt(alpha ** 2 + beta ** 2)

                    #Singularity check
                    if b < 1e-6:
                        total_r += 0
                        total_g += 0
                        total_b += 0
                        continue

                    #More accurate impact parameter
                    b_crit = r_photon * math.sqrt(27.0 / 4.0 * (rs / r_photon) ** 2)

                    #Check if photon is captured (for photon sphere)
                    if b < 2.6 * rs:  #Capture radius
                        total_r += 0
                        total_g += 0
                        total_b += 0
                        continue

                    #Improved ray tracing
                    #Calculate deflection angle more accurately
                    u_max = 1.0 / (b * math.sqrt(1.0 - rs * rs / (4.0 * b * b)))
                    r_min = 1.0 / u_max

                    #Check if ray falls
                    if r_min <= rs * 1.05:
                        total_r += 0
                        total_g += 0
                        total_b += 0
                        continue

                    #More accurate deflection
                    if b > 3.0 * rs:
                        deflection = 4.0 * rs / b + 15.0 * math.pi * rs ** 2 / (4.0 * b ** 2)  #Second-order correction
                    else:
                        deflection = 4.0 * rs / b  #First-order only for close approaches

                    #Check disk intersection
                    hit_disk = False
                    disk_r = 0.0
                    disk_temp = 0.0

                    if disk_rin <= r_min <= disk_rout:
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

                    if hit_disk:
                        #Added Doppler effects
                        #Doppler shift approximation
                        velocity_factor = math.sqrt(rs / disk_r)  #Orbital velocity
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
                        gravity_redshift = math.sqrt(1.0 - rs / disk_r)
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

                    else:
                        #Gravitational lensing
                        final_alpha = alpha + deflection * math.cos(math.atan2(beta, alpha))
                        final_beta = beta + deflection * math.sin(math.atan2(beta, alpha))

                        #More interesting background
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
                        lensing_factor = 1.0 + 0.5 * deflection / (4.0 * rs / b) if b > rs else 1.0

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

    #Progress tracking
    print("Rendering geodesic ray traced black hole...")
    print("This may take a moment on first run (Numba compilation)...")

    import time
    start_time = time.time()
    image = geodesic_raytrace(width, height)
    end_time = time.time()

    print(f"Render completed in {end_time - start_time:.2f} seconds")

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    image = np.flipud(image)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, image)

    glBindTexture(GL_TEXTURE_2D, 0)

    while not glfw.window_should_close(window):
        glfw.poll_events()

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