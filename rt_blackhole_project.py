import glfw
from OpenGL.GL import *
import numpy as np
import ctypes

def ray_sphere_intersect(origin, direction, center, radius):
    vect = center - origin #vector from center to ray origin
    closest_approach = np.dot(vect, direction) #dot vector and direction vector to get distance from origin to closest point to speher sentre
    dsqr = np.dot(vect, vect) - closest_approach * closest_approach #Size of centre to origin minus size of closest approach
    if dsqr > radius * radius:
        return False, None
    half_chord = np.sqrt(radius * radius - dsqr) #Half chord dis
    d0 = closest_approach - half_chord #Find intersection points
    d1 = closest_approach + half_chord
    if d0 < 0 and d1 < 0: #Check if behind the ray origin cause then invalid
        return False, None
    d = d0 if d0 > 0 else d1
    return True, d

#We are spraying rays at every pixel
def simple_blackhole_raytrace(width, height):
    image = np.zeros((height, width, 3),dtype=np.uint8)
    camera =  np.array([0,0,0])
    bh_center = np.array([0,0,-5])
    bh_radius = 1.0
    accretion_inner = 1.1
    accretion_outer = 1.4

    for y in range(height):
        for x in range(width):
            px = (2 * (x+0.5)/width - 1) * (width/height) #Converts co-ords to pixel ones
            py = 1 -2 * (y+0.5)/height
            ray_dir  = np.array([py, px, -1]) #Ray heading towards this point
            ray_dir = ray_dir / np.linalg.norm(ray_dir)



            hit, dist = ray_sphere_intersect(camera, ray_dir, bh_center, bh_radius) #See if ray it hits
            if hit:
                image[y,x] = [0,0,0]
            else:
                d = (bh_center[2] - camera[2])/ray_dir[2]
                hitpoint = camera + ray_dir*d
                dist_plane = np.linalg.norm(hitpoint[:2] -bh_center[:2])
                if accretion_inner < dist_plane < accretion_outer:
                    intensity = 1 - (dist_plane - accretion_inner) / (accretion_outer - accretion_inner)
                    color = np.array([255, 160, 40]) * intensity + np.array([10,20,50]) * (1 - intensity)
                    image[y, x] = color.astype(np.uint8)
                else:
                    image[y, x] = [10,20,50]
    return image

#This is an OpenGL fullscreen setup, we render this firt and then our shader is applied to it
#Fullscreen Vertices
quad_vertices = np.array([
    -1.0, -1.0, 0.0,
    1.0, -1.0, 0.0,
    -1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
], dtype=np.float32)

#Indices for two triangles to make a rectangle to fill the whole screen numbers apply to co-ords above
quad_indices = np.array([
    0,1,2,
    2,1,3
], dtype=np.int32)

# Vertex shader for texturing
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
out vec2 TexCoords;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoords = (aPos.xy + 1.0) / 2.0;  // map from [-1,1] to [0,1]
}
"""

# Fragment shader for texturing
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

#Funtion that compiles shaders
def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise Exception(f"Shader compile error: {error}")
    return shader

#Funtion to create program that applys shaders
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

#Function to open window
def main():
    if not glfw.init():
        raise Exception("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    width, height = 640, 480
    window = glfw.create_window(width, height, "CPU Ray Tracer Display", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Window creation failed")

    glfw.make_context_current(window)

    # Compile shader program
    shader_program = create_shader_program()

    # Setup VAO/VBO for quad
    VAO = glGenVertexArrays(1) #stores overall vertex states
    VBO = glGenBuffers(1) #stores actual vertex data
    EBO = glGenBuffers(1) #stores the indices that specify how to connect the vertices to triangles

    #Binding all of these
    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * quad_vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Generate ray traced image once
    image = simple_blackhole_raytrace(width, height)

    # Create OpenGL texture
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Upload image data to texture
    image = np.flipud(image)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, image)

    glBindTexture(GL_TEXTURE_2D, 0)

    # Render loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)

        glActiveTexture(GL_TEXTURE0)  # Activate texture unit 0
        glBindTexture(GL_TEXTURE_2D, texture)

        # Set uniform 'screenTexture' to use texture unit 0
        glUniform1i(glGetUniformLocation(shader_program, "screenTexture"), 0)

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    # Cleanup
    glfw.terminate()

if __name__ == '__main__':
    main()