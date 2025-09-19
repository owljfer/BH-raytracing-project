# BH Ray Tracer Project

A CPU-based ray tracer implementation showcasing 3D rendering techniques and mathematical algorithms, rendering an image of a blackhole.

## Current Features
- Ray-sphere and disk intersection
- Approximated geodesic Schwarzchild black hole visualization with accretion disk, photon sphere, doppler effect, redshift and gravitational lensing
- Proper mathematical treatment of rays close to singularity using intergration of the geodesic
- OpenGL display integration
- Pixel processing using numba
- Real-time rendering
- Interactive controls of camera and parameters

## Planned Features
- [ ] Advanced effects (Kerr black hole, multiple disk components)
- [ ] Performance optimizations
- [ ] GPU acceleration
- [ ] Visualise space-time warping

## Requirements
- Python 3.x
- PyOpenGL
- GLFW
- NumPy
- Numba

## Installation
```bash
pip install -r requirements.txt 
```

## Usage
```bash
python rt_blackhole_project.py 
```
This will open a window displaying the black hole visualisation, instructions for controls are printed in the terminal. Close the window to exit.

## Sample Output
![Black Hole Visualization](images/BH_raytrace_SS_V3.PNG)
