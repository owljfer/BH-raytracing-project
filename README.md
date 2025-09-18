# BH Ray Tracer Project

A CPU-based ray tracer implementation showcasing 3D rendering techniques and mathematical algorithms, rendering an image of a blackhole with enhanced 3D accretion disk.

## Current Features
- Ray-sphere intersection
- Advanced geodesic Schwarzchild black hole visualization with enhanced 3D accretion disk
- **NEW: 3D Accretion Disk Rendering** with realistic relativistic effects
- **NEW: Disk rotation and orbital mechanics** 
- **NEW: Enhanced Doppler shifts and gravitational redshift**
- **NEW: Temperature gradients with azimuthal variations**
- **NEW: Spiral patterns and magnetic field effects**
- OpenGL display integration
- Pixel processing using numba
- Real-time rendering
- Interactive controls of camera and parameters

## Enhanced 3D Disk Features
- **Realistic Orbital Mechanics**: Disk rotation based on Kepler's laws
- **Relativistic Effects**: Advanced Doppler shift, gravitational redshift, and beaming
- **3D Geometry**: Proper height profiles, warping, and structural variations
- **Temperature Mapping**: Radial and azimuthal temperature gradients
- **Visual Enhancements**: Spiral patterns, turbulence, and magnetic field effects

## Controls
- **W/S**: Move camera closer/farther
- **Q/E**: Decrease/Increase disk inner radius
- **R/T**: Increase/Decrease disk outer radius  
- **G/H**: Decrease/Increase disk height
- **F/V**: Decrease/Increase disk rotation speed *(NEW)*
- **C/B**: Rotate disk azimuthal angle *(NEW)*
- **Z/X**: Decrease/Increase field of view
- **-/=**: Decrease/Increase anti-aliasing samples
- **1**: Toggle stars on/off
- **2**: Toggle disk on/off
- **SPACE**: Reset view to default
- **P**: Print current parameters
- **Mouse Drag**: Rotate camera
- **Mouse Scroll**: Zoom in/out

## Planned Features
- [ ] Add proper geodesic solving, handling complex orbits and multiple disk crosses
- [ ] Advanced effects (lensing, Kerr black hole, multiple disk components)
- [ ] Performance optimizations
- [ ] GPU acceleration

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
This will open a window displaying the black hole visualization with enhanced 3D accretion disk. Close the window to exit.

## 3D Disk Enhancement Details
See [3D_DISK_ENHANCEMENT.md](3D_DISK_ENHANCEMENT.md) for detailed documentation of the new 3D accretion disk rendering features.

## Sample Output
![Black Hole Visualization](images/BH_raytrace_SS_V2.PNG)

*The enhanced version now features realistic 3D accretion disk with rotation, relativistic effects, and temperature gradients.*
