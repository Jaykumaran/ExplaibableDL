Reference:


More Appropirate to call is Visual Odometry as *Loop Closure isn't yet integrated.

https://github.com/gpdaniels/slam/




**Run Without Docker**
```sudo apt update
sudo apt install build-essential cmake git libeigen3-dev libopencv-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```

```rm -rf build # Clean previous docker build artifacts if any
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) # Or cmake --build . --config Release
```

**With Docker** 
**Run in an isolated Environment by interfacing the host display and docker env**
```
bash setup.sh 
> mkdir build
> cd build
> cmake -DCMAKE_BUILD_TYPE=Release ..
# Runs CMake to generate platform-specific build files (Makefiles, Visual Studio projects, etc.).

# -DCMAKE_BUILD_TYPE=Release tells CMake to set up an optimized release build (turning on compiler optimizations, turning off debug symbols).

# .. points CMake back to your project root (where your CMakeLists.txt lives), telling it “read that file to know what to build.”

> cmake --build . --config Release

# Tells CMake to actually compile and link your code now that the build files are in place.

# --build . means “build using the files generated in the current directory (build/).”

# --config Release (mainly relevant on multi-configuration generators like Visual Studio) says “use the Release configuration” that you just set up.
```



## Usage ##

After building in `./build/` it should be possible to run the system using these commands:

Sample data from https://vision.in.tum.de/data/datasets/rgbd-dataset/download

```
./slam "../videos/freiburgxyz_525.mp4" 525
```

Sample data from https://www.cvlibs.net/datasets/kitti/

```
./slam "../videos/kitti_984.mp4" 984
```

#### Results:

<img src = "readme_media/kitti_slam.gif">



<img src="https://learnopencv.com/wp-content/uploads/2024/06/Untitled-Diagram-Page-1-6.jpg">
