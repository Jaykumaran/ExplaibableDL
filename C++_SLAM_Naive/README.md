Reference:


More Appropirate to call is Visual Odometry as *Loop Closure isn't yet integrated.
Currently uses Global BA instead of loop-closure.

https://github.com/gpdaniels/slam/



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