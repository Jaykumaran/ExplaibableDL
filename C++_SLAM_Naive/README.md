Reference:


More Appropirate to call is Visual Odometry as *Loop Closure isn't yet integrated.

https://github.com/gpdaniels/slam/




### **Run Without Docker**
```sudo apt update
sudo apt install build-essential cmake git libeigen3-dev libopencv-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```

```
rm -rf build # Clean previous docker build artifacts if any
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) # Or cmake --build . --config Release
```

## **With Docker (Recommended)** 
**Run in an isolated Environment by interfacing the host display and docker env**
```
docker build -t monocular_slam .
```

### Build Executables

So, it will go inside the docker container,
```bash
> mkdir build
> cd build
> cmake -DCMAKE_BUILD_TYPE=Release ..
> cmake --build . --config Release
```

### Run Docker container
```
bash setup.sh 
```

Runs CMake to generate platform-specific build files (Makefiles, Visual Studio projects, etc.).

-DCMAKE_BUILD_TYPE=Release tells CMake to set up an optimized release build (turning on compiler optimizations, turning off debug symbols).

.. points CMake back to your project root (where your CMakeLists.txt lives), telling it “read that file to know what to build.”

Tells CMake to actually compile and link your code now that the build files are in place.

 --build . means “build using the files generated in the current directory (build/).”

 --config Release (mainly relevant on multi-configuration generators like Visual Studio) says “use the Release configuration” that you just set up.

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



---
### Docker Issues:

If you are facing

> Ubuntu 22.04 - "/dev/dri": no such file or directory

and not able to access your GPU or camera binding to the docker, the most likley culprit is Docker

If you're facing GPU/camera setup issues in Docker on Linux (/dev/video0, /dev/dri errors), you're likely using Docker Desktop.

❌ It's a VM — can't access host hardware directly.
✅ Fix: uninstall Docker Desktop, install native Docker Engine + NVIDIA toolkit.

For more details, check out my discussion here:  [Issue#165](https://github.com/linuxserver/docker-jellyfin/issues/165#issuecomment-2948731223)


Errors:

```docker: Error response from daemon: error gathering device information while adding custom device "/dev/video0": no such file or directory.

docker: Error response from daemon: error gathering device information while adding custom device "/dev/dri": no such file or directory
```


Few debug tests:
1. Check for your graphics devices (GPU):
`ls -l /dev/dri`
Output will look something like this, proving it exists:
crw-rw----+ 1 root render 226, 128 Jun 6 15:09 renderD128
crw-rw----+ 1 root render 226, 129 Jun 6 15:09 renderD129

2. Check for your webcams:
` ls -l /dev/video* `
crw-rw----+ 1 root video 81, 0 Jun 6 15:09 /dev/video0
crw-rw----+ 1 root video 81, 1 Jun 6 15:09 /dev/video1
