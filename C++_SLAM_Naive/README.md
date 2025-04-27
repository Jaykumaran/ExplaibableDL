Reference:


More Appropirate to call is Visual Odometry as *Loop Closure isn't yet integrated.
Currently uses Global BA instead of loop-closure.

https://github.com/gpdaniels/slam/



```
bash setup.sh
> mkdir build
> cd build
> cmake -DCMAKE_BUILD_TYPE=Release ..
> cmake --build . --config Release
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

##  ##