# PL_VO

## 1.Description
The RGB-D vision odometry with point feature and line feature. 

## 2.Prerequisites

### OpenCV
Dowload and install instructions can be found at: http://opencv.org.

Tested with OpenCV 3.2.

### glog
```sh
download new version gflag and compile.

$ git clone https://github.com/gflags/gflags.git

$ mkdir build & cd build

$ cmake -DGFLAGS_NAMESPACE=google -DCMAKE_CXX_FLAGS=-fPIC ..

$ make & sudo make install

$ git clone https://github.com/google/glog.git

$ cd glog

$ mkdir build && cd build

maybe you need to install cmake3

$ sudo apt-get install cmake3

$ export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1

$ make

$ sudo make install
```
### Eigen
 Download and install instructions can be found at: http://eigen.tuxfamily.org.

 Tested with Eigen 3.2.1

### Ceres
Dowload and install instructions can be found at: http://www.ceres-solver.org/installation.html

### Sophus
```sh
$ git checkeout a621ff2
```

## 3.Building
```sh
$ mkdir build

$ cd build

$ make -j2
```

## 4.Test
1. **test_line_feature** \
line feature detection use the lsd original code and opencv code
2. **test_line_match** \
line feature detection and matching use the lsd and lbd algorithm in the opencv.
2. **test_PLfeature_optimization** \
the optimization of the point feature and line feature, and just test the two frames. Optimize
the pose of the MapPoint and MapLine and camera pose.
