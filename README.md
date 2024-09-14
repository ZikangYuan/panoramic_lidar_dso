# panoramic_lidar_dso
A panoramic direct LiDAR-assisted visual odometry.

## Demo Video (2024-09-14 Update)

The **x16 Real-Time Performance** on the sequence *nclt_2012_01_08* of self-collected dataset from [**NCLT**](http://robots.engin.umich.edu/nclt/).

<div align="left">
<img src="doc/run_outdoor.gif" width=52.11% />
</div>

## Installation

### 1. Requirements

> GCC >= 7.5.0
>
> Cmake >= 3.16.0
> 
> [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.3.4
>
> [OpenCV](https://github.com/opencv/opencv) >= 3.3
>
> [PCL](https://pointclouds.org/downloads/) == 1.8 for Ubuntu 18.04, and == 1.10 for Ubuntu 20.04
> 
> [Pangolin](https://github.com/stevenlovegrove/Pangolin) == 0.5 or 0.6 for Ubuntu 20.04

##### Have Tested On:

| OS    | GCC  | Cmake | Eigen3 | OpenCV | PCL | Pangolin |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Ubuntu 20.04 | 9.4.0  | 3.16.3 | 3.3.7 | 4.2.0 | 1.10.0 | 0.5 |

### 2. Clone the directory and build

```bash
git clone https://github.com/ZikangYuan/panoramic_lidar_dso.git
mkdir build
cd build
cmake ..
make
```
