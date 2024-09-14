# Panoramic-LDSO
A panoramic direct LiDAR-assisted visual odometry.

## Demo Video (2024-09-14 Update)

The **x16 Real-Time Performance (Left)** and **Final Trjaectory and Sparse Map (Right)** on the segment of sequence *2012-01-08* from [*NCLT*](http://robots.engin.umich.edu/nclt/) dataset.

<div align="left">
<img src="doc/run_outdoor.gif" width=48% />
<img src="doc/final_trajectory.png" width=49.6% />
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

## Run on Public Datasets

Noted: Currently the package only supports interfaces to *NCLT* and *IJRR* datasets. If you want to run on other datasets, you'll need to modify the code yourself.

###  1. Run on [*NCLT*](http://robots.engin.umich.edu/nclt/)

Before running, please ensure the dataset format is as follow:

```bash
<PATH_OF_NCLT_FOLDER>
	|____________2012-01-08
			  |____________lb3
			  |____________velodyne_sync
	|____________2012-09-28
			  |____________lb3
			  |____________velodyne_sync
	|____________2012-11-04
			  |____________lb3
			  |____________velodyne_sync
	|____________2012-12-01
			  |____________lb3
			  |____________velodyne_sync
	|____________2013-02-23
			  |____________lb3
			  |____________velodyne_sync
	|____________2013-04-05
			  |____________lb3
			  |____________velodyne_sync
```

Then open the terminal in the path of the <PATH_OF_PROJECT_FOLDER>/build, and type:

```bash
./dso_dataset dataset=<PATH_OF_NCLT_FOLDER> sequence=<SEQUENCE_NAME> seg=<SEGMENT_NUMBER> calib=<PATH_OF_PROJECT_FOLDER>/calib/nclt/calib undistort=<PATH_OF_PROJECT_FOLDER>/calib/nclt/U2D_Cam pathSensorPrameter<PATH_OF_PROJECT_FOLDER>/sensor/nclt/x_lb3_c resultPath=<PATH_OF_PROJECT_FOLDER>/output/pose.txt mode=1 quiet=0 IJRR=0
```
