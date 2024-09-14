/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
 
#include "util/NumType.h"
// 2020.07.02 yzk
#include <queue>
#include "FullSystem/HessianBlocks.h"
// 2020.07.02 yzk

namespace dso
{

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};


class FrameHessian;
// 2022.1.10
class frame_hessian;
//

class PixelSelector
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// 2020.07.02
	int makeMaps(
		const frame_hessian* const fh,
		float* map_out, float density, int recursionsLeft=1, bool plot=false, float thFactor=1);
	// 2020.07.02

	int makeMapsFromLidar(const frame_hessian* const fh, float* map_out, float density, int recursionsLeft,
		bool plot, float thFactor, std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> &vCloudPixel, int id=0);
	Eigen::Matrix<int,5,1> simpleMakeMapsFromLidar(const FrameHessian* const Fh, float* map_out, float density, int recursionsLeft,
		bool plot, std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloudPixel[CAM_NUM], int id=0);

	PixelSelector(int w, int h);
	~PixelSelector();
	int currentPotential; 		//!< 当前选择像素点的潜力, 就是网格大小, 越大选点越少

	float currentTH;

	bool allowFast;
	void makeHists(const frame_hessian* const fh);

private:
	Eigen::Vector3i selectFromLidar(const frame_hessian* const fh, float* map_out, int pot, float thFactor, 
		std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> &vCloudPixel, int idx=0);
	Eigen::Matrix<int,5,1> simpleSelectFromLidar(const FrameHessian* const fh, float* map_out, float numWant,     
		std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloudPixel[CAM_NUM], int _idx=0);


	Eigen::Vector3i select(const frame_hessian* const fh,
			float* map_out, int pot, float thFactor=1);


	unsigned char* randomPattern;


	int* gradHist;  			//!< 根号梯度平方和分布直方图, 0是所有像素个数
	float* ths;					//!< 平滑之前的阈值
	float* thsSmoothed;			//!< 平滑后的阈值
	int thsStep;
	const frame_hessian* gradHistFrame;
};




}

