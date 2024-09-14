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
#include "util/settings.h"
#include "util/NumType.h"
#include <vector>

#include <opencv2/highgui/highgui.hpp>

namespace dso
{
	extern int wG[PYR_LEVELS], hG[PYR_LEVELS];

	extern float powG[PYR_LEVELS];
	extern float weightC[CAM_NUM];
	extern bool firstG;
	extern float lastRes;
	extern float fxG[CAM_NUM][PYR_LEVELS], fyG[CAM_NUM][PYR_LEVELS],
		  cxG[CAM_NUM][PYR_LEVELS], cyG[CAM_NUM][PYR_LEVELS];

	extern float fxiG[CAM_NUM][PYR_LEVELS], fyiG[CAM_NUM][PYR_LEVELS],
		  cxiG[CAM_NUM][PYR_LEVELS], cyiG[CAM_NUM][PYR_LEVELS];

	extern Eigen::Matrix3f KG[CAM_NUM][PYR_LEVELS],KiG[CAM_NUM][PYR_LEVELS];
	// 2021.11.13

	extern float wM3G;
	extern float hM3G;

	extern Eigen::Matrix<unsigned char,3,1> LUT[256];

	extern std::vector<SE3> T_lb_c;
	extern std::vector<SE3> T_c0_c;
	extern std::vector<SE3> T_c_c0; 
	extern bool* maskG[CAM_NUM][PYR_LEVELS];

	extern int _left, _right, _top, _bottom;

	extern int ids[CAM_NUM+2];

	void setGlobalCalib(int w, int h, const Eigen::Matrix3f &K, int cam_idx, bool* mask);
	void draw(cv::Mat& m, int Ku, int Kv, cv::Vec3b color, int margin);

	Eigen::Matrix4d xyzrpy2T(float ssc[6]);
	void setGlobalExtrin(float ssc[6], int cam_idx);
	void setGlobalExtrin_ijrr(Eigen::Matrix4d T, int cam_idx);
	
	void calGlobalExtrin();
	void setLUT();
}
