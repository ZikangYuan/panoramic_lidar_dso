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

#undef Success
#include <Eigen/Core>
#include "util/NumType.h"
#include <pangolin/pangolin.h>

#include <sstream>
#include <fstream>
// 2021.12.14
#include "util/settings.h"
// 2021.12.14

namespace dso
{
class CalibHessian;
class FrameHessian;
class FrameShell;

namespace IOWrap
{

template<int ppp>
struct InputPointSparse
{
	float u;
	float v;
	float idpeth;
	float idepth_hessian;
	float relObsBaseline;
	int numGoodRes;
	unsigned char color[ppp];
	unsigned char status;
};

struct MyVertex
{
	float point[3];
	unsigned char color[4];
};

// stores a pointcloud associated to a Keyframe.
class KeyFrameDisplay
{

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	KeyFrameDisplay();
	~KeyFrameDisplay();

	// copies points from KF over to internal buffer,
	// keeping some additional information so we can render it differently.
	void setFromKF(FrameHessian* fh, CalibHessian* HCalib);

	// copies points from KF over to internal buffer,
	// keeping some additional information so we can render it differently.
	void setFromF(FrameShell* fs, CalibHessian* HCalib);

	// copies & filters internal data to GL buffer for rendering. if nothing to do: does nothing.
	bool refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity);

	// renders cam & pointcloud.
	void drawCam(float lineWidth = 1, float* color = 0, float sizeFactor=1);
	void drawPC(float pointSize);

	int id;
	bool active;
	SE3 camToWorld; 
	// 2021.12.20
	SE3 camToWorld_gt;
	// 2021.12.20

    inline bool operator < (const KeyFrameDisplay& other) const
    {
        return (id < other.id);
    }


private:
	// 2021.12.13
	/*
	float fx,fy,cx,cy;
	float fxi,fyi,cxi,cyi;
	*/
	float fx[CAM_NUM],fy[CAM_NUM],cx[CAM_NUM],cy[CAM_NUM];
	float fxi[CAM_NUM],fyi[CAM_NUM],cxi[CAM_NUM],cyi[CAM_NUM];
	// 2021.12.13
	int width, height;

	float my_scaledTH, my_absTH, my_scale;
	int my_sparsifyFactor;
	int my_displayMode;
	float my_minRelBS;
	bool needRefresh;


	// 2021.12.14
	// int numSparsePoints;
	// int numSparseBufferSize;
    // InputPointSparse<MAX_RES_PER_POINT>* originalInputSparse;
	int numSparsePoints[1];
	int numSparseBufferSize[1];
    InputPointSparse<MAX_RES_PER_POINT>* originalInputSparse[1];
	// 2021.12.14


	bool bufferValid;
	int numGLBufferPoints;
	int numGLBufferGoodPoints;
	// 2021.12.14
	pangolin::GlBuffer vertexBuffer[1/*CAM_NUM*/];
	pangolin::GlBuffer colorBuffer[1];
	// 2021.12.14
};

}
}
