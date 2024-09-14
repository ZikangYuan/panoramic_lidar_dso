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
#include "vector"
#include <math.h>
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include <opencv2/video/tracking.hpp>
#include "util/lsd.h"
#include <random>
#include <omp.h>
#include <queue>




namespace dso
{
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class CoarseTracker {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseTracker(int w, int h);
	~CoarseTracker();

	bool trackNewestCoarse(
			FrameHessian* newFrameHessian,
			SE3 &lastToNew_out,	int coarsestLvl, 
			Vec5 minResForAbort, SE3 slastToLast, int tries,
			IOWrap::Output3DWrapper* wrap=0);

	bool trackNewestCoarse_inv(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out,	int coarsestLvl, 
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper* wrap=0);
	int nThreads;
	std::vector<std::mt19937> m_RandEngines; // Mersenne twister high quality RNG that support *OpenMP* multi-threading
	bool trackNewestCoarse_ransac(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out,	int coarsestLvl, 
		Vec5 minResForAbort, SE3 slastToLast,
		IOWrap::Output3DWrapper* wrap=0);

	bool TrackNewestToSelectCam(
			SE3 &lastToNew_out,	int coarsestLvl, 
			Vec5 minResForAbort, Vec6& lastRes, bool first = false);
	void recurrent(
			int num,int numUse,
			SE3 &lastToNew_out,	int coarsestLvl,
			Vec5 &minResForAbort);

	void setCoarseTrackingRef(
			std::vector<FrameHessian*> frameHessians,
			std::vector<int> idxUse);

	// 2019.11.15 yzk
	void makeCoarseDepthForFirstFrame(FrameHessian* fh);
	void setCTRefForFirstFrame(std::vector<FrameHessian *> frameHessians);
	// 2019.11.15 yzk

	void makeK(CalibHessian* HCalib);

	bool debugPrint, debugPlot;


	Mat33f K[CAM_NUM][PYR_LEVELS];
	Mat33f Ki[CAM_NUM][PYR_LEVELS];
	float fx[CAM_NUM][PYR_LEVELS];
	float fy[CAM_NUM][PYR_LEVELS];
	float fxi[CAM_NUM][PYR_LEVELS];
	float fyi[CAM_NUM][PYR_LEVELS];
	float cx[CAM_NUM][PYR_LEVELS];
	float cy[CAM_NUM][PYR_LEVELS];
	float cxi[CAM_NUM][PYR_LEVELS];
	float cyi[CAM_NUM][PYR_LEVELS];


	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

	std::vector<int> idxUse;
	bool weight[CAM_NUM];

    void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

	FrameHessian* lastRef;			//!< 参考帧
	// AffLight lastRef_aff_g2l;
	AffLight lastRef_aff_g2l[CAM_NUM];
	FrameHessian* newFrame;			//!< 新来的一帧
	int refFrameID;					//!< 参考帧id

	bool hasChanged;

	// act as pure ouptut
	Vec6 lastResiduals;
	Vec3 lastFlowIndicators;		//!< 光流指示用, 只有平移和, 旋转+平移的像素移动
	double firstCoarseRMSE;
private:


	void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians);

	float* idepth[CAM_NUM][PYR_LEVELS];
	float* weightSums[PYR_LEVELS];
	float* weightSums_bak[PYR_LEVELS];


	Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
	float onlyCalcRes(const SE3 &refToNew, float cutoffTH);
	Vec6 calcRes(int lvl, const SE3 &refToNew, SE3 slastToLast, float cutoffTH);
	Vec6 calcRes_ransac(int lvl, const SE3 &refToNew, SE3 slastToLast, float cutoffTH, std::vector<int>& idx_to_reproject);
	Vec6 calcRes_cross(int lvl, const SE3 &refToNew, SE3 slastToLast, float cutoffTH, std::queue<int> idx[][CAM_NUM]);
	void preCompute(int lvl);
	Vec6 calcResInv(int lvl, const SE3 &refToNew, float cutoffTH, std::queue<int>& resInliers, Mat66f &H, Vec6f &J_res);
	Vec3 calcExtrinRes(
		std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>>& cld,
		const SE3 &T_lb3_lidar, float cutoffTH, Mat66f& H, Vec6f& J_res, int itr);
	Vec3 calcExtrinRes(
		std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>>& cld,
		const SE3 &T_lb3_lidar, float cutoffTH);
	void calcGSSSE(int lvl, Mat66 &H_out, Vec6 &b_out, SE3 lastToNew, SE3 slastToLast);
	void calcGSSSE_inv(int lvl, Mat66f &H_out, Vec6f &b_out, std::queue<int>& idx);
	void calcGSSSE_MutualPSimple(int lvl, Mat66 &H_out, Vec6 &b_out, SE3 lastToNew, SE3 slastToLast);
	void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

	// pc buffers
	float* pc_u[CAM_NUM][PYR_LEVELS];				//!< 每层上的有逆深度点的坐标x
	float* pc_v[CAM_NUM][PYR_LEVELS];				//!< 每层上的有逆深度点的坐标y
	float* pc_idepth[CAM_NUM][PYR_LEVELS];			//!< 每层上点的逆深度
	float* pc_color[CAM_NUM][PYR_LEVELS];			//!< 每层上点的颜色值
	float* pc_Idu[CAM_NUM][PYR_LEVELS];				//!< 每层上点的颜色值在u方向上的梯度
	float* pc_Idv[CAM_NUM][PYR_LEVELS];				//!< 每层上点的颜色值在v方向上的梯度
	int pc_n[CAM_NUM][PYR_LEVELS];					//!< 每层上点的个数


	// warped buffers
	// float* buf_warped_X;					//!< 投影得到的点在主相机坐标系下的X
	// float* buf_warped_Y;					//!< 投影得到的点在主相机坐标系下的Y
	// float* buf_warped_Z;					//!< 投影得到的点在主相机坐标系下的Z
	// float* buf_warped_idepth;				//!< 投影得到的点的逆深度
	// float* buf_warped_u;					//!< 投影得到的归一化坐标
	// float* buf_warped_v;					//!< 同上
	// float* buf_warped_dx;					//!< 投影点的图像梯度
	// float* buf_warped_dy;					//!< 同上
	// float* buf_warped_residual;				//!< 投影得到的残差
	// float* buf_warped_weight;				//!< 投影的huber函数权重
	float* buf_warped_Xs[12];					//!< 投影得到的点在主相机坐标系下的X
	float* buf_warped_Ys[12];					//!< 投影得到的点在主相机坐标系下的Y
	float* buf_warped_Zs[12];					//!< 投影得到的点在主相机坐标系下的Z
	float* buf_warped_idepths[12];				//!< 投影得到的点的逆深度
	float* buf_warped_us[12];					//!< 投影得到的归一化坐标
	float* buf_warped_vs[12];					//!< 同上
	float* buf_warped_dxs[12];					//!< 投影点的图像梯度
	float* buf_warped_dys[12];					//!< 同上
	float* buf_warped_residuals[12];				//!< 投影得到的残差
	float* buf_warped_weights[12];				//!< 投影的huber函数权重
	int buf_warpeds[12][CAM_NUM*CAM_NUM];
	int buf_warped_n;						//!< 投影点的个数
	Vec6f* buf_dI_dT[PYR_LEVELS];
	Mat66f* buf_H[PYR_LEVELS];
	float* buf_res[PYR_LEVELS];
	float* buf_hw[PYR_LEVELS];


    std::vector<float*> ptrToDelete;				//!< 所有的申请的内存指针, 用于析构删除

	Accumulator acc;
	Accumulator7 acc7;
};


class CoarseDistanceMap {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseDistanceMap(int w, int h);
	~CoarseDistanceMap();

	void makeDistanceMap(
			std::vector<FrameHessian*> frameHessians,
			FrameHessian* frame, int cam_idx);

	void makeInlierVotes(
			std::vector<FrameHessian*> frameHessians);

	void makeK( CalibHessian* HCalib, int cam_idx);


	float* fwdWarpedIDDistFinal;		//!< 距离场的数值

	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

	void addIntoDistFinal(int u, int v);


private:

	PointFrameResidual** coarseProjectionGrid;	
	int* coarseProjectionGridNum;				
	Eigen::Vector2i* bfsList1;					//!< 投影到frame的坐标
	Eigen::Vector2i* bfsList2;					//!< 和1轮换使用

	void growDistBFS(int bfsNum);
};

}

