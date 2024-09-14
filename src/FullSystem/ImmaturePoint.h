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
 
#include "FullSystem/HessianBlocks.h"
namespace dso
{

// ImmaturePoint的状态, 残差和目标帧
struct ImmaturePointTemporaryResidual
{
public:
	ResState state_state; 		//!< 逆深度残差的状态
	double state_energy;		//!< 残差值
	ResState state_NewState;	//!< 新计算的逆深度残差的状态
	double state_NewEnergy;		//!< 新计算的残差值
	frame_hessian* target;
};


enum ImmaturePointStatus {
	IPS_GOOD=0,					// traced well and good
	// 搜索区间超出图像, 尺度变化太大, 两次残差都大于阈值, 不再搜索
	IPS_OOB,					// OOB: end tracking & marginalize!
	// 第一次残差大于阈值, 外点
	IPS_OUTLIER,				// energy too high: if happens again: outlier!
	// 搜索区间太短，但是没激活
	IPS_SKIPPED,				// traced well and good (but not actually traced).
	// 梯度和极线夹角太大
	IPS_BADCONDITION,			// not traced because of bad condition.
	IPS_UNINITIALIZED};			// not even traced once.


class ImmaturePoint
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	float color[MAX_RES_PER_POINT];		//!< 原图上pattern上对应的像素值
	float weights[MAX_RES_PER_POINT];	//!< 原图上pattern对应的权重(与梯度成反比)

    // 2019.12.04 yzk
    float idepth_fromSensor;
    // 2019.12.04 yzk

	static const int halfpatch_size_ = 4;
	static const int patch_size_ = 8;

	Mat22f gradH;				//!< 图像梯度hessian矩阵
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH;
	float u,v;					//!< host里的像素坐标
	frame_hessian* host; // 2022.1.11
	int idxInImmaturePoints;

	float quality;				//!< 第二误差/第一误差 作为搜索质量, 越大越好

	float my_type;

	float idepth_min;			//!< 逆深度范围
	float idepth_max;

	// 2019.12.02 yzk
	bool isFromSensor;
	// 2019.12.02 yzk

	ImmaturePoint(int u_, int v_, frame_hessian* host_, float type, CalibHessian* HCalib);
	~ImmaturePoint();

	ImmaturePointStatus traceOn(frame_hessian* frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine, CalibHessian* HCalib, bool debugPrint=false);

	ImmaturePointStatus lastTraceStatus;		//!< 上一次跟踪状态
	Vec2f lastTraceUV;							//!< 上一次搜索得到的位置
	float lastTracePixelInterval;				//!< 上一次的搜索范围长度

	float idepth_GT;

	double linearizeResidual(
	 		CalibHessian *  HCalib, const float outlierTHSlack,
	 		ImmaturePointTemporaryResidual* tmpRes,
	 		float &Hdd, float &bd,
	 		float idepth);
	void resetPosition(float u_, float v_);
	double linearizeResPixel(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			cv::Mat &Hdd, cv::Mat &bd, int level_ref,
			float idepth);
	double calcResPixel(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth, float res);
	double ResPixel(
			const float outlierTHSlack,
			FrameHessian* target, float u, float v,
			cv::Mat &Hdd, cv::Mat &bd, int level,
			float mean_diff, float idepth);
	void getWarpMatrixAffine(
			const double idepth,
			const Mat33f &KRKi, const Vec3f &Kt,
			Mat22f& A_cur_ref);
	int getBestSearchLevel(
			const Mat22f& A_cur_ref,
			const int max_level);
	float getdPixdd(
			CalibHessian *  HCalib,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

	float calcResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

private:
};
}

