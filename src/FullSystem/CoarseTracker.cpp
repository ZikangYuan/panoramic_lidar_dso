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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include <algorithm>

// 2019.12.09 yzk
#include "iostream"
#include "fstream"

using namespace std;
// 2019.12.09 yzk

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

//! 生成2^b个字节对齐
template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T)); //? 为什么加上这个值  答: 为了对齐,下面会移动b
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((uintptr_t)(ptr+padT)) >> b) << b);  //! 左移右移之后就会按照2的b次幂字节对齐, 丢掉不对齐的
    return alignedPtr;
}

//@ 构造函数, 申请内存, 初始化
CoarseTracker::CoarseTracker(int ww, int hh) /*: lastRef_aff_g2l(0,0)*/
{
	nThreads = std::max(1, omp_get_max_threads());
	std::cout << "[ INFO ]: Maximum usable threads: " << nThreads << std::endl;
	for (int i = 0; i < nThreads; ++i)
	{
		std::random_device SeedDevice;
		m_RandEngines.push_back(std::mt19937(SeedDevice()));
	}

	for(int i=0;i<cam_num;i++){
		lastRef_aff_g2l[i] = AffLight(0,0);
	}
	// make coarse tracking templates.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = ww>>lvl;
        int hl = hh>>lvl;

		weightSums[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
		weightSums_bak[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
		for(int idx=0;idx<cam_num;idx++)
		{
			idepth[idx][lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
			pc_u[idx][lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
			pc_v[idx][lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
			pc_idepth[idx][lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
			pc_color[idx][lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

			pc_Idu[idx][lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
			pc_Idv[idx][lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
		}
		buf_dI_dT[lvl] = new Vec6f[cam_num*wl*hl];
		buf_H[lvl] = new Mat66f[cam_num*wl*hl];
		buf_res[lvl] = new float[cam_num*wl*hl];
		buf_hw[lvl] = new float[cam_num*wl*hl];
	}
	for(int i=0;i<12;i++)
	{
	buf_warped_Xs[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete); 
	buf_warped_Ys[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete); 
	buf_warped_Zs[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete); 

    buf_warped_idepths[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete);
    buf_warped_us[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete);
    buf_warped_vs[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete);
    buf_warped_dxs[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete);
    buf_warped_dys[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete);
    buf_warped_residuals[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete);
    buf_warped_weights[i] = allocAligned<4,float>(cam_num*ww*hh, ptrToDelete);
	}

	newFrame = 0;
	lastRef = 0;
	debugPlot = debugPrint = true;
	w[0]=h[0]=0;
	refFrameID=-1;
}
CoarseTracker::~CoarseTracker()
{
    for(float* ptr : ptrToDelete)
        delete[] ptr;
    ptrToDelete.clear();
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		delete buf_dI_dT[lvl];
		delete buf_H[lvl];
		delete buf_res[lvl];
		delete buf_hw[lvl];
	}
}

void CoarseTracker::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	for(int i=0;i<cam_num;i++){
		fx[i][0] = HCalib->fxl(i);
		fy[i][0] = HCalib->fyl(i);
		cx[i][0] = HCalib->cxl(i);
		cy[i][0] = HCalib->cyl(i);

		for (int level = 1; level < pyrLevelsUsed; ++ level)
		{
			if(i==0){
				w[level] = w[0] >> level;
				h[level] = h[0] >> level;
			}
			fx[i][level] = fx[i][level-1] * 0.5;
			fy[i][level] = fy[i][level-1] * 0.5;
			cx[i][level] = (cx[i][0] + 0.5) / ((int)1<<level) - 0.5;
			cy[i][level] = (cy[i][0] + 0.5) / ((int)1<<level) - 0.5;
		}

		for (int level = 0; level < pyrLevelsUsed; ++ level)
		{
			K[i][level]  << fx[i][level], 0.0, cx[i][level], 0.0, fy[i][level], cy[i][level], 0.0, 0.0, 1.0;
			Ki[i][level] = K[i][level].inverse();
			fxi[i][level] = Ki[i][level](0,0);
			fyi[i][level] = Ki[i][level](1,1);
			cxi[i][level] = Ki[i][level](0,2);
			cyi[i][level] = Ki[i][level](1,2);
		}
	}
}

void CoarseTracker::makeCoarseDepthForFirstFrame(FrameHessian* fh)
{
    // make coarse tracking templates for latstRef.
	for(int idx=0;idx<cam_num;idx++){
		memset(idepth[idx][0], 0, sizeof(float)*w[0]*h[0]);
		memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);

		for(PointHessian* ph : fh->frame[idx]->pointHessians)
		{
			int u = ph->u + 0.5f;
			int v = ph->v + 0.5f;
			float new_idepth = ph->idepth;
			float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));

			idepth[idx][0][u+w[0]*v] += new_idepth *weight;
			weightSums[0][u+w[0]*v] += weight;

		}

		for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
		{
			int lvlm1 = lvl-1;
			int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

			float* idepth_l = idepth[idx][lvl];
			float* weightSums_l = weightSums[lvl];

			float* idepth_lm = idepth[idx][lvlm1];
			float* weightSums_lm = weightSums[lvlm1];

			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					int bidx = 2*x   + 2*y*wlm1;
					idepth_l[x + y*wl] = 		idepth_lm[bidx] +
												idepth_lm[bidx+1] +
												idepth_lm[bidx+wlm1] +
												idepth_lm[bidx+wlm1+1];

					weightSums_l[x + y*wl] = 	weightSums_lm[bidx] +
												weightSums_lm[bidx+1] +
												weightSums_lm[bidx+wlm1] +
												weightSums_lm[bidx+wlm1+1];
				}
		}

		// dilate idepth by 1.
		for(int lvl=0; lvl<2; lvl++)
		{
			int numIts = 1;


			for(int it=0;it<numIts;it++)
			{
				int wh = w[lvl]*h[lvl]-w[lvl];
				int wl = w[lvl];
				float* weightSumsl = weightSums[lvl];
				float* weightSumsl_bak = weightSums_bak[lvl];
				memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
				float* idepthl = idepth[idx][lvl];	// dont need to make a temp copy of depth, since I only
				// read values with weightSumsl>0, and write ones with weightSumsl<=0.
				for(int i=w[lvl];i<wh;i++)
				{
					if(weightSumsl_bak[i] <= 0)
					{
						float sum=0, num=0, numn=0;
						if(weightSumsl_bak[i+1+wl] > 0) { sum += idepthl[i+1+wl]; num+=weightSumsl_bak[i+1+wl]; numn++;}
						if(weightSumsl_bak[i-1-wl] > 0) { sum += idepthl[i-1-wl]; num+=weightSumsl_bak[i-1-wl]; numn++;}
						if(weightSumsl_bak[i+wl-1] > 0) { sum += idepthl[i+wl-1]; num+=weightSumsl_bak[i+wl-1]; numn++;}
						if(weightSumsl_bak[i-wl+1] > 0) { sum += idepthl[i-wl+1]; num+=weightSumsl_bak[i-wl+1]; numn++;}
						if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
					}
				}
			}
		}


		// dilate idepth by 1 (2 on lower levels).
		for(int lvl=2; lvl<pyrLevelsUsed; lvl++)
		{
			int wh = w[lvl]*h[lvl]-w[lvl];
			int wl = w[lvl];
			float* weightSumsl = weightSums[lvl];
			float* weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
			float* idepthl = idepth[idx][lvl];	// dotnt need to make a temp copy of depth, since I only
			// read values with weightSumsl>0, and write ones with weightSumsl<=0.
			for(int i=w[lvl];i<wh;i++)
			{
				if(weightSumsl_bak[i] <= 0)
				{
					float sum=0, num=0, numn=0;
					if(weightSumsl_bak[i+1] > 0) { sum += idepthl[i+1]; num+=weightSumsl_bak[i+1]; numn++;}
					if(weightSumsl_bak[i-1] > 0) { sum += idepthl[i-1]; num+=weightSumsl_bak[i-1]; numn++;}
					if(weightSumsl_bak[i+wl] > 0) { sum += idepthl[i+wl]; num+=weightSumsl_bak[i+wl]; numn++;}
					if(weightSumsl_bak[i-wl] > 0) { sum += idepthl[i-wl]; num+=weightSumsl_bak[i-wl]; numn++;}
					if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
				}
			}
		}


		// normalize idepths and weights.
		for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
		{
			float* weightSumsl = weightSums[lvl];
			float* idepthl = idepth[idx][lvl];
			Eigen::Vector3f* dIRefl = lastRef->frame[idx]->dIp[lvl];

			int wl = w[lvl], hl = h[lvl];

			int lpc_n=0;
			float* lpc_u = pc_u[idx][lvl];
			float* lpc_v = pc_v[idx][lvl];
			float* lpc_idepth = pc_idepth[idx][lvl];
			float* lpc_color = pc_color[idx][lvl];
			float* lpc_Idu = pc_Idu[idx][lvl];
			float* lpc_Idv = pc_Idv[idx][lvl];


			for(int y=2;y<hl-2;y++)
				for(int x=2;x<wl-2;x++)
				{
					int i = x+y*wl;

					if(weightSumsl[i] > 0)
					{
						idepthl[i] /= weightSumsl[i];
						lpc_u[lpc_n] = x;
						lpc_v[lpc_n] = y;
						lpc_idepth[lpc_n] = idepthl[i];
						lpc_color[lpc_n] = dIRefl[i][0];
						lpc_Idu[lpc_n] = dIRefl[i][1];
						lpc_Idv[lpc_n] = dIRefl[i][2];



						if(!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i]>0))
						{
							idepthl[i] = -1;
							continue;	// just skip if something is wrong.
						}
						lpc_n++;
					}
					else
						idepthl[i] = -1;

					weightSumsl[i] = 1;
				}

			pc_n[idx][lvl] = lpc_n;
		}
	}

}



//@ 使用在当前帧上投影的点的逆深度, 来生成每个金字塔层上点的逆深度值
void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians)
{
	// make coarse tracking templates for latstRef.
	// XTL:遍历所有关键帧，对其上的关键点的最新的残差项进行判断，根据这个设置idepth，便于可视化
	for(int idx=0;idx<cam_num;idx++)
	{
		memset(idepth[idx][0], 0, sizeof(float)*w[0]*h[0]);
		memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);
	//[ ***step 1*** ] 计算其它点在最新帧投影第0层上的各个像素的逆深度权重, 和加权逆深度
		if(setting_useMultualPBack)
		{
			for(FrameHessian* fh : frameHessians)
			{
				if(fh->w[idx]==false)
					continue;
				if(fh == frameHessians.back())
				{
					for(PointHessian* ph : fh->frame[idx]->pointHessians)
					{ 
						if(ph->isFromSensor == true)
						{
							int u = ph->u;
							int v = ph->v;
							float new_idepth = ph->idepth;
							float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12)); // 协方差逆做权重

							idepth[idx][0][u+w[0]*v] += new_idepth *weight; // 加权后的
							weightSums[0][u+w[0]*v] += weight;
						}
					}
				}
				else
				{
					for(frame_hessian* f : fh->frame)
						for(PointHessian* ph : f->pointHessians)
						{
							if(ph->isFromSensor == false)
								continue;
							if(ph->lastResiduals[2*idx].first != 0 && ph->lastResiduals[2*idx].second == ResState::IN)
							{
								PointFrameResidual* r = ph->lastResiduals[2*idx].first;

								assert(r->efResidual->isActive() && r->target == lastRef->frame[idx]); // 点的残差是好的, 且target是这次的ref
                                
								int u = r->centerProjectedTo[0] + 0.5f;  // 四舍五入
								int v = r->centerProjectedTo[1] + 0.5f;
								float new_idepth = r->centerProjectedTo[2];
								float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12)); // 协方差逆做权重

								idepth[idx][0][u+w[0]*v] += new_idepth *weight; // 加权后的
								weightSums[0][u+w[0]*v] += weight;
							}
						}
				}
			}
		}
		else
		{
			for(FrameHessian* fh : frameHessians)
			{
				if(fh->w[idx]==false)
					continue;
				for(PointHessian* ph : fh->frame[idx]->pointHessians)
				{
					// XTL 点从最新帧上取得，且是雷达点
					// 2020.07.18 yzk shiyong
					if(fh == frameHessians.back() && ph->isFromSensor == true)
					{
						int u = ph->u;
						int v = ph->v;
						float new_idepth = ph->idepth;
						float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12)); // 协方差逆做权重

						idepth[idx][0][u+w[0]*v] += new_idepth *weight; // 加权后的
						weightSums[0][u+w[0]*v] += weight;
					}
					// 2020.07.18 yzk shiyong
					// XTL 点在最新帧上构建了残差，且残差的状态是好的
					else
					{
						if(ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN)
						{
							if(fh == frameHessians.back() /*|| ph->isFromSensor == false*/)// 2021.12.27
								continue;

							PointFrameResidual* r = ph->lastResiduals[0].first;

							assert(r->efResidual->isActive() && r->target == lastRef->frame[idx]); // 点的残差是好的, 且target是这次的ref
							int u = r->centerProjectedTo[0] + 0.5f;  // 四舍五入
							int v = r->centerProjectedTo[1] + 0.5f;
							float new_idepth = r->centerProjectedTo[2];
							float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12)); // 协方差逆做权重

							idepth[idx][0][u+w[0]*v] += new_idepth *weight; // 加权后的
							weightSums[0][u+w[0]*v] += weight;
						}
					}
				}
			}
		}

	//[ ***step 2*** ] 从下层向上层生成逆深度和权重
		for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
		{
			int lvlm1 = lvl-1;
			int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

			float* idepth_l = idepth[idx][lvl];
			float* weightSums_l = weightSums[lvl];

			float* idepth_lm = idepth[idx][lvlm1];
			float* weightSums_lm = weightSums[lvlm1];

			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					int bidx = 2*x   + 2*y*wlm1;

					idepth_l[x + y*wl] = 		idepth_lm[bidx] +
												idepth_lm[bidx+1] +
												idepth_lm[bidx+wlm1] +
												idepth_lm[bidx+wlm1+1];

					weightSums_l[x + y*wl] = 	weightSums_lm[bidx] +
												weightSums_lm[bidx+1] +
												weightSums_lm[bidx+wlm1] +
												weightSums_lm[bidx+wlm1+1];
				}
		}

	//[ ***step 3*** ] 0和1层 对于没有深度的像素点, 使用周围斜45度的四个点来填充
		// dilate idepth by 1.
		// 2020.07.18 yzk shiyong
		for(int lvl=0; lvl<2; lvl++)
		{
			int numIts = 1;


			for(int it=0;it<numIts;it++)
			{
				int wh = w[lvl]*h[lvl]-w[lvl]; // 空出一行
				int wl = w[lvl];
				float* weightSumsl = weightSums[lvl];
				float* weightSumsl_bak = weightSums_bak[lvl];
				memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float)); // 备份
				float* idepthl = idepth[idx][lvl];	// dotnt need to make a temp copy of depth, since I only
												// read values with weightSumsl>0, and write ones with weightSumsl<=0.
				for(int i=w[lvl];i<wh;i++) // 上下各空一行
				{
					if(weightSumsl_bak[i] <= 0)
					{
						// 使用四个角上的点来填充没有深度的
						//bug: 对于竖直边缘上的点不太好把, 使用上两行的来计算
						float sum=0, num=0, numn=0;
						if(weightSumsl_bak[i+1+wl] > 0) { sum += idepthl[i+1+wl]; num+=weightSumsl_bak[i+1+wl]; numn++;}
						if(weightSumsl_bak[i-1-wl] > 0) { sum += idepthl[i-1-wl]; num+=weightSumsl_bak[i-1-wl]; numn++;}
						if(weightSumsl_bak[i+wl-1] > 0) { sum += idepthl[i+wl-1]; num+=weightSumsl_bak[i+wl-1]; numn++;}
						if(weightSumsl_bak[i-wl+1] > 0) { sum += idepthl[i-wl+1]; num+=weightSumsl_bak[i-wl+1]; numn++;}
						if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
					}
				}
			}
		}

	//[ ***step 4*** ] 2层向上, 对于没有深度的像素点, 使用上下左右的四个点来填充
		// dilate idepth by 1 (2 on lower levels).
		for(int lvl=2; lvl<pyrLevelsUsed; lvl++)
		{
			int wh = w[lvl]*h[lvl]-w[lvl];
			int wl = w[lvl];
			float* weightSumsl = weightSums[lvl];
			float* weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
			float* idepthl = idepth[idx][lvl];	// dotnt need to make a temp copy of depth, since I only
											// read values with weightSumsl>0, and write ones with weightSumsl<=0.
			for(int i=w[lvl];i<wh;i++)
			{
				if(weightSumsl_bak[i] <= 0)
				{
					float sum=0, num=0, numn=0;
					if(weightSumsl_bak[i+1] > 0) { sum += idepthl[i+1]; num+=weightSumsl_bak[i+1]; numn++;}
					if(weightSumsl_bak[i-1] > 0) { sum += idepthl[i-1]; num+=weightSumsl_bak[i-1]; numn++;}
					if(weightSumsl_bak[i+wl] > 0) { sum += idepthl[i+wl]; num+=weightSumsl_bak[i+wl]; numn++;}
					if(weightSumsl_bak[i-wl] > 0) { sum += idepthl[i-wl]; num+=weightSumsl_bak[i-wl]; numn++;}
					if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
				}
			}
		}
		// 2020.07.18 yzk shiyong

	//[ ***step 5*** ] 归一化点的逆深度并赋值给成员变量pc_*
		// normalize idepths and weights.
		for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
		{
			float* weightSumsl = weightSums[lvl];
			float* idepthl = idepth[idx][lvl];
			Eigen::Vector3f* dIRefl = lastRef->frame[idx]->dIp[lvl];

			int wl = w[lvl], hl = h[lvl];

			int lpc_n=0;

			float* lpc_u = pc_u[idx][lvl]; 
			float* lpc_v = pc_v[idx][lvl];
			float* lpc_idepth = pc_idepth[idx][lvl];
			float* lpc_color = pc_color[idx][lvl];
			float* lpc_Idu = pc_Idu[idx][lvl];
			float* lpc_Idv = pc_Idv[idx][lvl];


			for(int y=2;y<hl-2;y++)
				for(int x=2;x<wl-2;x++)
				{
					int i = x+y*wl;

					if(weightSumsl[i] > 0) // 有值的
					{
						idepthl[i] /= weightSumsl[i];
						lpc_u[lpc_n] = x;
						lpc_v[lpc_n] = y;
						lpc_idepth[lpc_n] = idepthl[i];
						lpc_color[lpc_n] = dIRefl[i][0];
						lpc_Idu[lpc_n] = dIRefl[i][1];
						lpc_Idv[lpc_n] = dIRefl[i][2];



						if(!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i]>0))
						{
							idepthl[i] = -1;
							continue;	// just skip if something is wrong.
						}
						lpc_n++;
					}
					else
						idepthl[i] = -1;

					weightSumsl[i] = 1;  // 求完就变成1了
				}

			pc_n[idx][lvl] = lpc_n;
		}
	}

}


//@ 对跟踪的最新帧和参考帧之间的残差, 求 Hessian 和 b
void CoarseTracker::calcGSSSE(int lvl, Mat66 &H_out, Vec6 &b_out, SE3 lastToNew, SE3 slastToLast)
{
	acc7.initialize();

	__m128 one = _mm_set1_ps(1);
	__m128 minusOne = _mm_set1_ps(-1);
	__m128 zero = _mm_set1_ps(0);

	Mat33f R = lastToNew.rotationMatrix().cast<float>();
	float r = atan2(R(2,1),R(2,2))*RTOD;

	__m128 J[8];
	int lastn = 0;
	int thread_idx = omp_get_thread_num();
	float* buf_warped_X=buf_warped_Xs[thread_idx];
	float* buf_warped_Y=buf_warped_Ys[thread_idx];
	float* buf_warped_Z=buf_warped_Zs[thread_idx];
	float* buf_warped_dx=buf_warped_dxs[thread_idx];
	float* buf_warped_dy=buf_warped_dys[thread_idx];
	float* buf_warped_u=buf_warped_us[thread_idx];
	float* buf_warped_v=buf_warped_vs[thread_idx];
	float* buf_warped_idepth=buf_warped_idepths[thread_idx];
	float* buf_warped_residual=buf_warped_residuals[thread_idx];
	float* buf_warped_weight=buf_warped_weights[thread_idx];
	int* buf_warped = buf_warpeds[thread_idx];

	for(int idx=0;idx<cam_num;idx++)
	{
		if(weight[idx]==false)
			continue;
		__m128 fxl = _mm_set1_ps(fx[idx][lvl]);
		__m128 fyl = _mm_set1_ps(fy[idx][lvl]);
		int n;
		if(setting_resManyBlock)
			n = buf_warped[idx*cam_num+idx];
		else
			n = buf_warped[idx];
		//assert(n%4==0);
		Mat33f R = T_c_c0[idx].rotationMatrix().cast<float>();
		__m128 Rx0[9];
		for(int i=0;i<9;i++)
			Rx0[i] = _mm_set1_ps(R(i/3,i%3));
		for(int i=lastn;i<n;i+=4)
		{
			__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx+i), fxl); 	//! dx*fx
			__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy+i), fyl);	//! dy*fy
			__m128 u = _mm_load_ps(buf_warped_u+i);
			__m128 v = _mm_load_ps(buf_warped_v+i);
			__m128 id = _mm_load_ps(buf_warped_idepth+i);
			__m128 dxfxid = _mm_mul_ps(id,dx);
			__m128 dyfyid = _mm_mul_ps(id,dy);

			__m128 X = _mm_load_ps(buf_warped_X+i);
			__m128 Y = _mm_load_ps(buf_warped_Y+i);
			__m128 Z = _mm_load_ps(buf_warped_Z+i);

			for(int i=0;i<8;i++)
				J[i] = zero;
			__m128 P[3],rx,ry;
			for(int i=0;i<3;i++){
				rx = _mm_sub_ps(Rx0[i],_mm_mul_ps(u,Rx0[6+i]));
				ry = _mm_sub_ps(Rx0[3+i],_mm_mul_ps(v,Rx0[6+i]));
				J[i] = _mm_add_ps(_mm_mul_ps(rx,dxfxid),_mm_mul_ps(ry,dyfyid));
				if(i==0)
					for(int j = 0;j<3;j++)
						P[j] = _mm_sub_ps(_mm_mul_ps(Y,Rx0[2+3*j]),_mm_mul_ps(Z,Rx0[1+3*j]));
				else if(i==1)
					for(int j = 0;j<3;j++)
						P[j] = _mm_sub_ps(_mm_mul_ps(Z,Rx0[3*j]),_mm_mul_ps(X,Rx0[2+3*j]));
				else if(i==2)
					for(int j = 0;j<3;j++)
						P[j] = _mm_sub_ps(_mm_mul_ps(X,Rx0[1+3*j]),_mm_mul_ps(Y,Rx0[3*j]));
				rx = _mm_sub_ps(P[0],_mm_mul_ps(u,P[2]));
				ry = _mm_sub_ps(P[1],_mm_mul_ps(v,P[2]));
				J[i+3] = _mm_add_ps(_mm_mul_ps(rx,dxfxid),_mm_mul_ps(ry,dyfyid));
			}
			J[6] = _mm_load_ps(buf_warped_residual+i);
			J[7] = _mm_load_ps(buf_warped_weight+i);
			acc7.updateSSE_eighted(J);
		}
		lastn = n;
		if(setting_useMultualP&&setting_resManyBlock)
		{
			for(int _idx=0;_idx<cam_num;_idx++)
			{
				if(_idx==idx)
					continue;
				__m128 fxl = _mm_set1_ps(fx[_idx][lvl]);
				__m128 fyl = _mm_set1_ps(fy[_idx][lvl]);
				int n = buf_warped[idx*cam_num+_idx];
				Mat33f Rx0 = T_c0_c[_idx].rotationMatrix().transpose().cast<float>();
				__m128 R[3*3];
				for(int i=0;i<9;i++)
					R[i] = _mm_set1_ps(Rx0(i/3,i%3));
				for(int i=lastn;i<n;i+=4)
				{
					__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx+i), fxl); 	//! dx*fx
					__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy+i), fyl);	//! dy*fy
					__m128 u = _mm_load_ps(buf_warped_u+i);
					__m128 v = _mm_load_ps(buf_warped_v+i);
					__m128 id = _mm_load_ps(buf_warped_idepth+i);
					__m128 dxfxid = _mm_mul_ps(id,dx);
					__m128 dyfyid = _mm_mul_ps(id,dy);

					__m128 X = _mm_load_ps(buf_warped_X+i);
					__m128 Y = _mm_load_ps(buf_warped_Y+i);
					__m128 Z = _mm_load_ps(buf_warped_Z+i);

					for(int i=0;i<8;i++)
						J[i] = zero;
					
					__m128 P[3],rx,ry;
					for(int i=0;i<3;i++){
						rx = _mm_sub_ps(R[i],_mm_mul_ps(u,R[6+i]));
						ry = _mm_sub_ps(R[3+i],_mm_mul_ps(v,R[6+i]));
						J[i] = _mm_add_ps(_mm_mul_ps(rx,dxfxid),_mm_mul_ps(ry,dyfyid));
						if(i==0)
							for(int j = 0;j<3;j++)
								P[j] = _mm_sub_ps(_mm_mul_ps(Y,R[2+3*j]),_mm_mul_ps(Z,R[1+3*j]));
						else if(i==1)
							for(int j = 0;j<3;j++)
								P[j] = _mm_sub_ps(_mm_mul_ps(Z,R[3*j]),_mm_mul_ps(X,R[2+3*j]));
						else if(i==2)
							for(int j = 0;j<3;j++)
								P[j] = _mm_sub_ps(_mm_mul_ps(X,R[1+3*j]),_mm_mul_ps(Y,R[3*j]));
						rx = _mm_sub_ps(P[0],_mm_mul_ps(u,P[2]));
						ry = _mm_sub_ps(P[1],_mm_mul_ps(v,P[2]));
						J[i+3] = _mm_add_ps(_mm_mul_ps(rx,dxfxid),_mm_mul_ps(ry,dyfyid));
					}

					J[6] = _mm_load_ps(buf_warped_residual+i);
					J[7] = _mm_load_ps(buf_warped_weight+i);
					acc7.updateSSE_eighted(J);
				}
				lastn = n;
			}
		}
	}

	acc7.finish();

	H_out.topLeftCorner<6,6>() = acc7.H.topLeftCorner<6,6>().cast<double>() * (1.0f/buf_warped_n);
	b_out.head<6>() = acc7.H.topRightCorner<6,1>().cast<double>() * (1.0f/buf_warped_n);

	H_out.block<6,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<6,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<3,6>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,6>(3,0) *= SCALE_XI_TRANS;
	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
}

//  pre-compute
void CoarseTracker::preCompute(int lvl)
{
	int numTermsInWarped = 0;

	for(int idx=0;idx<cam_num;idx++)
	{
		float fxl = fx[idx][lvl];
		float fyl = fy[idx][lvl];
		float cxl = cx[idx][lvl];
		float cyl = cy[idx][lvl];

		Mat33f R = T_c0_c[idx].rotationMatrix().cast<float>();
		Vec3f t = T_c0_c[idx].translation().cast<float>();
		Mat33f R_c_c0 = T_c_c0[idx].rotationMatrix().cast<float>();

		//* 提取的点
		int nl = pc_n[idx][lvl];
		float* lpc_u = pc_u[idx][lvl];
		float* lpc_v = pc_v[idx][lvl];
		float* lpc_idepth = pc_idepth[idx][lvl];
		float* lpc_color = pc_color[idx][lvl]; 
		float* lpc_Idu = pc_Idu[idx][lvl];
		float* lpc_Idv = pc_Idv[idx][lvl];

		for(int i=0;i<nl;i++)
		{
			float id = lpc_idepth[i];
			float x = lpc_u[i];
			float y = lpc_v[i];

			// pre-compute
			Mat12f J_I_uv(lpc_Idu[i],lpc_Idv[i]);
			Mat23f J_uv_P; Mat36f J_P_T; Mat16f J_I_T;

			Vec3f pt_ref_cm = Ki[idx][lvl]*Vec3f(x, y, 1)/id;
			Vec3f pt_ref_c0 = R*pt_ref_cm+t;

			float z_inv = 1.0/pt_ref_cm[2];
			float z_inv_sq = z_inv*z_inv;
			J_uv_P(0,0) = fxl*z_inv;
			J_uv_P(0,1) = 0;
			J_uv_P(0,2) = -fxl*pt_ref_cm[0]*z_inv_sq;
			J_uv_P(1,0) = 0;
			J_uv_P(1,1) = fyl*z_inv;
			J_uv_P(1,2) = -fyl*pt_ref_cm[1]*z_inv_sq;

			J_P_T.block<3,3>(0,0) = R_c_c0;

			for(int j = 0;j<3;j++)
			{
				J_P_T(j,3) = pt_ref_c0(1,0)*R_c_c0(j,2)-pt_ref_c0(2,0)*R_c_c0(j,1);
				J_P_T(j,4) = pt_ref_c0(2,0)*R_c_c0(j,0)-pt_ref_c0(0,0)*R_c_c0(j,2);
				J_P_T(j,5) = pt_ref_c0(0,0)*R_c_c0(j,1)-pt_ref_c0(1,0)*R_c_c0(j,0);
			}

			J_I_T = J_I_uv*J_uv_P*J_P_T;

			buf_dI_dT[lvl][numTermsInWarped] = J_I_T.transpose();
			buf_H[lvl][numTermsInWarped] = buf_dI_dT[lvl][numTermsInWarped]*J_I_T;
			numTermsInWarped++;
		}
	}

	return ;
}

//@ 对跟踪的最新帧和参考帧之间的残差, 求 Hessian 和 b
void CoarseTracker::calcGSSSE_inv(int lvl, Mat66f &H_out, Vec6f &b_out, std::queue<int>& idx)
{
	H_out.setZero();
	b_out.setZero();
	while(!idx.empty())
	{
		int _idx = idx.front();
		idx.pop();
		b_out.noalias() += buf_dI_dT[lvl][_idx]*buf_res[lvl][_idx]*buf_hw[lvl][_idx];
		H_out.noalias() += buf_H[lvl][_idx]*buf_hw[lvl][_idx];
	}

	// H_out.topLeftCorner<6,6>() = acc7.H.topLeftCorner<6,6>().cast<double>() * (1.0f/buf_warped_n);
	// b_out.head<6>() = acc7.H.topRightCorner<6,1>().cast<double>() * (1.0f/buf_warped_n);

	// H_out.block<6,3>(0,0) *= SCALE_XI_ROT;
	// H_out.block<6,3>(0,3) *= SCALE_XI_TRANS;
	// H_out.block<3,6>(0,0) *= SCALE_XI_ROT;
	// H_out.block<3,6>(3,0) *= SCALE_XI_TRANS;

	// b_out.segment<3>(0) *= SCALE_XI_ROT;
	// b_out.segment<3>(3) *= SCALE_XI_TRANS;
}

Vec6 CoarseTracker::calcResInv(int lvl, const SE3 &refToNew, float cutoffTH, std::queue<int>& resInliers, Mat66f &H, Vec6f &J_res)
{
	std::queue<int> empty;
	resInliers.swap(empty);
	float ETotal = 0;
	int numTermsInETotal = 0;
	int numTermsInWarped = 0;
	int numSaturatedTotal=0;

	int wl = w[lvl];
	int hl = h[lvl];

	H.setZero();
	J_res.setZero();
	debugPlot = false;

	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	int sum = 0;

	for(int idx=0;idx<cam_num;idx++)
	{
		// if(idx!=0)	break;
		Eigen::Vector3f* dINewl = newFrame->frame[idx]->dIp[lvl];
		float fxl = fx[idx][lvl];
		float fyl = fy[idx][lvl];
		float cxl = cx[idx][lvl];
		float cyl = cy[idx][lvl];

		SE3 refToNew_tmp = T_c_c0[idx] * refToNew * T_c0_c[idx];

		// 经过huber函数后的能量阈值
		float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.
		
		Mat33f RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[idx][lvl]);
		Vec3f t = (refToNew_tmp.translation()).cast<float>();
	

		//* 提取的点
		int nl = pc_n[idx][lvl];
		float* lpc_u = pc_u[idx][lvl];
		float* lpc_v = pc_v[idx][lvl];
		float* lpc_idepth = pc_idepth[idx][lvl];
		float* lpc_color = pc_color[idx][lvl]; 


		std::queue<int> idx_to_reproject; 
		std::queue<int> idxJaco_to_reproject; 

		for(int i=0;i<nl;i++)
		{
			float id = lpc_idepth[i];
			float x = lpc_u[i];
			float y = lpc_v[i];
			
			//! 投影点
			Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
			float u = pt[0] / pt[2]; // 归一化坐标
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl; // 像素坐标
			float Kv = fyl * v + cyl;
			float new_idepth = id/pt[2]; // 当前帧上的深度

			if(lvl==0 && i%32==0)  //* 第0层 每隔32个点
			{
				//* 只正的平移 translation only (positive)
				Vec3f ptT = Ki[idx][lvl] * Vec3f(x, y, 1) + t*id;
				float uT = ptT[0] / ptT[2];
				float vT = ptT[1] / ptT[2];
				float KuT = fxl * uT + cxl;
				float KvT = fyl * vT + cyl;

				//* 只负的平移 translation only (negative)
				Vec3f ptT2 = Ki[idx][lvl] * Vec3f(x, y, 1) - t*id;
				float uT2 = ptT2[0] / ptT2[2];
				float vT2 = ptT2[1] / ptT2[2];
				float KuT2 = fxl * uT2 + cxl;
				float KvT2 = fyl * vT2 + cyl;

				//* 旋转+负的平移 translation and rotation (negative)
				Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
				float u3 = pt3[0] / pt3[2];
				float v3 = pt3[1] / pt3[2];
				float Ku3 = fxl * u3 + cxl;
				float Kv3 = fyl * v3 + cyl;

				//translation and rotation (positive)
				//already have it.
				
				//* 统计像素的移动大小
				sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
				sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
				sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
				sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
				sumSquaredShiftNum+=2;
			}

			if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[idx][lvl][(int)Ku+(int)Kv*wl]))
			{
				if(setting_useMultualP)
				{
					idx_to_reproject.push(i);
					idxJaco_to_reproject.push(numTermsInWarped);
				}
				numTermsInWarped++;
				continue;
			}

			// 计算残差
			float refColor = lpc_color[i];
			Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);  // 新帧上插值
			if(!std::isfinite((float)hitColor[0])) 
			{
				numTermsInWarped++;
				continue;
			}
			float residual = hitColor[0] - refColor;
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			numTermsInETotal++;		// ETotal 中数目

			if(fabs(residual) > cutoffTH)
			{
				ETotal += maxEnergy;		// 能量值
				numSaturatedTotal++;		// 大于阈值数目
			}
			else
			{
				sum++;
				ETotal += hw *residual*residual*(2-hw);
				buf_res[lvl][numTermsInWarped] = residual;
				buf_hw[lvl][numTermsInWarped] = hw;
				resInliers.push(numTermsInWarped);
				J_res.noalias() += buf_dI_dT[lvl][numTermsInWarped]*residual*hw;
				H.noalias() += buf_H[lvl][numTermsInWarped]*hw;
			}
			numTermsInWarped++;
		}
		if(setting_useMultualP)
		{
			for(int _idx=0;_idx<cam_num;_idx++)
			{
				if(_idx==idx)
					continue;

				Eigen::Vector3f* dINewl = newFrame->frame[_idx]->dIp[lvl];
				float fxl = fx[_idx][lvl];
				float fyl = fy[_idx][lvl];
				float cxl = cx[_idx][lvl];
				float cyl = cy[_idx][lvl];

				SE3 refToNew_tmp = T_c_c0[_idx] * refToNew * T_c0_c[idx];

				Mat33f RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[idx][lvl]);
				Vec3f t = (refToNew_tmp.translation()).cast<float>();

				for(int _i = 0; _i < idx_to_reproject.size(); _i ++)
				{
					int i = idx_to_reproject.front();
					idx_to_reproject.pop();
					int idxJaco = idxJaco_to_reproject.front();
					idxJaco_to_reproject.pop();

					float id = lpc_idepth[i];
					float x = lpc_u[i];
					float y = lpc_v[i];
					
					//! 投影点
					Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
					float u = pt[0] / pt[2]; // 归一化坐标
					float v = pt[1] / pt[2];
					float Ku = fxl * u + cxl; // 像素坐标
					float Kv = fyl * v + cyl;
					float new_idepth = id/pt[2]; // 当前帧上的深度
					
					if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[_idx][lvl][(int)Ku+(int)Kv*wl]))
					{
						idx_to_reproject.push(i);
						idxJaco_to_reproject.push(idxJaco);
						continue;
					}

					// 计算残差
					float refColor = lpc_color[i];
					Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);  // 新帧上插值
					if(!std::isfinite((float)hitColor[0])) continue;
					float residual = hitColor[0] - refColor;
					float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

					numTermsInETotal++;		// ETotal 中数目

					if(fabs(residual) > cutoffTH)
					{
						ETotal += maxEnergy;		// 能量值
						numSaturatedTotal++;		// 大于阈值数目
					}
					else
					{
						ETotal += hw *residual*residual*(2-hw);
						buf_res[lvl][idxJaco] = residual;
						buf_hw[lvl][idxJaco] = hw;
						resInliers.push(idxJaco);
					}
				}
			}
		}
	}


	// H = H * (1.0f/sum);
	// J_res = J_res * (1.0f/sum);

	// H.block<6,3>(0,0) *= SCALE_XI_ROT;
	// H.block<6,3>(0,3) *= SCALE_XI_TRANS;
	// H.block<3,6>(0,0) *= SCALE_XI_ROT;
	// H.block<3,6>(3,0) *= SCALE_XI_TRANS;

	// J_res.segment<3>(0) *= SCALE_XI_ROT;
	// J_res.segment<3>(3) *= SCALE_XI_TRANS;


	Vec6 rs;
	rs[0] = ETotal;												// 投影的能量值
	rs[1] = numTermsInETotal;									// 投影的点的数目
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);		// 纯平移时 平均像素移动的大小
	rs[3] = 0;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);		// 平移+旋转 平均像素移动大小                                                                   
	rs[5] = numSaturatedTotal / (float)numTermsInETotal;   			// 大于cutoff阈值的百分比

	// printf("Use %f points to track newest frame pose.\n",rs[3]);
	return rs;
}

//@ 计算当前位姿投影得到的残差(能量值), 并进行一些统计
//! 构造尽量多的点, 有助于跟踪
Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, SE3 slastToLast, float cutoffTH)
{
	float ETotal = 0;
	int numTermsInETotal = 0;
	int numTermsInWarped = 0;
	int numSaturatedTotal=0;

	int wl = w[lvl];
	int hl = h[lvl];

	debugPlot = false;

	bool debugsave = false;

	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	int sum = 0;

	int thread_idx = omp_get_thread_num();
	float* buf_warped_X=buf_warped_Xs[thread_idx];
	float* buf_warped_Y=buf_warped_Ys[thread_idx];
	float* buf_warped_Z=buf_warped_Zs[thread_idx];
	float* buf_warped_dx=buf_warped_dxs[thread_idx];
	float* buf_warped_dy=buf_warped_dys[thread_idx];
	float* buf_warped_u=buf_warped_us[thread_idx];
	float* buf_warped_v=buf_warped_vs[thread_idx];
	float* buf_warped_idepth=buf_warped_idepths[thread_idx];
	float* buf_warped_residual=buf_warped_residuals[thread_idx];
	float* buf_warped_weight=buf_warped_weights[thread_idx];
	int* buf_warped = buf_warpeds[thread_idx];
	cv::Mat imageFrame[cam_num];
	for(int idx=0;idx<cam_num;idx++)
	{
		if(!weight[idx]) continue;
		if(debugsave)
			imageFrame[idx] = newFrame->frame[idx]->getCvImages(lvl);
		// 经过huber函数后的能量阈值
		float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.

		//* 提取的点
		int nl = pc_n[idx][lvl];
		float* lpc_u = pc_u[idx][lvl];
		float* lpc_v = pc_v[idx][lvl];
		float* lpc_idepth = pc_idepth[idx][lvl];
		float* lpc_color = pc_color[idx][lvl]; 
		if(nl==0)
		{
			if(setting_resManyBlock)
				for(int _idx=0;_idx<cam_num;_idx++)
					buf_warped[cam_num*idx+_idx]=numTermsInWarped;
			else
				buf_warped[idx]=numTermsInWarped;
			continue;
		}

		float fxl = fx[idx][lvl];
		float fyl = fy[idx][lvl];
		float cxl = cx[idx][lvl];
		float cyl = cy[idx][lvl];
		Eigen::Vector3f* dINewl = newFrame->frame[idx]->dIp[lvl];
			
		SE3 refToNew_tmp = T_c_c0[idx] * refToNew * T_c0_c[idx];
		
		Mat33f RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[idx][lvl]);
		Vec3f t = (refToNew_tmp.translation()).cast<float>();

		sum+=nl;

		for(int i=0;i<nl;i++)
		{
			float id = lpc_idepth[i];
			float x = lpc_u[i];
			float y = lpc_v[i];
			
			//! 投影点
			Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
			float u = pt[0] / pt[2]; // 归一化坐标
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl; // 像素坐标
			float Kv = fyl * v + cyl;
			float new_idepth = id/pt[2]; // 当前帧上的深度

			if(lvl==0 && i%32==0)  //* 第0层 每隔32个点
			{
				//* 只正的平移 translation only (positive)
				Vec3f ptT = Ki[idx][lvl] * Vec3f(x, y, 1) + t*id;
				float uT = ptT[0] / ptT[2];
				float vT = ptT[1] / ptT[2];
				float KuT = fxl * uT + cxl;
				float KvT = fyl * vT + cyl;

				//* 只负的平移 translation only (negative)
				Vec3f ptT2 = Ki[idx][lvl] * Vec3f(x, y, 1) - t*id;
				float uT2 = ptT2[0] / ptT2[2];
				float vT2 = ptT2[1] / ptT2[2];
				float KuT2 = fxl * uT2 + cxl;
				float KvT2 = fyl * vT2 + cyl;

				//* 旋转+负的平移 translation and rotation (negative)
				Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
				float u3 = pt3[0] / pt3[2];
				float v3 = pt3[1] / pt3[2];
				float Ku3 = fxl * u3 + cxl;
				float Kv3 = fyl * v3 + cyl;

				//translation and rotation (positive)
				//already have it.
				
				//* 统计像素的移动大小
				sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
				sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
				sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
				sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
				sumSquaredShiftNum+=2;
			}

			if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[idx][lvl][(int)Ku+(int)Kv*wl]))
			{ 
				/*
				if(debugsave)
				{
					if(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3)
					{
						draw(imageFrame[idx],(int)Ku,(int)Kv,cv::Vec3b(0,0,255),1);
					}
				} 
				*/
				continue;
			}

			// 计算残差
			float refColor = lpc_color[i];
			Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);  // 新帧上插值
			if(!std::isfinite((float)hitColor[0])) continue;
			float residual = hitColor[0] - refColor;
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			if(debugsave)
			{
				Vec3b color = LUT[(int)(5*residual)+128];
				draw(imageFrame[idx],(int)Ku,(int)Kv,cv::Vec3b(0,255,0),1);
			}

			numTermsInETotal++;		// ETotal 中数目

			if(fabs(residual) > cutoffTH)
			{
				ETotal += maxEnergy;		// 能量值
				numSaturatedTotal++;		// 大于阈值数目
			}
			else
			{
				ETotal += hw *residual*residual*(2-hw);

				Mat33f R = T_c0_c[idx].rotationMatrix().cast<float>();
				Vec3f t = T_c0_c[idx].translation().cast<float>();
				Vec3f pt = R * Vec3f(u, v, 1)/new_idepth + t;
				
				buf_warped_X[numTermsInWarped] = pt[0]; //在主相机坐标系下X
				buf_warped_Y[numTermsInWarped] = pt[1]; //在主相机坐标系下Y
				buf_warped_Z[numTermsInWarped] = pt[2]; //在主相机坐标系下Z

				buf_warped_idepth[numTermsInWarped] = new_idepth; //逆深度
				buf_warped_u[numTermsInWarped] = u;	//归一化坐标
				buf_warped_v[numTermsInWarped] = v; //归一化坐标
				buf_warped_dx[numTermsInWarped] = hitColor[1];	//x方向梯度
				buf_warped_dy[numTermsInWarped] = hitColor[2];	//y方向梯度
				buf_warped_residual[numTermsInWarped] = residual;
				buf_warped_weight[numTermsInWarped] = hw;
				numTermsInWarped++;
			}
		}
		while(numTermsInWarped%4!=0) 
		{
			buf_warped_X[numTermsInWarped] = 0;
			buf_warped_Y[numTermsInWarped] = 0;
			buf_warped_Z[numTermsInWarped] = 0;

			buf_warped_idepth[numTermsInWarped] = 0;
			buf_warped_u[numTermsInWarped] = 0;
			buf_warped_v[numTermsInWarped] = 0;
			buf_warped_dx[numTermsInWarped] = 0;
			buf_warped_dy[numTermsInWarped] = 0;
			buf_warped_residual[numTermsInWarped] = 0;
			buf_warped_weight[numTermsInWarped] = 0;
			numTermsInWarped++;
		}
		if(setting_resManyBlock)
			for(int _idx=0;_idx<cam_num;_idx++)
				buf_warped[cam_num*idx+_idx]=numTermsInWarped;
		else
			buf_warped[idx]=numTermsInWarped;
		if(debugsave)
		{
			char str[300];
			sprintf(str, "/home/jeff/workspace/catkin_ws/src/omni_DSO_lidar/target/[%d]%d.png", 
				idx, newFrame->frame[idx]->shell->id);
			cv::imwrite(std::string(str), imageFrame[idx]);
			imageFrame[idx].release();
		}
	}
	buf_warped_n = numTermsInWarped;

	Vec6 rs;
	rs[0] = ETotal;												// 投影的能量值
	rs[1] = numTermsInETotal;									// 投影的点的数目
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);		// 纯平移时 平均像素移动的大小
	rs[3] = (float)numTermsInETotal/(float)sum;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);		// 平移+旋转 平均像素移动大小                                                                   
	rs[5] = numSaturatedTotal / (float)numTermsInETotal;   			// 大于cutoff阈值的百分比

	// std::cout<<"[calcRes]lvl:"<<lvl<<",res is"<<ETotal/(float)numTermsInETotal<<std::endl;
	// printf("Use %f points to track newest frame pose.\n",rs[3]);
	return rs;
}

float CoarseTracker::onlyCalcRes(const SE3 &refToNew, float cutoffTH)
{
	float ETotal = 0;
	int numTermsInETotal = 0;
	int numTermsInWarped = 0;
	int numSaturatedTotal=0;

	int wl = w[0];
	int hl = h[0];

	int sum = 0;
	
	for(int idx=0;idx<cam_num;idx++)
	{
		std::queue<int> idx_to_reproject;
		// 经过huber函数后的能量阈值
		float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.

		//* 提取的点
		int nl = pc_n[idx][0];
		float* lpc_u = pc_u[idx][0];
		float* lpc_v = pc_v[idx][0];
		float* lpc_idepth = pc_idepth[idx][0];
		float* lpc_color = pc_color[idx][0]; 
		if(nl==0)
		{
			continue;
		}

		float fxl = fx[idx][0];
		float fyl = fy[idx][0];
		float cxl = cx[idx][0];
		float cyl = cy[idx][0];
		Eigen::Vector3f* dINewl = newFrame->frame[idx]->dIp[0];
			
		SE3 refToNew_tmp = T_c_c0[idx] * refToNew * T_c0_c[idx];
		
		Mat33f RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[idx][0]);
		Vec3f t = (refToNew_tmp.translation()).cast<float>();

		sum+=nl;

		for(int i=0;i<nl;i++)
		{
			float id = lpc_idepth[i];
			float x = lpc_u[i];
			float y = lpc_v[i];
			
			//! 投影点
			Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
			float u = pt[0] / pt[2]; // 归一化坐标
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl; // 像素坐标
			float Kv = fyl * v + cyl;
			float new_idepth = id/pt[2]; // 当前帧上的深度

			if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[idx][0][(int)Ku+(int)Kv*wl]))
			{ 
				idx_to_reproject.push(i);
				continue;
			}

			// 计算残差
			float refColor = lpc_color[i];
			Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);  // 新帧上插值
			if(!std::isfinite((float)hitColor[0])) continue;
			float residual = hitColor[0] - refColor;
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			numTermsInETotal++;		// ETotal 中数目

			if(fabs(residual) > cutoffTH)
			{
				ETotal += maxEnergy;		// 能量值
				numSaturatedTotal++;		// 大于阈值数目
			}
			else
				ETotal += hw *residual*residual*(2-hw);
		}
		for(int _idx=0;_idx<cam_num;_idx++)
		{
			if(_idx==idx)
				continue;

			fxl = fx[_idx][0];
			fyl = fy[_idx][0];
			cxl = cx[_idx][0];
			cyl = cy[_idx][0];
			dINewl = newFrame->frame[_idx]->dIp[0];
			SE3	refToNew_tmp = T_c_c0[_idx] * refToNew * T_c0_c[idx];
			Mat33f RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[idx][0]);
			Vec3f t = (refToNew_tmp.translation()).cast<float>();

			int num = idx_to_reproject.size();
			for(int _i=0;_i<num;_i++)
			{
				int i = idx_to_reproject.front();
				idx_to_reproject.pop();

				float id = lpc_idepth[i];
				float x = lpc_u[i];
				float y = lpc_v[i];
				
				//! 投影点
				Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
				float u = pt[0] / pt[2]; // 归一化坐标
				float v = pt[1] / pt[2];
				float Ku = fxl * u + cxl; // 像素坐标
				float Kv = fyl * v + cyl;
				float new_idepth = id/pt[2]; // 当前帧上的深度

				if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[_idx][0][(int)Ku+(int)Kv*wl]))
				{	
					idx_to_reproject.push(i);	
					continue;	
				}

				// 计算残差
				float refColor = lpc_color[i];
				Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);  // 新帧上插值
				if(!std::isfinite((float)hitColor[0])) continue;
				float residual = (hitColor[0] - refColor);
				float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

				numTermsInETotal++;		// ETotal 中数目

				if(fabs(residual) > cutoffTH)
				{
					ETotal += maxEnergy;		// 能量值
					numSaturatedTotal++;		// 大于阈值数目
				}
				else
					ETotal += hw *residual*residual*(2-hw);
			}
		}
	}

	float rs;
	rs = ETotal/(float)numTermsInETotal;

	return rs;
}

Vec6 CoarseTracker::calcRes_ransac(int lvl, const SE3 &refToNew, SE3 slastToLast, float cutoffTH, std::vector<int>& reprojectVec)
{
	std::vector<int> empty;
	std::swap(reprojectVec, empty);

	float ETotal = 0;
	int numTermsInETotal = 0;
	int numTermsInWarped = 0;
	int numSaturatedTotal=0;

	int wl = w[lvl];
	int hl = h[lvl];

	debugPlot = false;

	bool debugsave = false;

	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	int sum = 0;

	int thread_idx = omp_get_thread_num();
	float* buf_warped_X=buf_warped_Xs[thread_idx];
	float* buf_warped_Y=buf_warped_Ys[thread_idx];
	float* buf_warped_Z=buf_warped_Zs[thread_idx];
	float* buf_warped_dx=buf_warped_dxs[thread_idx];
	float* buf_warped_dy=buf_warped_dys[thread_idx];
	float* buf_warped_u=buf_warped_us[thread_idx];
	float* buf_warped_v=buf_warped_vs[thread_idx];
	float* buf_warped_idepth=buf_warped_idepths[thread_idx];
	float* buf_warped_residual=buf_warped_residuals[thread_idx];
	float* buf_warped_weight=buf_warped_weights[thread_idx];
	int* buf_warped = buf_warpeds[thread_idx];
	cv::Mat imageFrame[cam_num];
	for(int idx=0;idx<cam_num;idx++)
	{
		if(!weight[idx]) continue;
		if(debugsave)
			imageFrame[idx] = newFrame->frame[idx]->getCvImages(lvl);
		// 经过huber函数后的能量阈值
		float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.

		//* 提取的点
		int nl = pc_n[idx][lvl];
		float* lpc_u = pc_u[idx][lvl];
		float* lpc_v = pc_v[idx][lvl];
		float* lpc_idepth = pc_idepth[idx][lvl];
		float* lpc_color = pc_color[idx][lvl]; 
		if(nl==0)
		{
			if(setting_resManyBlock)
				for(int _idx=0;_idx<cam_num;_idx++)
					buf_warped[cam_num*idx+_idx]=numTermsInWarped;
			else
				buf_warped[idx]=numTermsInWarped;
			continue;
		}

		float fxl = fx[idx][lvl];
		float fyl = fy[idx][lvl];
		float cxl = cx[idx][lvl];
		float cyl = cy[idx][lvl];
		Eigen::Vector3f* dINewl = newFrame->frame[idx]->dIp[lvl];
			
		SE3 refToNew_tmp = T_c_c0[idx] * refToNew * T_c0_c[idx];
		
		Mat33f RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[idx][lvl]);
		Vec3f t = (refToNew_tmp.translation()).cast<float>();

		for(int i=0;i<nl;i++)
		{
			float id = lpc_idepth[i];
			float x = lpc_u[i];
			float y = lpc_v[i];
			
			//! 投影点
			Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
			float u = pt[0] / pt[2]; // 归一化坐标
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl; // 像素坐标
			float Kv = fyl * v + cyl;
			float new_idepth = id/pt[2]; // 当前帧上的深度

			if(lvl==0 && i%32==0)  //* 第0层 每隔32个点
			{
				//* 只正的平移 translation only (positive)
				Vec3f ptT = Ki[idx][lvl] * Vec3f(x, y, 1) + t*id;
				float uT = ptT[0] / ptT[2];
				float vT = ptT[1] / ptT[2];
				float KuT = fxl * uT + cxl;
				float KvT = fyl * vT + cyl;

				//* 只负的平移 translation only (negative)
				Vec3f ptT2 = Ki[idx][lvl] * Vec3f(x, y, 1) - t*id;
				float uT2 = ptT2[0] / ptT2[2];
				float vT2 = ptT2[1] / ptT2[2];
				float KuT2 = fxl * uT2 + cxl;
				float KvT2 = fyl * vT2 + cyl;

				//* 旋转+负的平移 translation and rotation (negative)
				Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
				float u3 = pt3[0] / pt3[2];
				float v3 = pt3[1] / pt3[2];
				float Ku3 = fxl * u3 + cxl;
				float Kv3 = fyl * v3 + cyl;

				//translation and rotation (positive)
				//already have it.
				
				//* 统计像素的移动大小
				sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
				sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
				sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
				sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
				sumSquaredShiftNum+=2;
			}

			if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[idx][lvl][(int)Ku+(int)Kv*wl]))
			{
				reprojectVec.push_back(i+sum);
				continue;
			}

			// 计算残差
			float refColor = lpc_color[i];
			Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);  // 新帧上插值
			if(!std::isfinite((float)hitColor[0])) continue;
			float residual = hitColor[0] - refColor;
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			numTermsInETotal++;		// ETotal 中数目

			if(fabs(residual) > cutoffTH)
			{
				ETotal += maxEnergy;		// 能量值
				numSaturatedTotal++;		// 大于阈值数目
			}
			else
			{
				ETotal += hw *residual*residual*(2-hw);

				Mat33f R = T_c0_c[idx].rotationMatrix().cast<float>();
				Vec3f t = T_c0_c[idx].translation().cast<float>();
				Vec3f pt = R * Vec3f(u, v, 1)/new_idepth + t;
				
				buf_warped_X[numTermsInWarped] = pt[0]; //在主相机坐标系下X
				buf_warped_Y[numTermsInWarped] = pt[1]; //在主相机坐标系下Y
				buf_warped_Z[numTermsInWarped] = pt[2]; //在主相机坐标系下Z

				buf_warped_idepth[numTermsInWarped] = new_idepth; //逆深度
				buf_warped_u[numTermsInWarped] = u;	//归一化坐标
				buf_warped_v[numTermsInWarped] = v; //归一化坐标
				buf_warped_dx[numTermsInWarped] = hitColor[1];	//x方向梯度
				buf_warped_dy[numTermsInWarped] = hitColor[2];	//y方向梯度
				buf_warped_residual[numTermsInWarped] = residual;
				buf_warped_weight[numTermsInWarped] = hw;
				numTermsInWarped++;
			}
		}
		while(numTermsInWarped%4!=0) 
		{
			buf_warped_X[numTermsInWarped] = 0;
			buf_warped_Y[numTermsInWarped] = 0;
			buf_warped_Z[numTermsInWarped] = 0;

			buf_warped_idepth[numTermsInWarped] = 0;
			buf_warped_u[numTermsInWarped] = 0;
			buf_warped_v[numTermsInWarped] = 0;
			buf_warped_dx[numTermsInWarped] = 0;
			buf_warped_dy[numTermsInWarped] = 0;
			buf_warped_residual[numTermsInWarped] = 0;
			buf_warped_weight[numTermsInWarped] = 0;
			numTermsInWarped++;
		}
		if(setting_resManyBlock)
			for(int _idx=0;_idx<cam_num;_idx++)
				buf_warped[cam_num*idx+_idx]=numTermsInWarped;
		else
			buf_warped[idx]=numTermsInWarped;

		sum+=nl;
		if(debugsave)
		{
			char str[300];
			sprintf(str, "/home/jeff/workspace/catkin_ws/src/omni_DSO_lidar/target/[%d]%d.png", 
				idx, newFrame->frame[idx]->shell->id);
			cv::imwrite(std::string(str), imageFrame[idx]);
			imageFrame[idx].release();
		}
	}
	buf_warped_n = numTermsInWarped;

	Vec6 rs;
	rs[0] = ETotal;												// 投影的能量值
	rs[1] = numTermsInETotal;									// 投影的点的数目
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);		// 纯平移时 平均像素移动的大小
	rs[3] = (float)numTermsInETotal/(float)sum;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);		// 平移+旋转 平均像素移动大小                                                                   
	rs[5] = numSaturatedTotal / (float)numTermsInETotal;   			// 大于cutoff阈值的百分比

	return rs;
}

Vec6 CoarseTracker::calcRes_cross(int lvl, const SE3 &refToNew, SE3 slastToLast, float cutoffTH, std::queue<int> idx_to_reproject[][CAM_NUM])
{
	float ETotal = 0;
	int numTermsInETotal = 0;
	int numTermsInWarped = 0;
	int numSaturatedTotal=0;

	int wl = w[lvl];
	int hl = h[lvl];


	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	int sum = 0;
	int thread_idx = omp_get_thread_num();
	float* buf_warped_X=buf_warped_Xs[thread_idx];
	float* buf_warped_Y=buf_warped_Ys[thread_idx];
	float* buf_warped_Z=buf_warped_Zs[thread_idx];
	float* buf_warped_dx=buf_warped_dxs[thread_idx];
	float* buf_warped_dy=buf_warped_dys[thread_idx];
	float* buf_warped_u=buf_warped_us[thread_idx];
	float* buf_warped_v=buf_warped_vs[thread_idx];
	float* buf_warped_idepth=buf_warped_idepths[thread_idx];
	float* buf_warped_residual=buf_warped_residuals[thread_idx];
	float* buf_warped_weight=buf_warped_weights[thread_idx];
	int* buf_warped = buf_warpeds[thread_idx];
	for(int idx=0;idx<cam_num;idx++)
	{
		// 经过huber函数后的能量阈值
		float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.

		//* 提取的点
		int nl = pc_n[idx][lvl];
		float* lpc_u = pc_u[idx][lvl];
		float* lpc_v = pc_v[idx][lvl];
		float* lpc_idepth = pc_idepth[idx][lvl];
		float* lpc_color = pc_color[idx][lvl]; 
		if(nl==0)
		{
			if(!setting_resManyBlock)
				buf_warped[idx]=numTermsInWarped;
			else
				for(int _idx=0;_idx<cam_num;_idx++)
					buf_warped[cam_num*idx+_idx]=numTermsInWarped;
			continue;
		}

		float fxl = fx[idx][lvl];
		float fyl = fy[idx][lvl];
		float cxl = cx[idx][lvl];
		float cyl = cy[idx][lvl];
		Eigen::Vector3f* dINewl = newFrame->frame[idx]->dIp[lvl];
			
		SE3 refToNew_tmp = T_c_c0[idx] * refToNew * T_c0_c[idx];
		
		Mat33f RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[idx][lvl]);
		Vec3f t = (refToNew_tmp.translation()).cast<float>();

		for(int i=0;i<nl;i++)
		{
			float id = lpc_idepth[i];
			float x = lpc_u[i];
			float y = lpc_v[i];
			
			//! 投影点
			Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
			float u = pt[0] / pt[2]; // 归一化坐标
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl; // 像素坐标
			float Kv = fyl * v + cyl;
			float new_idepth = id/pt[2]; // 当前帧上的深度

			if(lvl==0 && i%32==0)  //* 第0层 每隔32个点
			{
				//* 只正的平移 translation only (positive)
				Vec3f ptT = Ki[idx][lvl] * Vec3f(x, y, 1) + t*id;
				float uT = ptT[0] / ptT[2];
				float vT = ptT[1] / ptT[2];
				float KuT = fxl * uT + cxl;
				float KvT = fyl * vT + cyl;

				//* 只负的平移 translation only (negative)
				Vec3f ptT2 = Ki[idx][lvl] * Vec3f(x, y, 1) - t*id;
				float uT2 = ptT2[0] / ptT2[2];
				float vT2 = ptT2[1] / ptT2[2];
				float KuT2 = fxl * uT2 + cxl;
				float KvT2 = fyl * vT2 + cyl;

				//* 旋转+负的平移 translation and rotation (negative)
				Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
				float u3 = pt3[0] / pt3[2];
				float v3 = pt3[1] / pt3[2];
				float Ku3 = fxl * u3 + cxl;
				float Kv3 = fyl * v3 + cyl;

				//translation and rotation (positive)
				//already have it.
				
				//* 统计像素的移动大小
				sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
				sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
				sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
				sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
				sumSquaredShiftNum+=2;
			}

			if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[idx][lvl][(int)Ku+(int)Kv*wl]))
				continue;

			// 计算残差
			float refColor = lpc_color[i];
			Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);  // 新帧上插值
			if(!std::isfinite((float)hitColor[0])) continue;
			float residual = hitColor[0] - refColor;
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			numTermsInETotal++;		// ETotal 中数目

			if(fabs(residual) > cutoffTH)
			{
				ETotal += maxEnergy;		// 能量值
				numSaturatedTotal++;		// 大于阈值数目
			}
			else
			{
				ETotal += hw *residual*residual*(2-hw);

				Mat33f R = T_c0_c[idx].rotationMatrix().cast<float>();
				Vec3f t = T_c0_c[idx].translation().cast<float>();
				Vec3f pt = R * Vec3f(u, v, 1)/new_idepth + t;
				
				buf_warped_X[numTermsInWarped] = pt[0]; //在主相机坐标系下X
				buf_warped_Y[numTermsInWarped] = pt[1]; //在主相机坐标系下Y
				buf_warped_Z[numTermsInWarped] = pt[2]; //在主相机坐标系下Z

				buf_warped_idepth[numTermsInWarped] = new_idepth; //逆深度
				buf_warped_u[numTermsInWarped] = u;	//归一化坐标
				buf_warped_v[numTermsInWarped] = v; //归一化坐标
				buf_warped_dx[numTermsInWarped] = hitColor[1];	//x方向梯度
				buf_warped_dy[numTermsInWarped] = hitColor[2];	//y方向梯度
				buf_warped_residual[numTermsInWarped] = residual;
				buf_warped_weight[numTermsInWarped] = hw;
				numTermsInWarped++;
			}
		}
		if(setting_resManyBlock)
		{
			while(numTermsInWarped%4!=0) 
			{
				buf_warped_X[numTermsInWarped] = 0;
				buf_warped_Y[numTermsInWarped] = 0;
				buf_warped_Z[numTermsInWarped] = 0;

				buf_warped_idepth[numTermsInWarped] = 0;
				buf_warped_u[numTermsInWarped] = 0;
				buf_warped_v[numTermsInWarped] = 0;
				buf_warped_dx[numTermsInWarped] = 0;
				buf_warped_dy[numTermsInWarped] = 0;
				buf_warped_residual[numTermsInWarped] = 0;
				buf_warped_weight[numTermsInWarped] = 0;
				numTermsInWarped++;
			}
			buf_warped[cam_num*idx+idx]=numTermsInWarped;
		}

		for(int _idx=0;_idx<cam_num;_idx++)
		{
			if(_idx==idx)
				continue;
			std::queue<int>& tmp=idx_to_reproject[lvl][idx];
			SE3 refToNew_tmp;	Mat33f RKi;
			if(setting_resManyBlock)
			{
				fxl = fx[_idx][lvl];
				fyl = fy[_idx][lvl];
				cxl = cx[_idx][lvl];
				cyl = cy[_idx][lvl];
				dINewl = newFrame->frame[_idx]->dIp[lvl];
				tmp = idx_to_reproject[lvl][idx];
				refToNew_tmp = T_c_c0[_idx] * refToNew * T_c0_c[idx];
				RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[idx][lvl]);
			}
			else
			{
				//* 提取的点
				lpc_u = pc_u[_idx][lvl];
				lpc_v = pc_v[_idx][lvl];
				lpc_idepth = pc_idepth[_idx][lvl];
				lpc_color = pc_color[_idx][lvl]; 
				tmp = idx_to_reproject[lvl][_idx];
				refToNew_tmp = T_c_c0[idx] * refToNew * T_c0_c[_idx];
				RKi = (refToNew_tmp.rotationMatrix().cast<float>() * Ki[_idx][lvl]);
			}
			Vec3f t = (refToNew_tmp.translation()).cast<float>();

			int num = tmp.size();
			for(int _i=0;_i<num;_i++)
			{
				int i = tmp.front();
				tmp.pop();

				float id = lpc_idepth[i];
				float x = lpc_u[i];
				float y = lpc_v[i];
				
				//! 投影点
				Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
				float u = pt[0] / pt[2]; // 归一化坐标
				float v = pt[1] / pt[2];
				float Ku = fxl * u + cxl; // 像素坐标
				float Kv = fyl * v + cyl;
				float new_idepth = id/pt[2]; // 当前帧上的深度

				if(setting_resManyBlock)
				{
					if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[_idx][lvl][(int)Ku+(int)Kv*wl]))
					{	
						tmp.push(i);	
						continue;	
					}
				}
				else
				{
					if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0 && maskG[idx][lvl][(int)Ku+(int)Kv*wl]))
					{
						tmp.push(i);
						continue;
					}
				}

				// 计算残差
				float refColor = lpc_color[i];
				Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);  // 新帧上插值
				if(!std::isfinite((float)hitColor[0])) continue;
				float residual = (hitColor[0] - refColor);
				float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

				numTermsInETotal++;		// ETotal 中数目

				if(fabs(residual) > cutoffTH)
				{
					ETotal += maxEnergy;		// 能量值
					numSaturatedTotal++;		// 大于阈值数目
				}
				else
				{
					ETotal += hw *residual*residual*(2-hw);

					Mat33f R;	Vec3f t;
					if(setting_resManyBlock)
					{
						R = T_c0_c[_idx].rotationMatrix().cast<float>();
						t = T_c0_c[_idx].translation().cast<float>();
					}
					else
					{
						R = T_c0_c[idx].rotationMatrix().cast<float>();
						t = T_c0_c[idx].translation().cast<float>();
					}
					Vec3f pt = R * Vec3f(u, v, 1)/new_idepth + t;
					
					buf_warped_X[numTermsInWarped] = pt[0]; //在主相机坐标系下X
					buf_warped_Y[numTermsInWarped] = pt[1]; //在主相机坐标系下Y
					buf_warped_Z[numTermsInWarped] = pt[2]; //在主相机坐标系下Z

					buf_warped_idepth[numTermsInWarped] = new_idepth; //逆深度
					buf_warped_u[numTermsInWarped] = u;	//归一化坐标
					buf_warped_v[numTermsInWarped] = v; //归一化坐标
					buf_warped_dx[numTermsInWarped] = hitColor[1];	//x方向梯度
					buf_warped_dy[numTermsInWarped] = hitColor[2];	//y方向梯度
					buf_warped_residual[numTermsInWarped] = residual;
					buf_warped_weight[numTermsInWarped] = hw;
					numTermsInWarped++;
				}
			}
			if(setting_resManyBlock)
			{
				while(numTermsInWarped%4!=0) 
				{
					buf_warped_X[numTermsInWarped] = 0;
					buf_warped_Y[numTermsInWarped] = 0;
					buf_warped_Z[numTermsInWarped] = 0;

					buf_warped_idepth[numTermsInWarped] = 0;
					buf_warped_u[numTermsInWarped] = 0;
					buf_warped_v[numTermsInWarped] = 0;
					buf_warped_dx[numTermsInWarped] = 0;
					buf_warped_dy[numTermsInWarped] = 0;
					buf_warped_residual[numTermsInWarped] = 0;
					buf_warped_weight[numTermsInWarped] = 0;
					numTermsInWarped++;
				}
				buf_warped[cam_num*idx+_idx]=numTermsInWarped;
			}
		}
		if(!setting_resManyBlock)
		{
			while(numTermsInWarped%4!=0) 
			{
				buf_warped_X[numTermsInWarped] = 0;
				buf_warped_Y[numTermsInWarped] = 0;
				buf_warped_Z[numTermsInWarped] = 0;

				buf_warped_idepth[numTermsInWarped] = 0;
				buf_warped_u[numTermsInWarped] = 0;
				buf_warped_v[numTermsInWarped] = 0;
				buf_warped_dx[numTermsInWarped] = 0;
				buf_warped_dy[numTermsInWarped] = 0;
				buf_warped_residual[numTermsInWarped] = 0;
				buf_warped_weight[numTermsInWarped] = 0;
				numTermsInWarped++;
			}
			buf_warped[idx]=numTermsInWarped;
		}
		// std::cout<<"After:tmp size"<<tmp.size()<<std::endl;
	}
	buf_warped_n = numTermsInWarped;

	Vec6 rs;
	rs[0] = ETotal;												// 投影的能量值
	rs[1] = numTermsInETotal;									// 投影的点的数目
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);		// 纯平移时 平均像素移动的大小
	rs[3] = (float)numTermsInETotal/(float)sum;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);		// 平移+旋转 平均像素移动大小                                                                   
	rs[5] = numSaturatedTotal / (float)numTermsInETotal;   			// 大于cutoff阈值的百分比

	return rs;
}

void CoarseTracker::setCTRefForFirstFrame(std::vector<FrameHessian *> frameHessians)
{
    assert(frameHessians.size()>0);
    lastRef = frameHessians.back();

	hasChanged = true;

	memset(weight,false,sizeof(bool)*cam_num);
	for(int i:idx_use)
		weight[i] = 1;

    makeCoarseDepthForFirstFrame(lastRef);

    refFrameID = lastRef->shell->id;

	for(int i=0;i<cam_num;i++)
    	lastRef_aff_g2l[i] = lastRef->aff_g2l(i);

    firstCoarseRMSE=-1;
}


//@ 把优化完的最新帧设为参考帧
void CoarseTracker::setCoarseTrackingRef(
		std::vector<FrameHessian*> frameHessians,
		std::vector<int> idxUse)
{
	assert(frameHessians.size()>0);

	lastRef = frameHessians.back();

	hasChanged = true;
	
	memset(weight,false,sizeof(bool)*cam_num);
	// this->idxUse = idx_use;
	for(int i:idx_use)
		weight[i] = 1;

	makeCoarseDepthL0(frameHessians);  // 生成逆深度估值

	refFrameID = lastRef->shell->id;

	for(int i=0;i<cam_num;i++)
		lastRef_aff_g2l[i] = lastRef->aff_g2l(i);

	firstCoarseRMSE=-1;

}


// XTL：用给的位姿把新帧点投影至参考帧上，构建残差项，计算导数，优化位姿和光度参数
bool CoarseTracker::trackNewestCoarse(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out,	int coarsestLvl, 
		Vec5 minResForAbort, SE3 slastToLast, int tries,
		IOWrap::Output3DWrapper* wrap)
{
	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);

	newFrame = newFrameHessian;

	int maxIterations[] = {10,20,50,50,50};  	// 不同层迭代的次数
	
	float lambdaExtrapolationLimit = 0.001;

	SE3 refToNew_current = lastToNew_out;		// 优化的初始值

	bool haveRepeated = false;

	// ProjectForVisualization(refToNew_current);
	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
		Mat66 H = Mat66::Zero(); Vec6 b = Vec6::Zero();
		float levelCutoffRepeat=1;
//[ ***step 1*** ] 计算残差, 保证最多60%残差大于阈值, 计算正规方程
		Vec6 resOld = calcRes(lvl, refToNew_current, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat);
		
		//* 保证residual大于阈值的点小于60%
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50) 
		{
			levelCutoffRepeat*=2;		// residual超过阈值的多, 则放大阈值重新计算
			resOld = calcRes(lvl, refToNew_current, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat);

			if(!setting_debugout_runquiet)
				printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}

		calcGSSSE(lvl, H, b, refToNew_current, slastToLast);

		float lambda = 0.01;

		if(debugPrint)
		{	
			printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					0.0f,
					resOld[0] / resOld[1],
					0,(int)resOld[1],
					0.0f);
			std::cout << refToNew_current.log().transpose() << ")\n";
		}

//[ ***step 2*** ] 迭代优化
		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			//[ ***step 2.1*** ] 计算增量
			Mat66 Hl = H;

			for(int i=0;i<6;i++) Hl(i,i) *= (1+lambda);
			Vec6 inc = Hl.ldlt().solve(-b);

			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec6 incScaled = inc;
			// XTL 这里写的有问题，se3平移在前，旋转在后 TODO
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;

			if(!std::isfinite(incScaled.sum())) incScaled.setZero();
			//[ ***step 2.2*** ] 使用增量更新后, 重新计算能量值
			SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
			
			Vec6 resNew = calcRes(lvl, refToNew_new, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat);

			
			bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);  // 平均能量值小则接受

			if(debugPrint)
			{
				printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << ")\n";
			}
			//[ ***step 2.3*** ] 接受则求正规方程, 继续迭代, 优化到增量足够小
			if(accept)
			{
				calcGSSSE(lvl, H, b, refToNew_new, slastToLast);
				resOld = resNew;
				refToNew_current = refToNew_new;
				lambda *= 0.5;
			}
			else
			{
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
			}
			
			if(!(inc.norm() > 1e-3))
			{
				if(debugPrint)
					printf("inc too small, break!\n");
				break;
			}
		}
//[ ***step 3*** ] 记录上一次残差, 光流指示, 如果调整过阈值则重新计算这一层
		// set last residual for that level, as well as flow indicators.
		lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));  // 上一次的平均能量
		lastFlowIndicators = resOld.segment<3>(2);

		std::cout<<"lvl="<<lvl<<",Outlier ratio="<<resOld[5]<<std::endl;

		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]){
			return false;  //! 如果算出来大于最好的直接放弃
		}

		// XTL 代表这层的残差大部分超过了阈值，因此得重新算一遍这层
		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL!\n");
		}
	}
	// set!
	lastToNew_out = refToNew_current;

	return true;
}

bool CoarseTracker::trackNewestCoarse_inv(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out,	int coarsestLvl, 
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper* wrap)
{
	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);

	newFrame = newFrameHessian;

	int maxIterations[] = {10,20,50,50,50};  	// 不同层迭代的次数
	
	float lambdaExtrapolationLimit = 0.001;

	SE3 refToNew_current = lastToNew_out;		// 优化的初始值

	bool haveRepeated = false;


	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
		std::queue<int> resInliers;
		Mat66f H = Mat66f::Zero(); Vec6f b = Vec6f::Zero();
		float levelCutoffRepeat=1;
//[ ***step 1*** ] 计算残差, 保证最多60%残差大于阈值, 计算正规方程
		Vec6 resOld = calcResInv(lvl, refToNew_current, setting_coarseCutoffTH*levelCutoffRepeat, resInliers, H, b);
		
		//* 保证residual大于阈值的点小于60%
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50) 
		{
			levelCutoffRepeat*=2;		// residual超过阈值的多, 则放大阈值重新计算
			resOld = calcResInv(lvl, refToNew_current, setting_coarseCutoffTH*levelCutoffRepeat, resInliers, H, b);

			if(!setting_debugout_runquiet)
				printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}

		// calcGSSSE_inv(lvl, H, b, resInliers);

		float lambda = 0.01;

		if(debugPrint)
		{	
			printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					0.0f,
					resOld[0] / resOld[1],
					0,(int)resOld[1],
					0.0f);
			std::cout << refToNew_current.log().transpose() << ")\n";
		}

//[ ***step 2*** ] 迭代优化
		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			//[ ***step 2.1*** ] 计算增量
			Mat66f Hl = H;
			Mat66f Htmp;
			Vec6f btmp;

			for(int i=0;i<6;i++) Hl(i,i) *= (1+lambda);
			Vec6f inc = Hl.ldlt().solve(-b);

			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec6 incScaled = inc.cast<double>();
			// XTL 这里写的有问题，se3平移在前，旋转在后 TODO
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;

			if(!std::isfinite(incScaled.sum())) incScaled.setZero();
			//[ ***step 2.2*** ] 使用增量更新后, 重新计算能量值
			SE3 refToNew_new = SE3::exp((Vec6)incScaled) * refToNew_current;
			
			Vec6 resNew = calcResInv(lvl, refToNew_new, setting_coarseCutoffTH*levelCutoffRepeat, resInliers, Htmp, btmp);
			
			bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);  // 平均能量值小则接受

			if(debugPrint)
			{
				printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << ")\n";
			}
			//[ ***step 2.3*** ] 接受则求正规方程, 继续迭代, 优化到增量足够小
			if(accept)
			{
				// calcGSSSE_inv(lvl, H, b, resInliers);
				H = Htmp;
				b = btmp;
				resOld = resNew;
				refToNew_current = refToNew_new;
				lambda *= 0.5;
			}
			else
			{
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
			}
			
			if(!(inc.norm() > 1e-3))
			{
				if(debugPrint)
					printf("inc too small, break!\n");
				break;
			}
		}
//[ ***step 3*** ] 记录上一次残差, 光流指示, 如果调整过阈值则重新计算这一层
		// set last residual for that level, as well as flow indicators.
		lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));  // 上一次的平均能量
		lastFlowIndicators = resOld.segment<3>(2);

		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]){
			return false;  //! 如果算出来大于最好的直接放弃
		}

		// XTL 代表这层的残差大部分超过了阈值，因此得重新算一遍这层
		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL!\n");
		}
	}
	// set!
	lastToNew_out = refToNew_current;

	return true;
}

bool CoarseTracker::trackNewestCoarse_ransac(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out,	int coarsestLvl, 
		Vec5 minResForAbort, SE3 slastToLast,
		IOWrap::Output3DWrapper* wrap)
{
	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);

	newFrame = newFrameHessian;

	int maxIterations[] = {10,20,50,50,50};  	// 不同层迭代的次数
	
	float lambdaExtrapolationLimit = 0.001;

	SE3 refToNew_current = lastToNew_out;		// 优化的初始值

	bool haveRepeated = false;

	SE3 pose_store[coarsestLvl+1];
	std::vector<int> reprojectVec[coarsestLvl+1];
	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
		Mat66 H = Mat66::Zero(); Vec6 b = Vec6::Zero();
		float levelCutoffRepeat=1;
//[ ***step 1*** ] 计算残差, 保证最多60%残差大于阈值, 计算正规方程
		Vec6 resOld = calcRes(lvl, refToNew_current, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat);
		
		//* 保证residual大于阈值的点小于60%
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50) 
		{
			levelCutoffRepeat*=2;		// residual超过阈值的多, 则放大阈值重新计算
			resOld = calcRes(lvl, refToNew_current, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat);

			if(!setting_debugout_runquiet)
				printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}

		calcGSSSE(lvl, H, b, refToNew_current, slastToLast);

		float lambda = 0.01;

		if(debugPrint)
		{	
			printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					0.0f,
					resOld[0] / resOld[1],
					0,(int)resOld[1],
					0.0f);
			std::cout << refToNew_current.log().transpose() << ")\n";
		}

//[ ***step 2*** ] 迭代优化
		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			//[ ***step 2.1*** ] 计算增量
			Mat66 Hl = H;

			for(int i=0;i<6;i++) Hl(i,i) *= (1+lambda);
			Vec6 inc = Hl.ldlt().solve(-b);

			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec6 incScaled = inc;
			// XTL 这里写的有问题，se3平移在前，旋转在后 TODO
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;

			if(!std::isfinite(incScaled.sum())) incScaled.setZero();
			//[ ***step 2.2*** ] 使用增量更新后, 重新计算能量值
			SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
			
			Vec6 resNew = calcRes_ransac(lvl, refToNew_new, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat, reprojectVec[lvl]);
			
			bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);  // 平均能量值小则接受

			if(debugPrint)
			{
				printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << ")\n";
			}
			//[ ***step 2.3*** ] 接受则求正规方程, 继续迭代, 优化到增量足够小
			if(accept)
			{
				calcGSSSE(lvl, H, b, refToNew_new, slastToLast);
				resOld = resNew;
				refToNew_current = refToNew_new;
				pose_store[lvl] = refToNew_new;
				lambda *= 0.5;
			}
			else
			{
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
			}
			
			if(!(inc.norm() > 1e-3))
			{
				if(debugPrint)
					printf("inc too small, break!\n");
				break;
			}
		}
//[ ***step 3*** ] 记录上一次残差, 光流指示, 如果调整过阈值则重新计算这一层
		// set last residual for that level, as well as flow indicators.
		lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));  // 上一次的平均能量
		lastFlowIndicators = resOld.segment<3>(2);

		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]){
			return false;  //! 如果算出来大于最好的直接放弃 TODO
		}

		// XTL 代表这层的残差大部分超过了阈值，因此得重新算一遍这层
		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL!\n");
		}
	}
	
	
	int itr = 0;
	float min_Res = lastResiduals[0];
	SE3 refToNew_BestRansac;
	int max_itr = nThreads;
	// lvl 越小， reprojectVec的size越大
	int startLvl = 3;
	int numCross[4];
	numCross[0] = std::min(150,(int)reprojectVec[0].size()/2);
	numCross[1] = std::min(100,(int)reprojectVec[1].size()/2);
	numCross[2] = std::min(80,(int)reprojectVec[2].size()/2);
	numCross[3] = std::min(50,(int)reprojectVec[3].size()/2);
	// nThreads = std::min(max_itr,nThreads);
	if(numCross[3]>5&&numCross[2]>5&&numCross[1]>5&&numCross[0]>5)
	{
		float lastResiduals_[startLvl+1];
		SE3 refToNew_ransac = pose_store[startLvl]/* pose_store[startLvl] */;		// 优化的初始值
		#pragma omp parallel for num_threads(nThreads)
		for(itr=0;itr<max_itr;itr++)
		{
			float avgRes[startLvl+1];
			std::queue<int> idx_to_reproject[coarsestLvl+1][CAM_NUM];
			for(int i=0;i<=startLvl;i++)
			{
				std::random_device rd;
				std::mt19937 g(rd());
				std::vector<int> copy_(reprojectVec[i]);
				std::shuffle(copy_.begin(), copy_.end(), /* g */ m_RandEngines[omp_get_thread_num()]); // TODO 可能选到一样的点
				copy_.resize(numCross[i]);
				sort(copy_.begin(), copy_.end());
				{
					int shuffleIdx = 0;
					int sum = 0;
					int idx = 0;
					int nl = pc_n[0][i];
					while(idx<cam_num)
					{
						if(copy_[shuffleIdx]<sum+nl)
						{
							idx_to_reproject[i][idx].push(copy_[shuffleIdx]-sum);
							shuffleIdx++;
							if(shuffleIdx==numCross[i])
								break;
						}
						else
						{
							idx++;
							sum += nl;
							nl = pc_n[idx][i];
						}
					}
				}
			}
			bool fail = false;
			haveRepeated = false;
			for(int lvl=startLvl; lvl>=0; lvl--)
			{
				Mat66 H = Mat66::Zero(); Vec6 b = Vec6::Zero();
				float levelCutoffRepeat=1;
		//[ ***step 1*** ] 计算残差, 保证最多60%残差大于阈值, 计算正规方程
				Vec6 resOld = calcRes_cross(lvl, refToNew_ransac, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat, idx_to_reproject);
				
				//* 保证residual大于阈值的点小于60%
				while(resOld[5] > 0.6 && levelCutoffRepeat < 50) 
				{
					levelCutoffRepeat*=2;		// residual超过阈值的多, 则放大阈值重新计算
					resOld = calcRes_cross(lvl, refToNew_ransac, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat, idx_to_reproject);

					if(!setting_debugout_runquiet)
						printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
				}

				calcGSSSE(lvl, H, b, refToNew_ransac, slastToLast);

				float lambda = 0.01;

				if(debugPrint)
				{	
					printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
							lvl, -1, lambda, 1.0f,
							"INITIA",
							0.0f,
							resOld[0] / resOld[1],
							0,(int)resOld[1],
							0.0f);
					std::cout << refToNew_ransac.log().transpose() << ")\n";
				}

		//[ ***step 2*** ] 迭代优化
				for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
				{
					//[ ***step 2.1*** ] 计算增量
					Mat66 Hl = H;

					for(int i=0;i<6;i++) Hl(i,i) *= (1+lambda);
					Vec6 inc = Hl.ldlt().solve(-b);

					float extrapFac = 1;
					if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
					inc *= extrapFac;

					Vec6 incScaled = inc;
					// XTL 这里写的有问题，se3平移在前，旋转在后 TODO
					incScaled.segment<3>(0) *= SCALE_XI_ROT;
					incScaled.segment<3>(3) *= SCALE_XI_TRANS;

					if(!std::isfinite(incScaled.sum())) incScaled.setZero();
					//[ ***step 2.2*** ] 使用增量更新后, 重新计算能量值
					SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_ransac;
					
					Vec6 resNew = calcRes_cross(lvl, refToNew_new, slastToLast, setting_coarseCutoffTH*levelCutoffRepeat, idx_to_reproject);

					
					bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);  // 平均能量值小则接受

					if(debugPrint)
					{
						printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
								lvl, iteration, lambda,
								extrapFac,
								(accept ? "ACCEPT" : "REJECT"),
								resOld[0] / resOld[1],
								resNew[0] / resNew[1],
								(int)resOld[1], (int)resNew[1],
								inc.norm());
						std::cout << refToNew_new.log().transpose() << ")\n";
					}
					//[ ***step 2.3*** ] 接受则求正规方程, 继续迭代, 优化到增量足够小
					if(accept)
					{
						calcGSSSE(lvl, H, b, refToNew_new, slastToLast);
						resOld = resNew;
						refToNew_ransac = refToNew_new;
						lambda *= 0.5;
					}
					else
					{
						lambda *= 4;
						if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
					}
					
					if(!(inc.norm() > 1e-3))
					{
						if(debugPrint)
							printf("inc too small, break!\n");
						break;
					}
				}
		//[ ***step 3*** ] 记录上一次残差, 光流指示, 如果调整过阈值则重新计算这一层
				// set last residual for that level, as well as flow indicators.
				avgRes[lvl] = sqrtf((float)(resOld[0] / resOld[1]));  // 上一次的平均能量

				if(avgRes[lvl] > 1.5*lastResiduals[lvl] || avgRes[lvl] > 1.5*minResForAbort[lvl])
				{
					fail = true;
					break;
				}

				// lastFlowIndicators = resOld.segment<3>(2); TODO

				// XTL 代表这层的残差大部分超过了阈值，因此得重新算一遍这层
				if(levelCutoffRepeat > 1 && !haveRepeated)
				{
					lvl++;
					haveRepeated=true;
					printf("REPEAT LEVEL!\n");
				}
			}
			if(!fail)
			{
				#pragma omp critical
				{
				if(avgRes[0]<min_Res)
				{
					min_Res = avgRes[0];
					refToNew_BestRansac = refToNew_ransac;
					for(int i=0;i<startLvl+1;i++)
						lastResiduals_[i] = avgRes[i];
				}
				}
			}
		}
		// set!
		if(min_Res<lastResiduals[0])
		{
			std::cout<<"residual ori="<<lastResiduals[0]<<",residual now="<<min_Res<<std::endl;
			lastToNew_out = refToNew_BestRansac;
			for(int i=0;i<startLvl+1;i++)
				lastResiduals[i] = lastResiduals_[i];
		}
		else
			lastToNew_out = refToNew_current;
	}
	else
	// set!
	lastToNew_out = refToNew_current;

	return true;
}


void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;


	int lvl = 0;

	{
		std::vector<float> allID;
		for(int idx=0;idx<cam_num;idx++)
			for(int i=0;i<h[lvl]*w[lvl];i++)
			{
				// 2021.11.23 TODO
				if(idepth[idx][lvl][i] > 0)
					allID.push_back(idepth[idx][lvl][i]);
			}
		std::sort(allID.begin(), allID.end());
		int n = allID.size()-1;

		float minID, maxID;
		if(n>0)
		{
			float minID_new = allID[(int)(n*0.05)];
			float maxID_new = allID[(int)(n*0.95)];

			minID = minID_new;
			maxID = maxID_new;
			if(minID_pt!=0 && maxID_pt!=0)
			{
				if(*minID_pt < 0 || *maxID_pt < 0)
				{
					*maxID_pt = maxID;
					*minID_pt = minID;
				}
				else
				{

					// slowly adapt: change by maximum 10% of old span.
					float maxChange = 0.3*(*maxID_pt - *minID_pt);

					if(minID < *minID_pt - maxChange)
						minID = *minID_pt - maxChange;
					if(minID > *minID_pt + maxChange)
						minID = *minID_pt + maxChange;


					if(maxID < *maxID_pt - maxChange)
						maxID = *maxID_pt - maxChange;
					if(maxID > *maxID_pt + maxChange)
						maxID = *maxID_pt + maxChange;

					*maxID_pt = maxID;
					*minID_pt = minID;
				}
			}
		}
		else
		{
			*maxID_pt = *minID_pt = 1;
			minID = maxID = 1;
		}

		// 2021.11.23 TODO 这个mf离开它的作用域后会被删除吗？
		std::vector<MinimalImageB3*> _mf(CAM_NUM);
		for(int i=0;i<cam_num;i++){
			_mf[i] = new MinimalImageB3(h[lvl],w[lvl]);
		}
		for(int c_idx=0;c_idx<cam_num;c_idx++){
			_mf[c_idx]->setBlack();
			for(int x=0;x<h[lvl];x++){
				for(int y=0;y<w[lvl];y++){
					int c = lastRef->frame[c_idx]->dIp[lvl][x*w[lvl]+y][0]*0.9f;
					if(c>255) c=255;
					_mf[c_idx]->at(x,y) = Vec3b(c,c,c);
				}
			}
			int wl = w[lvl];
			for(int y=3;y<h[lvl]-3;y++)
				for(int x=3;x<wl-3;x++)
				{
					int idx=x+y*wl;
					float sid=0, nid=0;
					float* bp = idepth[c_idx][lvl]+idx;

					if(bp[0] > 0) {sid+=bp[0]; nid++;}
					if(bp[1] > 0) {sid+=bp[1]; nid++;}
					if(bp[-1] > 0) {sid+=bp[-1]; nid++;}
					if(bp[wl] > 0) {sid+=bp[wl]; nid++;}
					if(bp[-wl] > 0) {sid+=bp[-wl]; nid++;}

					if(bp[0] > 0 || nid >= 3)
					{
						float id = ((sid / nid)-minID) / ((maxID-minID));
						_mf[c_idx]->setPixelCirc(y,x,makeJet3B(id));
					}
				}
		}

		for(IOWrap::Output3DWrapper* ow : wraps)
			ow->pushDepthImage(_mf);

		if(debugSaveImages)
		{
			char buf[1000];
			for(int i:idx_use)
			{
				snprintf(buf, 1000, "images_out/predicted_%05d_%05d_%05d.png", lastRef->shell->id, refFrameID,i);
				IOWrap::writeImage(buf,_mf[i]);
			}
		}

		for(int i=0;i<cam_num;i++)
			delete 	_mf[i];
	}
}



void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;
    int lvl = 0;
    MinimalImageF mim(w[lvl], h[lvl], idepth[0][lvl]);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImageFloat(&mim, lastRef);
}











CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
{
	//* 在第一层上算的, 所以除4
	fwdWarpedIDDistFinal = new float[ww*hh/4];

	bfsList1 = new Eigen::Vector2i[ww*hh/4];
	bfsList2 = new Eigen::Vector2i[ww*hh/4];

	int fac = 1 << (pyrLevelsUsed-1);


	coarseProjectionGrid = new PointFrameResidual*[2048*(ww*hh/(fac*fac))];
	coarseProjectionGridNum = new int[ww*hh/(fac*fac)];

	w[0]=h[0]=0;
}
CoarseDistanceMap::~CoarseDistanceMap()
{
	delete[] fwdWarpedIDDistFinal;
	delete[] bfsList1;
	delete[] bfsList2;
	delete[] coarseProjectionGrid;
	delete[] coarseProjectionGridNum;
}




//@ 对于目前所有的地图点投影, 生成距离场图
void CoarseDistanceMap::makeDistanceMap(
		std::vector<FrameHessian*> frameHessians,
		FrameHessian* frame, int cam_idx)
{
	int w1 = w[1];
	int h1 = h[1];
	int wh1 = w1*h1;
	for(int i=0;i<wh1;i++)
		fwdWarpedIDDistFinal[i] = 1000;


	// make coarse tracking templates for latstRef.
	int numItems = 0;

	for(FrameHessian* fh : frameHessians)
	{
		if(frame == fh) continue;
/* 		if(setting_useMultualPBack)
		{
			for(int j=0;j<5;j++)
			{
				SE3 fhToNew = T_c_c0[cam_idx] * frame->PRE_worldToCam * fh->PRE_camToWorld * T_c0_c[j];

				Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]); // 0层到1层变换
				Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());
				
				for(PointHessian* ph : fh->frame[j]->pointHessians)// 2021.11.15
				{
					assert(ph->status == PointHessian::ACTIVE);
					Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*ph->idepth_scaled; // 投影到frame帧
					int u = ptp[0] / ptp[2] + 0.5f;
					int v = ptp[1] / ptp[2] + 0.5f;

					if(!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
					if(!(maskG[cam_idx][1][u+v*w[1]])) continue;

					fwdWarpedIDDistFinal[u+w1*v]=0;
					bfsList1[numItems] = Eigen::Vector2i(u,v);
					numItems++;
				}
			}
		}
		else
		{ */
			SE3 fhToNew = T_c_c0[cam_idx] * frame->PRE_worldToCam * fh->PRE_camToWorld * T_c0_c[cam_idx];

			Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]); // 0层到1层变换
			Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());
			
			for(PointHessian* ph : fh->frame[cam_idx]->pointHessians)// 2021.11.15
			{
				assert(ph->status == PointHessian::ACTIVE);
				Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*ph->idepth_scaled; // 投影到frame帧
				int u = ptp[0] / ptp[2] + 0.5f;
				int v = ptp[1] / ptp[2] + 0.5f;

				if(!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
				if(!(maskG[cam_idx][1][u+v*w[1]])) continue;

				fwdWarpedIDDistFinal[u+w1*v]=0;
				bfsList1[numItems] = Eigen::Vector2i(u,v);
				numItems++;
			}
		// }
	}

	growDistBFS(numItems);
}




void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian*> frameHessians)
{

}


//@ 生成每一层的距离, 第一层为1, 第二层为2....
// 2021.12.01 这个算法很有意思，bfsList2只是个容器，里面原先的内容不影响结果。
// 通过不断交换两个bfsList，可以在fwdWarpedIDDistFinal里面保存下来信息：fwdWarpedIDDistFinal[x+y*w]=(x,y)与最近的投影点的距离
void CoarseDistanceMap::growDistBFS(int bfsNum)
{
	assert(w[0] != 0);
	int w1 = w[1], h1 = h[1];
	for(int k=1;k<40;k++) // 只考虑40步能走到的区域
	{
		int bfsNum2 = bfsNum;
		//* 每一次都是在上一次的点周围找
		std::swap<Eigen::Vector2i*>(bfsList1,bfsList2); // 每次迭代一遍就交换
		bfsNum=0;

		if(k%2==0) // 偶数
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;
				
				//* 右边
				if(fwdWarpedIDDistFinal[idx+1] > k) // 没有赋值的位置
				{
					fwdWarpedIDDistFinal[idx+1] = k; // 赋值为2, 4, 6 ....
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				//* 左边
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				//* 下边
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				//* 上边
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}
			}
		}
		else
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;
				//* 上下左右
				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}

				//* 四个角
				if(fwdWarpedIDDistFinal[idx+1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y-1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y-1); bfsNum++;
				}
			}
		}
	}
}

//@ 在点(u, v)附近生成距离场
void CoarseDistanceMap::addIntoDistFinal(int u, int v)
{
	if(w[0] == 0) return;
	bfsList1[0] = Eigen::Vector2i(u,v);
	fwdWarpedIDDistFinal[u+w[1]*v] = 0;
	growDistBFS(1);
}


void CoarseDistanceMap::makeK(CalibHessian* HCalib, int cam_idx)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[cam_idx] = HCalib->fxl(cam_idx);
	fy[cam_idx] = HCalib->fyl(cam_idx);
	cx[cam_idx] = HCalib->cxl(cam_idx);
	cy[cam_idx] = HCalib->cyl(cam_idx);

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

}
