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

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace dso
{


//@ 对于关键帧的边缘化策略 1. 活跃点只剩下5%的; 2. 和最新关键帧曝光变化大于0.7; 3. 距离最远的关键帧
void FullSystem::flagFramesForMarginalization(FrameHessian* newFH)
{
	if(setting_minFrameAge > setting_maxFrames)
	{
		for(int i=setting_maxFrames;i<(int)frameHessians.size();i++)
		{
			FrameHessian* fh = frameHessians[i-setting_maxFrames];  // setting_maxFrames个之前的都边缘化掉
			for(int i=0;i<cam_num;i++)
				fh->frame[i]->flaggedForMarginalization = true;
		}
		return;
	}

	int flagged = 0;  // 标记为边缘化的个数
	// marginalize all frames that have not enough points.
	for(int i=0;i<(int)frameHessians.size();i++)
	{
		int in = 0;
		int out = 0;
		FrameHessian* fh = frameHessians[i];
		for(int idx=0;idx<cam_num;idx++){
			if(fh->w[idx]==false)
				continue;
			in += fh->frame[idx]->pointHessians.size() + fh->frame[idx]->immaturePoints.size(); // 还在的点
			out += fh->frame[idx]->pointHessiansMarginalized.size() + fh->frame[idx]->pointHessiansOut.size(); // 边缘化和丢掉的点
		}		
		// if(i==0)
		// printf("frame %d:%d points inVision,totally %d points!\n",
		// 		fh->frameID, in, in+out);

		// assert(in+out!=0);

		Vec2 refToFh=AffLight::fromToVecExposure(frameHessians.back()->ab_exposure[0], fh->ab_exposure[0],
				frameHessians.back()->aff_g2l(0), fh->aff_g2l(0)); // 2021.12.14 TODO
		//* 这一帧里的内点少, 曝光时间差的大, 并且边缘化掉后还有5-7帧, 则边缘化

		if( ((in < setting_minPointsRemaining *(in+out) /*|| fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow*/)
				&& ((int)frameHessians.size())-flagged > setting_minFrames) || in+out==0 )
		{
			//printf("MARGINALIZE frame %d, as only %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
			//		fh->frameID, in, in+out,
			//		(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
			//		(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
			//		visInLast, outInLast,
			//		fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
			
			for(int idx=0;idx<cam_num;idx++)
				fh->frame[idx]->flaggedForMarginalization = true;
			
			flagged+=1;
		}
		else
		{
			//printf("May Keep frame %d, as %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
			//		fh->frameID, in, in+out,
			//		(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
			//		(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
			//		visInLast, outInLast,
			//		fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
		}
	}
	// marginalize one.
	if((int)frameHessians.size()-flagged >= setting_maxFrames)// 2021.11.15
	{
		double smallestScore = 1;
		FrameHessian* toMarginalize=0;
		FrameHessian* latest = frameHessians.back();


		for(FrameHessian* fh : frameHessians)
		{
			//* 至少是setting_minFrameAge个之前的帧 (保留了当前帧)
			if(fh->frameID > latest->frameID-setting_minFrameAge || fh->frameID == 0) continue;
			//if(fh==frameHessians.front() == 0) continue;

			double distScore = 0;
			for(FrameFramePrecalc &ffh : fh->targetPrecalc)
			{
				if(ffh.target->frameID > latest->frameID-setting_minFrameAge+1 || ffh.target == ffh.host) continue;
				distScore += 1/(1e-5+ffh.distanceLL); // 帧间距离

			}
			//* 有负号, 与最新帧距离占所有目标帧最大的被边缘化掉, 离得最远的, 
			// 论文有提到, 启发式的良好的3D空间分布, 关键帧更接近
			distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL); 


			if(distScore < smallestScore)
			{
				smallestScore = distScore;
				toMarginalize = fh;
			}
		}

		// printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
		//		toMarginalize->frameID, smallestScore);
		for(int i=0;i<cam_num;i++){
			toMarginalize->frame[i]->flaggedForMarginalization = true;
		}
		flagged += 1;
	}
	//	printf("FRAMES LEFT: ");
	//	for(FrameHessian* fh : frameHessians)
	//		printf("%d ", fh->frameID);
	//	printf("\n");
}



//@ 边缘化一个关键帧, 删除该帧上的残差
void FullSystem::marginalizeFrame(FrameHessian* frame)
{
	//! marginalize or remove all this frames points.
	for(int i=0;i<cam_num;i++)
		assert((int)frame->frame[i]->pointHessians.size()==0);

	ef->marginalizeFrame(frame->efFrame);

	// drop all observations of existing points in that frame.
	//* 删除其它帧在被边缘化帧上的残差
	for(FrameHessian* fh : frameHessians)
	{
		if(fh==frame) continue;

		for(int idx=0;idx<cam_num;idx++)
			for(PointHessian* ph : fh->frame[idx]->pointHessians)
				for(unsigned int i=0;i<ph->residuals.size();i++)
				{
					PointFrameResidual* r = ph->residuals[i];

					if(setting_useMultualPBack)
					{
						for(int j=0;j<5;j++)
						{
							if(r->target == frame->frame[j])
							{
								if(ph->lastResiduals[2*j].first == r)
									ph->lastResiduals[2*j].first=0;
								else if(ph->lastResiduals[2*j+1].first == r)
									ph->lastResiduals[2*j+1].first=0;

								if(r->host->fh0->frameID < r->target->fh0->frameID)// 2022.1.11
									statistics_numForceDroppedResFwd++;
								else
									statistics_numForceDroppedResBwd++;

								ef->dropResidual(r->efResidual);
								deleteOut<PointFrameResidual>(ph->residuals,i);
								i--; // XTL 源代码没加这个，但是对于源代码来说确实不需要
								break;
							}
						}
					}
					else
					{
						if(r->target == frame->frame[idx])
						{
							if(ph->lastResiduals[0].first == r)
								ph->lastResiduals[0].first=0;
							else if(ph->lastResiduals[1].first == r)
								ph->lastResiduals[1].first=0;

							if(r->host->fh0->frameID < r->target->fh0->frameID)// 2022.1.11
								statistics_numForceDroppedResFwd++;
							else
								statistics_numForceDroppedResBwd++;

							ef->dropResidual(r->efResidual);
							deleteOut<PointFrameResidual>(ph->residuals,i);
							// i--; // XTL 源代码没加这个，但是对于源代码来说确实不需要
							break;
						}
					}
				}
	}


    {
        std::vector<FrameHessian*> v;
        v.push_back(frame);
        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishKeyframes(v, true, &Hcalib);
    }


	frame->shell->marginalizedAt = frameHessians.back()->shell->id;
	frame->shell->movedByOpt = frame->w2c_leftEps().norm();

	for(int i=0;i<cam_num;i++){
		delete frame->frame[i];
	}

	deleteOutOrder<FrameHessian>(frameHessians, frame);

	for(unsigned int i=0;i<frameHessians.size();i++)
	{
		frameHessians[i]->idx = i;
	}

	setPrecalcValues();
	ef->setAdjointsF(&Hcalib);

}




}
