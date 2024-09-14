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
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace dso
{

// XTL：点是从之前关键帧上提取的，符合要求的未成熟点
PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
	int nres = 0;
	// XTL：新的未成熟点与之前的关键帧构建残差，对残差初始化
	// state_NewEnergy=state_energy=0,state_NewState为外点，state_state为IN，目标帧为新关键帧
	if(!setting_useMultualPBack)
	{
	for(FrameHessian* fh : frameHessians)
	{
		if(fh->frame[point->host->_cam_idx] != point->host)  // 不创建和自己的
		{
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh->frame[point->host->_cam_idx];
			nres++; // 观测数
		}
	}
		assert(nres == (((int)frameHessians.size())-1));
	}
	else{
		for(FrameHessian* fh : frameHessians)
		{
			int i = point->host->_cam_idx;
			if(fh->frame[i] == point->host)  // 跳过同一个关键帧
				continue;
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh->frame[i];
			nres++; // 观测数
			int j = 1, idx;
			bool isSecond = false;
			while(j<3)
			{
				if(!isSecond)
				{
					idx = i+j;
					if(idx>=5) idx-=5;
				}
				else
				{
					idx = i-j;
					j++;
					if(idx<0) idx+=5;
				}
				isSecond=!isSecond;
				residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
				residuals[nres].state_NewState = ResState::OUTLIER;
				residuals[nres].state_state = ResState::IN;
				residuals[nres].target = fh->frame[idx];
				nres++; // 观测数
			}/* 
			for(int i=0;i<5;i++)
			{
				if(i==point->host->_cam_idx) continue;  // 跳过已经构建的residual
				if(fh->frame[point->host->_cam_idx] != point->host)  // 跳过同一个关键帧
				{
					residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
					residuals[nres].state_NewState = ResState::OUTLIER;
					residuals[nres].state_state = ResState::IN;
					residuals[nres].target = fh->frame[i];
					nres++; // 观测数
				}
			} */
		}
		assert(nres == (((int)frameHessians.size())-1)*5);
	}

	bool print = false/*rand()%10==0*/;

	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;
	// 2020.06.22 yzk shiyong
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;
	float trueDepth = currentIdepth;
	// 2020.06.22 yzk shiyong

	clock_t started = clock();
	if(point->isFromSensor == false || setting_optDepth)
	{
		// 对该点在所有的帧中的残差项求和，并且求计算逆深度需要的H、b，并且设置各个残差项的状态
		for(int i=0;i<nres;i++)
		{
			lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth);
			residuals[i].state_state = residuals[i].state_NewState;
			residuals[i].state_energy = residuals[i].state_NewEnergy;
		}
		int cnt = 0;
		for(int i=0;i<nres;i++)
		{
			if(residuals[i].state_NewState != ResState::IN)
				cnt++;
		}
		printf("Num of outlier is:%d/7\n",cnt);

		if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
		{
			if(print)
				printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres, lastHdd, lastEnergy);
			return 0;
		}

		if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
				nres, lastHdd,lastEnergy,currentIdepth);

		float lambda = 0.1;
		// XTL：优化逆深度，若残差变成无穷或者H矩阵过小，则直接退出该函数
		for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
		{
			float H = lastHdd;
			H *= 1+lambda;
			float step = (1.0/H) * lastbd;
			float newIdepth = currentIdepth - step;

			float newHdd=0; float newbd=0; float newEnergy=0;
			for(int i=0;i<nres;i++)
				newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,newHdd, newbd, newIdepth);

			if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
			{
				if(print) printf("OptPoint: Not well-constrained (%d res). E=%f. SKIP!\n",
						nres,
						lastEnergy);
				return 0;
			}

			if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
					(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
					iteration,
					log10(lambda),
					"",
					lastEnergy, newEnergy, newIdepth);

			if(newEnergy < lastEnergy)
			{
				currentIdepth = newIdepth;
				lastHdd = newHdd;
				lastbd = newbd;
				lastEnergy = newEnergy;
				for(int i=0;i<nres;i++)
				{
					residuals[i].state_state = residuals[i].state_NewState;
					residuals[i].state_energy = residuals[i].state_NewEnergy;
				}

				lambda *= 0.5;
			}
			else
			{
				lambda *= 5;
			}
			std::cout<<"fabsf(step) < 0.0001*currentIdepth?" << (fabsf(step) < 0.0001*currentIdepth) << std::endl;
			if(fabsf(step) < 0.0001*currentIdepth)
				break;
		}
	}
	clock_t ended = clock();
	if(print)
		printf("Time spent in optPixel is %.2fms\n",1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC));

	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		// 丢弃无穷的点
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	//* 所有观测里面统计good数, 小了则返回
	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs)
	{
		if(print) printf("OptPoint: OUTLIER!\n");
		//! niubility
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}


	// XTL： 利用未成熟点初始化关键点
	PointHessian* p = new PointHessian(point, &Hcalib);
	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));}

	p->isFromSensor = point->isFromSensor;
	p->hasDepthPrior = p->isFromSensor;
	
	for(int i=0;i<10;i++)
	{
		p->lastResiduals[i].first = 0;
		p->lastResiduals[i].second = ResState::OOB;
	}
	// 2020.09.28 yzk
	if(p->isFromSensor == true)
	{
		p->setIdepthZero(trueDepth);
		p->setIdepth(trueDepth);
	}
	else
	{
		p->setIdepthZero(currentIdepth);
		p->setIdepth(currentIdepth);
	}
	// 2020.09.28 yzk
	p->setPointStatus(PointHessian::ACTIVE);

	// XTL：遍历所有未成熟残差项，若状态是IN，则构建残差项，将残差项放入点的residuals当中，设置残差状态为IN，NewState为OUTLIER，Energy=0
	// XTL：若残差的目标帧为 关键帧的最后一帧/倒数第二帧（关键帧数大于1）/0（只有1个关键帧），则设置该点的lastResiduals[0/1].first=r，second为IN
	if(!setting_useMultualPBack)
	{
		for(int i=0;i<nres;i++)
			if(residuals[i].state_state == ResState::IN)
			{
				PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
				r->state_NewEnergy = r->state_energy = 0;
				r->state_NewState = ResState::OUTLIER;
				r->setState(ResState::IN);
				p->residuals.push_back(r);

				if(r->target == frameHessians.back()->frame[r->target->_cam_idx]) // 和最新帧的残差
				{
					p->lastResiduals[0].first = r;
					p->lastResiduals[0].second = ResState::IN;
				}
				else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]->frame[r->target->_cam_idx])) // 和最新帧上一帧
				{
					p->lastResiduals[1].first = r;
					p->lastResiduals[1].second = ResState::IN;
				}
			}
	}
	else
	{
		for(int i_=0;i_<nres;i_+=5)
		{
			for(int j=0;j<5;j++)
			{
				int i = i_+j;
				if(residuals[i].state_state == ResState::IN)
				{
					int idx = residuals[i].target->_cam_idx;
					PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
					r->state_NewEnergy = r->state_energy = 0;
					r->state_NewState = ResState::OUTLIER;
					r->setState(ResState::IN);
					p->residuals.push_back(r);

					if(r->target == frameHessians.back()->frame[idx]) // 和最新帧的残差
					{
						p->lastResiduals[2*idx].first = r;
						p->lastResiduals[2*idx].second = ResState::IN;
					}
					else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]->frame[idx])) // 和最新帧上一帧
					{
						p->lastResiduals[2*idx+1].first = r;
						p->lastResiduals[2*idx+1].second = ResState::IN;
					}
					break;
				}
			}
		}
	}

	if(print) printf("point activated!\n");

	statistics_numActivatedPoints++;
	return p;
}

}
