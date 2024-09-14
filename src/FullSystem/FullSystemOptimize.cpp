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

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace dso
{




//@ 参数: [true是applyRes, 并去掉不好的残差] [false不进行固定线性化]
// XTL 遍历所有activeResiduals中的残差项，计算残差，根据计算结果设置state_NewState，将残差累加至(*stats)[0]当中
void FullSystem::linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		PointFrameResidual* r = activeResiduals[k];
		// XTL：将残差项的点投影至目标帧上，若超过边界，设置state_NewState为OOB，返回state_energy;
		// XTL：计算残差和导数，设置各种导数J，设置state_NewEnergyWithOutlier为计算出来的残差
		// XTL：若残差大于主帧和目标帧的能量阈值，则设置state_NewState为OUTLIER;否则，设为IN;设置state_NewEnergy=energyLeft
		(*stats)[0] += r->linearize(&Hcalib); // 线性化得到能量


		// XTL：在OPTIMIZE中调用该函数时，不会运行下面的代码
		if(fixLinearization /*&& weightC[r->host->_cam_idx]!=0*/) // 固定线性化（优化后执行）
		{
			r->applyRes(true); // 把值给efResidual

			// 若isActiveAndIsGoodNEW为true（等价于r的新状态为IN）
			if(r->efResidual->isActive()) // 残差是in的
			{
				// XTL：将残差项当中的点在其主帧下的深度设为无穷，投影至目标帧上，计算其坐标，再将残差项以真实的深度投影过去，计算两个坐标的距离，选择最大值设置为该点的maxRelBaseline
				if(r->isNew)
				{
					PointHessian* p = r->point;
					// xtl 2022.1.30
					int idx_i = r->host->_cam_idx;
					int idx_j = r->target->_cam_idx;
					Vec3f ptp_inf = r->host->fh0->targetPrecalc[r->target->fh0->idx].PRE_KRKiTll[idx_i*cam_num+idx_j] * Vec3f(p->u,p->v, 1);	// projected point assuming infinite depth.
					Vec3f ptp = ptp_inf + r->host->fh0->targetPrecalc[r->target->fh0->idx].PRE_KtTll[idx_i*cam_num+idx_j]*p->idepth_scaled;	// projected point with real depth.
					
					float relBS = 0.01*((ptp_inf.head<2>() / ptp_inf[2])-(ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel.
					
					if(relBS > p->maxRelBaseline)
						p->maxRelBaseline = relBS; // 正比于点的基线长度

					p->numGoodResiduals++;
				}
			}
			else// XTL：若状态不为IN，将残差项放入toRemove
			{
				toRemove[tid].push_back(activeResiduals[k]); // 残差太大则移除
			}
		}
	}
}

//@ 把线性化结果传给能量函数efResidual, copyJacobians [true: 更新jacobian] [false: 不更新]
void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++)
	{
		// if(weightC[activeResiduals[k]->host->_cam_idx]!=0)
			activeResiduals[k]->applyRes(true);
	}
}

// XTL：计算最新一帧的能量阈值
void FullSystem::setNewFrameEnergyTH()
{

	// collect all residuals and make decision on TH.
	// 2021.11.18 TODO 用空间换时间
	FrameHessian* newFrame = frameHessians.back();
	allResVec.reserve(activeResiduals.size());
	allResVec.clear();
	for(int i=0;i<cam_num;i++)
	{
		// if(weightC[i]==0&&setting_BackChoose)
		// 	continue;

		// XTL：遍历所有计算过残差且目标帧为最新一帧的残差项，将残差放入allResVec
		for(PointFrameResidual* r : activeResiduals)
			if(r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame->frame[i]) // 新的帧上残差
			{
				allResVec.push_back(r->state_NewEnergyWithOutlier);
			}
	}

	if(allResVec.size()==0)
	{
		for(int i=0;i<cam_num;i++)
			newFrame->frame[i]->frameEnergyTH = 12*12*patternNum;
		return;		// should never happen, but lets make sure.
	}

	int nthIdx = setting_frameEnergyTHN*allResVec.size(); // 以 setting_frameEnergyTHN 的能量为阈值

	assert(nthIdx < (int)allResVec.size());
	assert(setting_frameEnergyTHN < 1);

	std::nth_element(allResVec.begin(), allResVec.begin()+nthIdx, allResVec.end()); // 排序
	float nthElement = sqrtf(allResVec[nthIdx]); // 70% 的值都小于这个值

	for(int i=0;i<cam_num;i++)
	{
		newFrame->frame[i]->frameEnergyTH = nthElement*setting_frameEnergyTHFacMedian;
		newFrame->frame[i]->frameEnergyTH = 26.0f*setting_frameEnergyTHConstWeight + newFrame->frame[i]->frameEnergyTH*(1-setting_frameEnergyTHConstWeight);
		newFrame->frame[i]->frameEnergyTH = newFrame->frame[i]->frameEnergyTH*newFrame->frame[i]->frameEnergyTH;
		newFrame->frame[i]->frameEnergyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;
	}


	//
	//	int good=0,bad=0;
	//	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
	//	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
	//			meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
	//			good, bad);
}
// XTL：遍历所有activeResiduals中的残差项，计算残差及导数，设置残差项状态，将残差累加至(*stats)[0]当中
// XTL：遍历所有activeResiduals中的残差，更新对应lastResiduals状态为残差的state_state
// 若残差需要删除（计算出来的残差大于阈值或者投影到目标帧时超过范围），则清空对应点的对应lastResiduals和对应residuals
Vec3 FullSystem::linearizeAll(bool fixLinearization)
{
	double lastEnergyP = 0;
	double lastEnergyR = 0;
	double num = 0;


	std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
	for(int i=0;i<NUM_THREADS;i++) toRemove[i].clear();

	// XTL：遍历所有activeResiduals中的残差项，将残差累加至(*stats)[0]当中
	if(multiThreading)
	{
		treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
		lastEnergyP = treadReduce.stats[0];
	}
	else
	{
		Vec10 stats;
		linearizeAll_Reductor(fixLinearization, toRemove, 0,activeResiduals.size(),&stats,0);
		lastEnergyP = stats[0];
	}

	setNewFrameEnergyTH();

	// XTL：遍历所有activeResiduals中的残差，更新残差对应点的lastResiduals状态
	// 若残差需要删除（计算出来的残差大于阈值或者投影到目标帧时超过范围），则清空对应点的对应lastResiduals和对应residuals
	if(fixLinearization)
	{
		for(PointFrameResidual* r : activeResiduals)
		{
			PointHessian* ph = r->point;
			if(setting_useMultualPBack)
				for(int i=0;i<5;i++)
				{
					if(ph->lastResiduals[2*i].first == r)
						ph->lastResiduals[2*i].second = r->state_state;
					else if(ph->lastResiduals[2*i+1].first == r)
						ph->lastResiduals[2*i+1].second = r->state_state;
				}
			else
			{
				if(ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].second = r->state_state;
				else if(ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].second = r->state_state;
			}
		}
		
		int nResRemoved=0;
		for(int i=0;i<NUM_THREADS;i++)
		{
			for(PointFrameResidual* r : toRemove[i])
			{
				PointHessian* ph = r->point;
				if(setting_useMultualPBack)
					for(int i=0;i<5;i++)
					{
						if(ph->lastResiduals[i*2].first == r)
							ph->lastResiduals[i*2].first=0;
						else if(ph->lastResiduals[i*2+1].first == r)
							ph->lastResiduals[i*2+1].first=0;
					}
				else
				{
					if(ph->lastResiduals[0].first == r)
						ph->lastResiduals[0].first=0;
					else if(ph->lastResiduals[1].first == r)
						ph->lastResiduals[1].first=0;
				}

				for(unsigned int k=0; k<ph->residuals.size();k++)
					if(ph->residuals[k] == r)
					{
						// if(weightC[r->host->_cam_idx]!=0)
							ef->dropResidual(r->efResidual);
						deleteOut<PointFrameResidual>(ph->residuals,k); // residuals删除第k个
						nResRemoved++;
						break;
					}
			}
		}
		//printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved, (int)activeResiduals.size());

	}

	return Vec3(lastEnergyP, lastEnergyR, num);
}




// applies step to linearization point.
bool FullSystem::doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD)
{
	//	float meanStepC=0,meanStepP=0,meanStepD=0;
	//	meanStepC += Hcalib.step.norm();

	// 步长
	Vec10 pstepfac;
	pstepfac.segment<3>(0).setConstant(stepfacT);
	pstepfac.segment<3>(3).setConstant(stepfacR);

	pstepfac.segment<4>(6).setConstant(stepfacA);

	float sumA=0, sumB=0, sumT=0, sumR=0, sumID=0, numID=0;

	float sumNID=0;

	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		for(int i=0;i<cam_num;i++){
			Hcalib.setValue(Hcalib.value_backup[i] + Hcalib.step[i],i);
		}
		// Hcalib.setValue(Hcalib.value_backup + Hcalib.step); // 内参的值进行update
		for(FrameHessian* fh : frameHessians)
		{
			Vec10 step = fh->step;
			step.head<6>() += 0.5f*(fh->step_backup.head<6>());

			fh->setState(fh->state_backup + step);  // 位姿 光度 update
			sumA += step[6]*step[6];
			sumB += step[7]*step[7];
			sumT += step.segment<3>(0).squaredNorm();  	// 平移增量
			sumR += step.segment<3>(3).squaredNorm();	// 旋转增量

			for(int i=0;i<cam_num;i++){
				for(PointHessian* ph : fh->frame[i]->pointHessians)
				{
					float step = ph->step+0.5f*(ph->step_backup);
					ph->setIdepth(ph->idepth_backup + step);
					sumID += step*step;		// 逆深度增量平方
					sumNID += fabsf(ph->idepth_backup);	 	// 逆深度求和
					numID++;

					//* 逆深度没有使用FEJ
					ph->setIdepthZero(ph->idepth_backup + step); 
				}
			}
		}
	}
	else
	{	//* 相机内参更新状态
		for(int i=0;i<cam_num;i++){
			Hcalib.setValue(Hcalib.value_backup[i] + stepfacC*Hcalib.step[i],i);
		}
		// Hcalib.setValue(Hcalib.value_backup + stepfacC*Hcalib.step);
		for(FrameHessian* fh : frameHessians)
		{
			fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
			sumA += fh->step[6]*fh->step[6];
			sumB += fh->step[7]*fh->step[7];
			sumT += fh->step.segment<3>(0).squaredNorm();
			sumR += fh->step.segment<3>(3).squaredNorm();
			
			for(int i=0;i<cam_num;i++){
				//* 点的逆深度更新, 注意点逆深度没使用FEJ
				for(PointHessian* ph : fh->frame[i]->pointHessians)
				{
					ph->setIdepth(ph->idepth_backup + stepfacD*ph->step);
					sumID += ph->step*ph->step;
					sumNID += fabsf(ph->idepth_backup);
					numID++;

					ph->setIdepthZero(ph->idepth_backup + stepfacD*ph->step);
				}
			}
		}
	}

	sumA /= frameHessians.size();
	sumB /= frameHessians.size();
	sumR /= frameHessians.size();
	sumT /= frameHessians.size();
	sumID /= numID;
	sumNID /= numID;


	//if(!setting_debugout_runquiet)
	/*
	printf("STEPS:A %.1f;B %.1f; R %.1f; T %.1f. \n",
				sqrtf(sumA) < 0.0005*setting_thOptIterations,
				sqrtf(sumB) < 0.0005*setting_thOptIterations,
                sqrtf(sumR) / (0.00005*setting_thOptIterations),
                sqrtf(sumT)*sumNID / (0.00005*setting_thOptIterations));
	*/
	

	EFDeltaValid=false;
	setPrecalcValues();  // 更新相对位姿, 光度


	return  sqrtf(sumA) < 0.0005*setting_thOptIterations &&
			sqrtf(sumB) < 0.00005*setting_thOptIterations &&/*canstop&&*/
			sqrtf(sumR) < 0.00005*setting_thOptIterations &&
			sqrtf(sumT)*sumNID < 0.00005*setting_thOptIterations;
	//
	//	printf("mean steps: %f %f %f!\n",
	//			meanStepC, meanStepP, meanStepD);
}



// sets linearization point.
//@ 对帧, 点, 内参的step和state进行备份
void FullSystem::backupState(bool backupLastStep)
{
	if(setting_solverMode & SOLVER_MOMENTUM)
	{
		if(backupLastStep)  // 不是第0步
		{
			// 2021.11.16 TODO different step
			for(int i=0;i<cam_num;i++){
				Hcalib.step_backup[i] = Hcalib.step[i];
				Hcalib.value_backup[i] = Hcalib.value[i];
			}
			for(FrameHessian* fh : frameHessians)
			{
				fh->step_backup = fh->step;
				fh->state_backup = fh->get_state();
				for(int i=0;i<cam_num;i++){
					for(PointHessian* ph : fh->frame[i]->pointHessians)
					{
						ph->idepth_backup = ph->idepth;
						ph->step_backup = ph->step;
					}
				}
			}
		}
		else // 迭代前初始化
		{
			for(int i=0;i<cam_num;i++){
				Hcalib.step_backup[i].setZero();
				Hcalib.value_backup[i] = Hcalib.value[i];
			}
			for(FrameHessian* fh : frameHessians)
			{
				fh->step_backup.setZero();
				fh->state_backup = fh->get_state();
				for(int i=0;i<cam_num;i++){
					for(PointHessian* ph : fh->frame[i]->pointHessians)
					{
						ph->idepth_backup = ph->idepth;
						ph->step_backup = ph->step;
					}
				}
			}
		}
	}
	else
	{
		for(int i=0;i<cam_num;i++){
			Hcalib.value_backup[i] = Hcalib.value[i];
		}
		for(FrameHessian* fh : frameHessians)
		{
			fh->state_backup = fh->get_state();
			for(int i=0;i<cam_num;i++)
				for(PointHessian* ph : fh->frame[i]->pointHessians)
					ph->idepth_backup = ph->idepth;
		}
	}
}


// sets linearization point.
void FullSystem::loadSateBackup()
{
	
	for(int i=0;i<cam_num;i++){
		Hcalib.setValue(Hcalib.value_backup[i],i);
	}
	// Hcalib.setValue(Hcalib.value_backup);
	
	for(FrameHessian* fh : frameHessians)
	{
		fh->setState(fh->state_backup);
		
		for(int i=0;i<cam_num;i++)
			for(PointHessian* ph : fh->frame[i]->pointHessians)
			{
				ph->setIdepth(ph->idepth_backup);

				ph->setIdepthZero(ph->idepth_backup); // 没用FEJ
			}

	}


	EFDeltaValid=false;
	setPrecalcValues();  // 更新当前的状态
}

//@ 计算能量, 计算的是绝对的能量
double FullSystem::calcMEnergy()
{
	if(setting_forceAceptStep) return 0;
	// calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
	//ef->makeIDX();
	//ef->setDeltaF(&Hcalib);
	return ef->calcMEnergyF();

}


void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b)
{
	printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
			res[0],
			sqrtf((float)(res[0] / (patternNum*ef->resInA))),
			ef->resInA,
			ef->resInM,
			a,
			b
	);

}

//@ 对所有关键帧进行GN优化
float FullSystem::optimize(int mnumOptIts)
{
	if(frameHessians.size() < 2) return 0;
	if(frameHessians.size() < 3) mnumOptIts = 20; // 迭代次数
	if(frameHessians.size() < 4) mnumOptIts = 15;

	// get statistics and active residuals.
	activeResiduals.clear();
	int numPoints = 0;
	int numLRes = 0;
	// 遍历所有关键帧上关键点的所有残差项，若未被线性化，则放入activeResiduals，并且初始化残差项（state_NewEnergy=state_energy=0,state_NewState=OUTLIER，state_state=IN）
	for(FrameHessian* fh : frameHessians)
		for(int i=0;i<cam_num;i++)
		{
			for(PointHessian* ph : fh->frame[i]->pointHessians)
			{
				for(PointFrameResidual* r : ph->residuals)
				{
					if(!r->efResidual->isLinearized) // 没有求线性误差
					{
						activeResiduals.push_back(r);  // 新加入的残差
						r->resetOOB(); // residual状态重置
					}
					else
						numLRes++;	//已经线性化过得计数
				}
				numPoints++;
			}
		}

    if(!setting_debugout_runquiet)
        printf("OPTIMIZE %d pts, %d active res, %d lin res!\n",ef->nPoints,(int)activeResiduals.size(), numLRes);
	
	// XTL：遍历所有activeResiduals中的残差项，计算残差及导数，设置残差项状态，将残差累加至(*stats)[0]当中
	// XTL：遍历所有activeResiduals中的残差，更新对应lastResiduals状态为残差的state_state
	// 若残差需要删除（计算出来的残差大于阈值或者投影到目标帧时超过范围），则清空对应点的对应lastResiduals和对应residuals
	Vec3 lastEnergy = linearizeAll(false);  
	double lastEnergyL = calcLEnergy(); // islinearized的量的能量
	double lastEnergyM = calcMEnergy(); // HM部分的能量

	// XTL：若残差项的state_state为OOB，判断isActiveAndIsGoodNEW为false，直接返回
	// XTL：若残差项的state_NewState为IN，isActiveAndIsGoodNEW置为true，并且交换对应的efResidual的J与残差项的J，然后计算JpJdF，否则isActiveAndIsGoodNEW置为false
	// XTL：更新状态、能量
	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
	else
		applyRes_Reductor(true,0,activeResiduals.size(),0,0);


    if(!setting_debugout_runquiet)
    {
        printf("Initial Error       \t");
        printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l(0).a, frameHessians.back()->aff_g2l(0).b);// 2021.12.14
    }

	debugPlotTracking();


	double lambda = 1e-1;
	float stepsize=1;

	VecX previousX = VecX::Constant(CPARS+8*frameHessians.size(), NAN);
	
	for(int iteration=0;iteration<mnumOptIts;iteration++)
	{
		// solve!
		backupState(iteration!=0);
		//solveSystemNew(0);
		solveSystem(iteration, lambda);
		double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm()); // 两次下降方向的点积（dot/模长）
		previousX = ef->lastX;

		if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) 
		{
			float newStepsize = exp(incDirChange*1.4);
			if(incDirChange<0 && stepsize>1) stepsize=1;

			stepsize = sqrtf(sqrtf(newStepsize*stepsize*stepsize*stepsize));
			if(stepsize > 2) stepsize=2;
			if(stepsize <0.25) stepsize=0.25;
		}
		bool canbreak = doStepFromBackup(stepsize,stepsize,stepsize,stepsize,stepsize);


		// eval new energy!
		//* 更新后重新计算
		Vec3 newEnergy = linearizeAll(false);
		double newEnergyL = calcLEnergy();
		double newEnergyM = calcMEnergy();

        if(!setting_debugout_runquiet)
        {
            printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
				(newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
						lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				incDirChange,
				stepsize);
            printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0, frameHessians.back()->aff_g2l(0).a, frameHessians.back()->aff_g2l(0).b);
        }
//[ ***step 4*** ] 判断是否接受这次计算
		if(setting_forceAceptStep || (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
				lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
		{  
			// 接受更新后的量
			if(multiThreading)
				treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
			else
				applyRes_Reductor(true,0,activeResiduals.size(),0,0);

			lastEnergy = newEnergy;
			lastEnergyL = newEnergyL;
			lastEnergyM = newEnergyM;

			lambda *= 0.25; // 固定lambda
		}
		else
		{ 	// 不接受, roll back
			loadSateBackup();
			lastEnergy = linearizeAll(false);
			lastEnergyL = calcLEnergy();
			lastEnergyM = calcMEnergy();
			lambda *= 1e2;
		}


		if(canbreak && iteration >= setting_minOptIterations) break;
	}

//[ ***step 5*** ] 把最新帧的位姿设为线性化点
	//* 最新一帧的位姿设为线性化点, 0-5是位姿增量因此是0, 6-7是值, 直接赋值
	Vec10 newStateZero = Vec10::Zero();
	newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6); 
	
	frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
			newStateZero); // 最新帧设置为线性化点, 待估计量
	EFDeltaValid=false;
	EFAdjointsValid=false;
	ef->setAdjointsF(&Hcalib); // 重新计算adj
	setPrecalcValues(); // 更新增量

	// 更新之后的能量
	lastEnergy = linearizeAll(true);


	//* 能量函数太大, 投影的不好, 跟丢
	if(!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
    {
        printf("KF Tracking failed: LOST!\n");
		isLost=true;
    }


	statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

	if(calibLog != 0)
	{
		(*calibLog) << Hcalib.value_scaled[0].transpose() <<
				" " << frameHessians.back()->get_state_scaled().transpose() <<
				" " << sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA))) <<
				" " << ef->resInM << "\n";
		calibLog->flush();
	}
//[ ***step 6*** ] 把优化的结果, 给每个帧的shell, 注意这里其他帧的线性点是不更新的
	//* 把优化结果给shell
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		for(FrameHessian* fh : frameHessians)
		{
			fh->shell->camToWorld = fh->PRE_camToWorld;
			// fh->shell->aff_g2l = fh->aff_g2l();
			for(int i=0;i<cam_num;i++)
				fh->shell->aff_g2l[i] = fh->aff_g2l(i);
		}
	}

	debugPlotTracking();

	//* 返回平均误差rmse
	return sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

}




void FullSystem::solveSystem(int iteration, double lambda)
{
	ef->lastNullspaces_forLogging = getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);

	ef->solveSystemF(iteration, lambda,&Hcalib);
}

// XTL 若是setting_forceAceptStep(defalut setting)，则不会计算L能量
double FullSystem::calcLEnergy()
{
	if(setting_forceAceptStep) return 0;

	double Ef = ef->calcLEnergyF_MT();
	return Ef;

}

// XTL：移除外点
// XTL：遍历每个关键帧的关键点，若残差数目为0,则删除该点
void FullSystem::removeOutliers()
{
	int numPointsDropped=0;
	for(FrameHessian* fh : frameHessians)
	{
		// 2021.11.18
		for(int idx=0;idx<cam_num;idx++)
			for(unsigned int i=0;i<fh->frame[idx]->pointHessians.size();i++)
			{
				PointHessian* ph = fh->frame[idx]->pointHessians[i];
				if(ph==0) continue;

				if(ph->residuals.size() == 0) // 如果该点的残差数为0, 则丢掉
				{
					fh->frame[idx]->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					fh->frame[idx]->pointHessians[i] = fh->frame[idx]->pointHessians.back();
					fh->frame[idx]->pointHessians.pop_back();
					i--;
					numPointsDropped++;
				}
			}
		// 2021.11.18
	}
	ef->dropPointsF();
}



// XTL:nullspaces_pose是一个vector，里面每个元素是一个大小为CPARS+frameHessians.size()*8的矢量，前面4个空着，后面每8个是一组，前6个为对应帧的零空间，后2=0
// XTL:nullspaces_scale是一个vector，里面每个元素是一个大小为CPARS+frameHessians.size()*8的矢量，前面4个空着，后面每8个是一组，前6个为对应帧的零空间,后2=0
// XTL:nullspaces_scale是一个vector，里面每个元素是一个大小为CPARS+frameHessians.size()*8的矢量，前面4个空着，后面每8个是一组，前6=0,后2个为对应帧的光度参数的零空间
std::vector<VecX> FullSystem::getNullspaces(
		std::vector<VecX> &nullspaces_pose,
		std::vector<VecX> &nullspaces_scale,
		std::vector<VecX> &nullspaces_affA,
		std::vector<VecX> &nullspaces_affB)
{
	nullspaces_pose.clear();  // size: 6; vec: 4+8*n
	nullspaces_scale.clear(); // size: 1; 
	nullspaces_affA.clear();  // size: 1
	nullspaces_affB.clear();  // size: 1 

	int n=CPARS+frameHessians.size()*8;

	std::vector<VecX> nullspaces_x0_pre;  // 所有的零空间
	
	//* 位姿的零空间
	for(int i=0;i<6;i++) // 第i个变量的零空间
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for(FrameHessian* fh : frameHessians)
		{
			nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_pose.col(i);
			nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE; // 去掉scale
			nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		nullspaces_pose.push_back(nullspace_x0);
	}
	//* 光度参数a b的零空间
	for(int i=0;i<2;i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for(FrameHessian* fh : frameHessians)
		{
			nullspace_x0.segment<2>(CPARS+fh->idx*8+6) = fh->nullspaces_affine.col(i).head<2>();
			nullspace_x0[CPARS+fh->idx*8+6] *= SCALE_A_INVERSE;
			nullspace_x0[CPARS+fh->idx*8+7] *= SCALE_B_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		if(i==0) nullspaces_affA.push_back(nullspace_x0);
		if(i==1) nullspaces_affB.push_back(nullspace_x0);
	}

	//* 尺度零空间
	VecX nullspace_x0(n);
	nullspace_x0.setZero();
	for(FrameHessian* fh : frameHessians)
	{
		nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_scale;
		nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;
		nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
	}
	nullspaces_x0_pre.push_back(nullspace_x0);
	nullspaces_scale.push_back(nullspace_x0);

	return nullspaces_x0_pre;
}

}
