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
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
int PointFrameResidual::instanceCounter = 0;


long runningResID=0;


PointFrameResidual::PointFrameResidual(){assert(false); instanceCounter++;}

PointFrameResidual::~PointFrameResidual(){assert(efResidual==0); instanceCounter--; delete J;}

PointFrameResidual::PointFrameResidual(PointHessian* point_, frame_hessian* host_, frame_hessian* target_) :// 2022.1.11
	point(point_),
	host(host_),
	target(target_)
{
	efResidual=0;
	instanceCounter++;
	resetOOB();
	J = new RawResidualJacobian(); // 各种雅克比
	assert(((long)J)%16==0); // 16位对齐

	isNew=true;
}



// XTL：将残差项的点投影至目标帧上，若超过边界，设置state_NewState为OOB，返回state_energy;
// XTL：计算残差和导数，设置各种导数J，设置state_NewEnergyWithOutlier为计算出来的残差
// XTL：若残差大于主帧和目标帧的能量阈值，则设置state_NewState为OUTLIER;否则，设为IN;设置state_NewEnergy=energyLeft
double PointFrameResidual::linearize(CalibHessian* HCalib)
{
	state_NewEnergyWithOutlier=-1;

	if(state_state == ResState::OOB)
		{ state_NewState = ResState::OOB; return state_energy; }

	FrameFramePrecalc* precalc = &(host->fh0->targetPrecalc[target->fh0->idx]); // 得到这个目标帧在主帧上的一些预计算参数
	float energyLeft=0;
	const Eigen::Vector3f* dIl = target->dI;

	int idx_i = host->_cam_idx;
	int idx_j = target->_cam_idx;
	assert(idx_j>=0&&idx_j<cam_num);


	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll[idx_i*cam_num+idx_j];
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll[idx_i*cam_num+idx_j];
	const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0[idx_i*cam_num+idx_j];
	const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0[idx_i*cam_num+idx_j];

	const float * const color = point->color;	// host帧上颜色
	const float * const weights = point->weights;

	Vec2f affLL = precalc->PRE_aff_mode[idx_i*cam_num+idx_j]; // 待优化的a和b, 就是host和target合的
	float b0 = precalc->PRE_b0_mode[idx_i];		// 主帧的单独 b

	Mat33f R0x = T_c0_c[idx_j].rotationMatrix().cast<float>();
	Mat33f R = T_c0_c[idx_j].rotationMatrix().transpose().cast<float>();
	Vec3f t0x = T_c0_c[idx_j].translation().cast<float>();
	
	Vec6f d_xi_x, d_xi_y;
	Vec4f d_C_x, d_C_y;
	float d_d_x, d_d_y;
	{
		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0, HCalib,
				PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth, idx_i, idx_j)) // 2021.12.29
		{
			state_NewState = ResState::OOB; return state_energy; 
		} // 投影不在图像里, 则返回OOB

		centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

		// diff d_idepth
		d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*SCALE_IDEPTH*HCalib->fxl(idx_j);
		d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*SCALE_IDEPTH*HCalib->fyl(idx_j);

		// diff calib
		if(/*idx_j==*/0)
		{
			d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));
			d_C_x[3] = HCalib->fxl(idx_j) * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) * HCalib->fyli(idx_j);
			d_C_x[0] = KliP[0]*d_C_x[2];
			d_C_x[1] = KliP[1]*d_C_x[3];


			d_C_y[2] = HCalib->fyl(idx_j) * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) * HCalib->fxli(idx_j);
			d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
			d_C_y[0] = KliP[0]*d_C_y[2];
			d_C_y[1] = KliP[1]*d_C_y[3];


			//* 第二部分 同样project时候一样使用了scaled的内参
			d_C_x[0] = (d_C_x[0]+u)*SCALE_F;
			d_C_x[1] *= SCALE_F;
			d_C_x[2] = (d_C_x[2]+1)*SCALE_C;
			d_C_x[3] *= SCALE_C;

			d_C_y[0] *= SCALE_F;
			d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
			d_C_y[2] *= SCALE_C;
			d_C_y[3] = (d_C_y[3]+1)*SCALE_C;
		}
		else
		{
			d_C_x.setZero();
			d_C_y.setZero();
		}

		// 2021.12.20 计算主相机帧下投影点坐标
		Vec3f p = Vec3f(u,v,1) / new_idepth;
		Vec3f p0 = R0x * p + t0x;
		float P[3];
		for(int i=0;i<3;i++){
			if(i==0)
				for(int j = 0;j<3;j++)
					P[j] = p0(1,0)*R(j,2)-p0(2,0)*R(j,1);
			else if(i==1)
				for(int j = 0;j<3;j++)
					P[j] = p0(2,0)*R(j,0)-p0(0,0)*R(j,2);
			else if(i==2)
				for(int j = 0;j<3;j++)
					P[j] = -p0(1,0)*R(j,0)-p0(0,0)*R(j,1);
			d_xi_x[i] = new_idepth*HCalib->fxl(idx_j)*(R(0,i)-u*R(2,i));
			d_xi_x[i+3] = new_idepth*HCalib->fxl(idx_j)*(P[0]-u*P[2]);
			d_xi_y[i] = new_idepth*HCalib->fyl(idx_j)*(R(1,i)-v*R(2,i));
			d_xi_y[i+3] = new_idepth*HCalib->fyl(idx_j)*(P[1]-v*P[2]);
		}
	}


	{
		J->Jpdxi[0] = d_xi_x;
		J->Jpdxi[1] = d_xi_y;

		J->Jpdc[0] = d_C_x;
		J->Jpdc[1] = d_C_y;

		J->Jpdd[0] = d_d_x;
		J->Jpdd[1] = d_d_y;

	}






	float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
	float JabJIdx_00=0, JabJIdx_01=0, JabJIdx_10=0, JabJIdx_11=0;
	float JabJab_00=0, JabJab_01=0, JabJab_11=0;

	float wJI2_sum = 0;

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
		// 其实和上面一样的....同时调用了setIdepth() setIdepthZero()
		// 这里是求图像导数, 由于线性误差大, 就不使用FEJ, 所以使用当前的状态
		if(!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv, idx_j))
		{ 
			// printf("OOB！！state_energy is%f\n",state_energy);
			state_NewState = ResState::OOB; 
			return state_energy; 
		}
		
		// 像素坐标
		projectedTo[idx][0] = Ku;
		projectedTo[idx][1] = Kv;

        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]); // 残差


		// 残差对光度仿射a求导
		// 光度参数使用固定线性化点了
		float drdA = (color[idx]-b0); 

		if(!std::isfinite((float)hitColor[0]))
		{ 
			// printf("Infinite color！！state_energy is%f\n",state_energy);
			state_NewState = ResState::OOB; 
			return state_energy; 
		}
		// 和梯度大小成比例的权重
		float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
        // 和patch位置相关的权重
		w = 0.5f*(w + weights[idx]); 

        // TODO 主相机和投影相机不一致时减小权重


		// huber函数, 能量值(chi2)
		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += w*w*hw *residual*residual*(2-hw); 

		{
			if(hw < 1) hw = sqrtf(hw);
			hw = hw*w;

			hitColor[1]*=hw;
			hitColor[2]*=hw;

			// 残差 res*w*sqrt(hw)
			J->resF[idx] = residual*hw; 

			// 图像导数 dx dy
			J->JIdx[0][idx] = hitColor[1];
			J->JIdx[1][idx] = hitColor[2];

			// Ij - a*Ii - b  (a = tj*e^aj / ti*e^ai,   b = bj - a*bi) 
			// 对光度合成后a b的导数 -[Ii-bi  1]
			J->JabF[0][idx] = drdA*hw;
			J->JabF[1][idx] = hw;
			
			// dIdx&dIdx hessian block
			JIdxJIdx_00+=hitColor[1]*hitColor[1];
			JIdxJIdx_11+=hitColor[2]*hitColor[2];
			JIdxJIdx_10+=hitColor[1]*hitColor[2];
			// dIdx&dIdab hessian block
			JabJIdx_00+= drdA*hw * hitColor[1];
			JabJIdx_01+= drdA*hw * hitColor[2];
			JabJIdx_10+= hw * hitColor[1];
			JabJIdx_11+= hw * hitColor[2];
			// dIdab&dIdab hessian block
			JabJab_00+= drdA*drdA*hw*hw;
			JabJab_01+= drdA*hw*hw;
			JabJab_11+= hw*hw;


			wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]); // 梯度平方

			if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;
			if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;
		}
	}

	// 都是对host到target之间的变化量导数
	J->JIdx2(0,0) = JIdxJIdx_00;
	J->JIdx2(0,1) = JIdxJIdx_10;
	J->JIdx2(1,0) = JIdxJIdx_10;
	J->JIdx2(1,1) = JIdxJIdx_11;
	J->JabJIdx(0,0) = JabJIdx_00;
	J->JabJIdx(0,1) = JabJIdx_01;
	J->JabJIdx(1,0) = JabJIdx_10;
	J->JabJIdx(1,1) = JabJIdx_11;
	J->Jab2(0,0) = JabJab_00;
	J->Jab2(0,1) = JabJab_01;
	J->Jab2(1,0) = JabJab_01;
	J->Jab2(1,1) = JabJab_11;

	state_NewEnergyWithOutlier = energyLeft;
	
	// 大于阈值则视为外点
	if(energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2)
	{
		energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
		state_NewState = ResState::OUTLIER;
	}
	else
	{
		state_NewState = ResState::IN;
	}

	state_NewEnergy = energyLeft;

	return energyLeft;
}

void PointFrameResidual::debugPlot()
{
	if(state_state==ResState::OOB) return;
	Vec3b cT = Vec3b(0,0,0);

	if(freeDebugParam5==0)
	{
		float rT = 20*sqrt(state_energy/9);
		if(rT<0) rT=0; if(rT>255)rT=255;
		cT = Vec3b(0,255-rT,rT);
	}
	else
	{
		if(state_state == ResState::IN) cT = Vec3b(255,0,0);
		else if(state_state == ResState::OOB) cT = Vec3b(255,255,0);
		else if(state_state == ResState::OUTLIER) cT = Vec3b(0,0,255);
		else cT = Vec3b(255,255,255);
	}

	for(int i=0;i<patternNum;i++)
	{
		// 2022.1.11暂时不用这个功能
		/*
		if((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0]-3 && projectedTo[i][1] < hG[0]-3 && maskG[target->_cam_idx][0][(int)projectedTo[i][0]+(int)projectedTo[i][1]*wG[0]]))
			target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1],cT);*/
		// 2022.1.11
	}
}


// XTL：若残差项的state_state为OOB，判断isActiveAndIsGoodNEW为false，直接返回
// XTL：若残差项的state_NewState为IN，isActiveAndIsGoodNEW置为true，并且交换对应的efResidual的J与残差项的J，然后计算JpJdF，否则isActiveAndIsGoodNEW置为false
// XTL：更新状态、能量
void PointFrameResidual::applyRes(bool copyJacobians)
{
	if(copyJacobians)
	{
		if(state_state == ResState::OOB)
		{
			assert(!efResidual->isActiveAndIsGoodNEW);
			return;	// can never go back from OOB
		}
		if(state_NewState == ResState::IN)// && )
		{
			efResidual->isActiveAndIsGoodNEW=true;
			
			efResidual->takeDataF();  // 从当前取jacobian数据
		}
		else
		{
			efResidual->isActiveAndIsGoodNEW=false;
		}
	}

	setState(state_NewState);
	state_energy = state_NewEnergy;
}


}
