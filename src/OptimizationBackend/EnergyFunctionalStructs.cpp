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


#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


// XTL：交换EFResidual的J和PointFrameResidual的J，计算EFResidual的JpJdF
void EFResidual::takeDataF()
{
	std::swap<RawResidualJacobian*>(J, data->J);

	Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;

	for(int i=0;i<6;i++)
		JpJdF[i] = J->Jpdxi[0][i]*JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];

	JpJdF.segment<2>(6) = J->JabJIdx*J->Jpdd;
}


// XTL：创建EFFrame时调用(每次创建关键帧时才会创建)，给位姿（仅第一帧）和光度参数设置prior，同时设置efframe的delta
void EFFrame::takeData()
{
	prior = data->getPrior().head<8>(); 	// 得到先验状态, 主要是光度仿射变换
	delta = data->get_state_minus_stateZero().head<8>();
	delta_prior =  (data->get_state() - data->getPriorZero()).head<8>();


	assert(data->frameID != -1);

	frameID = data->frameID;  // 所有帧的ID序号
}



// XTL：创建EFPoint时调用
void EFPoint::takeData()
{
	priorF = data->hasDepthPrior ? setting_idepthFixPrior*SCALE_IDEPTH*SCALE_IDEPTH : 0;
	if(setting_solverMode & SOLVER_REMOVE_POSEPRIOR) priorF=0;
	//in fact 这个priorF 只在初始化时候有， 其他时候统一set to zero
	deltaF = data->idepth - data->idepth_zero; // 当前状态逆深度减去线性化处
}

//@ 计算线性化更新后的残差
// XTL：固定进一步优化时，DeltaX造成的残差项的变化，保留在res_toZeroF当中 res_toZeroF = resF+delta_r
void EFResidual::fixLinearizationF(EnergyFunctional* ef)
{
	Mat18f dp = ef->adHTdeltaF[hostIDX+ef->nFrames*targetIDX]; // 得到hostIDX --> targetIDX的状态增量

	// compute Jp*delta
	__m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>())
								   +J->Jpdc[0].dot(ef->cDeltaF)
								   +J->Jpdd[0]*point->deltaF);
	__m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>())
								   +J->Jpdc[1].dot(ef->cDeltaF)
								   +J->Jpdd[1]*point->deltaF);

	__m128 delta_a = _mm_set1_ps((float)(dp[6]));
	__m128 delta_b = _mm_set1_ps((float)(dp[7]));

	for(int i=0;i<patternNum;i+=4)
	{
		// PATTERN: rtz = resF - [JI*Jp Ja]*delta.
		__m128 rtz = _mm_load_ps(((float*)&J->resF)+i); // 光度残差
		rtz = _mm_sub_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(J->JIdx))+i),Jp_delta_x));
		rtz = _mm_sub_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(J->JIdx+1))+i),Jp_delta_y));
		rtz = _mm_sub_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(J->JabF))+i),delta_a));
		rtz = _mm_sub_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(J->JabF+1))+i),delta_b));
		_mm_store_ps(((float*)&res_toZeroF)+i, rtz); // 存储在res_toZeroF
	}

	isLinearized = true;
}

}
