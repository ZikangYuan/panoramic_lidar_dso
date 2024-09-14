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



#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"
#include "FullSystem/ResidualProjections.h"

namespace dso
{
//! 这里u_ v_ 是加了0.5的
ImmaturePoint::ImmaturePoint(int u_, int v_, frame_hessian* host_, float type, CalibHessian* HCalib)
: u(u_), v(v_), host(host_), my_type(type), idepth_min(0), idepth_max(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{

	gradH.setZero();

	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		// 由于+0.5导致积分, 插值得到值3个 [像素值, dx, dy]
        Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy,wG[0]); 

		color[idx] = ptc[0];
		if(!std::isfinite(color[idx])) {energyTH=NAN; return;}

		// 梯度矩阵[dx*2, dxdy; dydx, dy^2]
		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();
		//! 点的权重 c^2 / ( c^2 + ||grad||^2 )
		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}

	energyTH = patternNum*setting_outlierTH;
	energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

	idepth_GT=0;
	quality=10000;
	lastTracePixelInterval = 0; // 2020.07.18 yzk shiyong
}

ImmaturePoint::~ImmaturePoint()
{
}


void ImmaturePoint::resetPosition(float _u, float _v)
{
	gradH.setZero();
	u = _u;
	v = _v;
	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		// 由于+0.5导致积分, 插值得到值3个 [像素值, dx, dy]
        Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy,wG[0]); 

		color[idx] = ptc[0];
		if(!std::isfinite(color[idx])) {energyTH=NAN; return;}

		// 梯度矩阵[dx*2, dxdy; dydx, dy^2]
		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();
		//! 点的权重 c^2 / ( c^2 + ||grad||^2 )
		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}
}

/* 
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
 //@ 使用深度滤波对未成熟点进行深度估计
ImmaturePointStatus ImmaturePoint::traceOn(frame_hessian* frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine, CalibHessian* HCalib, bool debugPrint)
{
	if(lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;
	// 2021.11.26
	if(isFromSensor){
		lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
		return lastTraceStatus;
	}
	// 2021.11.26

	debugPrint = false;//rand()%100==0;
	float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;  // 极限搜索的最大长度

	if(debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, idepth_max,
				hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

	//	const float stepsize = 1.0;				// stepsize for initial discrete search.
	//	const int GNIterations = 3;				// max # GN iterations
	//	const float GNThreshold = 0.1;				// GN stop after this stepsize.
	//	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
	//	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
	//	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.
	
	// ============== project min and max. return if one of them is OOB ===================
//[ ***step 1*** ] 计算出来搜索的上下限, 对应idepth_max, idepth_min
	Vec3f pr = hostToFrame_KRKi * Vec3f(u,v, 1);
	Vec3f ptpMin = pr + hostToFrame_Kt*idepth_min;
	float uMin = ptpMin[0] / ptpMin[2];
	float vMin = ptpMin[1] / ptpMin[2];

	// 如果超出图像范围则设为 OOB
	// 2021.11.24
	if(!(uMin > 4 && vMin > 4 && uMin < wG[0]-5 && vMin < hG[0]-5 && maskG[frame->_cam_idx][0][(int)uMin+(int)vMin*wG[0]]))
	{
		if(debugPrint) printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
				u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}
	// 2021.11.24

	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;
	if(std::isfinite(idepth_max))
	{
		ptpMax = pr + hostToFrame_Kt*idepth_max;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// 2021.11.24
		if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5 && maskG[frame->_cam_idx][0][(int)uMax+(int)vMax*wG[0]]))
		{
			if(debugPrint) printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		// 2021.11.24

		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
		dist = sqrtf(dist);
		//* 搜索的范围太小
		if(dist < setting_trace_slackInterval)
		{
			if(debugPrint)
				printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

			lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;  // 直接设为中值
			lastTracePixelInterval=dist;
			return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED; //跳过
		}
		assert(dist>0);
	}
	else
	{
		//* 上限无穷大, 则设为最大值
		dist = maxPixSearch;

		// project to arbitrary depth to get direction.
		ptpMax = pr + hostToFrame_Kt*0.01;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// direction.
		float dx = uMax-uMin;
		float dy = vMax-vMin;
		float d = 1.0f / sqrtf(dx*dx+dy*dy);

		//* 根据比例得到最大值
		// set to [setting_maxPixSearch].
		uMax = uMin + dist*dx*d;
		vMax = vMin + dist*dy*d;

		// may still be out!
		// 2021.11.24
		if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5 && maskG[frame->_cam_idx][0][(int)uMax+(int)vMax*wG[0]]))
		{
			if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		// 2021.11.24
		assert(dist>0);
	}

	//? 为什么是这个值呢??? 0.75 - 1.5 
	// 这个值是两个帧上深度的比值, 它的变化太大就是前后尺度变化太大了
	// set OOB if scale change too big.
	if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
	{
		if(debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

//[ ***step 2*** ] 计算误差大小(图像梯度和极线夹角大小), 夹角大, 小的几何误差会有很大影响
	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	float dx = setting_trace_stepsize*(uMax-uMin);
	float dy = setting_trace_stepsize*(vMax-vMin);

	//! (dIx*dx + dIy*dy)^2
	float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy)); 
	//! (dIx*dy - dIy*dx)^2
	float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx)); // (dx, dy)垂直方向的乘积
	// 计算的是极线方向和梯度方向的夹角大小，90度则a=0, errorInPixel变大；平行时候b=0
	float errorInPixel = 0.2f + 0.2f * (a+b) / a; // 没有使用LSD的方法, 估计是能有效防止位移小的情况

	//* errorInPixel大说明垂直, 这时误差会很大, 视为bad
	if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))
	{
		if(debugPrint)
			printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
		lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
		lastTracePixelInterval=dist;
		return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
	}

	if(errorInPixel >10) errorInPixel=10;



	// ============== do the discrete search ===================
//[ ***step 3*** ] 在极线上找到最小的光度误差的位置, 并计算和第二次的比值作为质量
	dx /= dist; // cos
	dy /= dist;	// sin

	if(debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, uMin, vMin,
				idepth_max, uMax, vMax,
				errorInPixel
				);


	if(dist>maxPixSearch)
	{
		uMax = uMin + maxPixSearch*dx;
		vMax = vMin + maxPixSearch*dy;
		dist = maxPixSearch;
	}

	int numSteps = 1.9999f + dist / setting_trace_stepsize; // 步数
	Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();

	float randShift = uMin*1000-floorf(uMin*1000); // 	取小数点后面的做随机数??
	float ptx = uMin-randShift*dx;
	float pty = vMin-randShift*dy;

	//* pattern在新的帧上的偏移量
	Vec2f rotatetPattern[MAX_RES_PER_POINT];
	for(int idx=0;idx<patternNum;idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);



	// 这个判断太多了, 学习学习, 全面考虑
	if(!std::isfinite(dx) || !std::isfinite(dy))
	{
		//printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}


	//* 沿着级线搜索误差最小的位置
	float errors[100];
	float bestU=0, bestV=0, bestEnergy=1e10;
	int bestIdx=-1;
	if(numSteps >= 100) numSteps = 99;

	for(int i=0;i<numSteps;i++)
	{
		float energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			float hitColor = getInterpolatedElement31(frame->dI,
										(float)(ptx+rotatetPattern[idx][0]),
										(float)(pty+rotatetPattern[idx][1]),
										wG[0]);

			if(!std::isfinite(hitColor)) {energy+=1e5; continue;}
			float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);
		}

		if(debugPrint)
			printf("step %.1f %.1f (id %f): energy = %f!\n",
					ptx, pty, 0.0f, energy);


		errors[i] = energy;
		if(energy < bestEnergy)
		{
			bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
		}

		// 每次走1 dist对应大小
		ptx+=dx; 
		pty+=dy;
	}

	//* 在一定的半径内找最到误差第二小的, 差的足够大, 才更好(这个常用)
	// find best score outside a +-2px radius.
	float secondBest=1e10;
	for(int i=0;i<numSteps;i++)
	{
		if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i];
	}
	float newQuality = secondBest / bestEnergy;
	if(newQuality < quality || numSteps > 10) quality = newQuality;

//[ ***step 4*** ] 在上面的最优位置进行线性搜索, 进行求精
	// ============== do GN optimization ===================
	float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
	if(setting_trace_GNIterations>0) bestEnergy = 1e5;
	int gnStepsGood=0, gnStepsBad=0;
	for(int it=0;it<setting_trace_GNIterations;it++)
	{
		float H = 1, b=0, energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			Vec3f hitColor = getInterpolatedElement33(frame->dI,
					(float)(bestU+rotatetPattern[idx][0]),
					(float)(bestV+rotatetPattern[idx][1]),wG[0]);

			if(!std::isfinite((float)hitColor[0])) {energy+=1e5; continue;}
			float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float dResdDist = dx*hitColor[1] + dy*hitColor[2]; // 极线方向梯度
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			H += hw*dResdDist*dResdDist;
			b += hw*residual*dResdDist;
			energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
		}


		if(energy > bestEnergy)
		{
			gnStepsBad++;

			// do a smaller step from old point.
			stepBack*=0.5;  		//* 减小步长再进行计算
			bestU = uBak + stepBack*dx;
			bestV = vBak + stepBack*dy;
			if(debugPrint)
				printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, stepBack,
						uBak, vBak, bestU, bestV);
		}
		else
		{
			gnStepsGood++;

			float step = -gnstepsize*b/H;
			//* 步长最大才0.5
			if(step < -0.5) step = -0.5;
			else if(step > 0.5) step=0.5;

			if(!std::isfinite(step)) step=0;

			uBak=bestU; // 备份
			vBak=bestV;
			stepBack=step;

			bestU += step*dx;
			bestV += step*dy;
			bestEnergy = energy;

			if(debugPrint)
				printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, step,
						uBak, vBak, bestU, bestV);
		}

		if(fabsf(stepBack) < setting_trace_GNThreshold) break;
	}


	// ============== detect energy-based outlier. ===================
	//	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
	//	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
	//	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
	//* 残差太大, 则设置为外点
	if(!(bestEnergy < energyTH*setting_trace_extraSlackOnTH))
	//			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
	//		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
	//			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
	{
		if(debugPrint)
			printf("OUTLIER!\n");

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)   
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;   //? 外点还有机会变回来???
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

//[ ***step 5*** ] 根据得到的最优位置重新计算逆深度的范围
	// ============== set new interval ===================
	//! u = (pr[0] + Kt[0]*idepth) / (pr[2] + Kt[2]*idepth) ==> idepth = (u*pr[2] - pr[0]) / (Kt[0] - u*Kt[2])
	//! v = (pr[1] + Kt[1]*idepth) / (pr[2] + Kt[2]*idepth) ==> idepth = (v*pr[2] - pr[1]) / (Kt[1] - v*Kt[2])
	//* 取误差最大的
	if(dx*dx>dy*dy)
	{
		idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
		idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
	}
	else
	{
		idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));
		idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
	}
	if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);


	if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
	{
		//printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	lastTracePixelInterval=2*errorInPixel; 	// 搜索的范围
	lastTraceUV = Vec2f(bestU, bestV);		// 上一次得到的最有位置
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD; 	//上一次的位置
}


float ImmaturePoint::getdPixdd(
		CalibHessian *  HCalib,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->fh0->targetPrecalc[tmpRes->target->fh0->idx]);// 2022.1.11
	const Vec3f &PRE_tTll = precalc->PRE_tTll[0];
	float drescale, u=0, v=0, new_idepth;
	float Ku, Kv;
	Vec3f KliP;

	projectPoint(this->u,this->v, idepth, 0, 0,HCalib,
			precalc->PRE_RTll[0],PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth, 0, 0);

	float dxdd = (PRE_tTll[0]-PRE_tTll[2]*u)*HCalib->fxl(0);
	float dydd = (PRE_tTll[1]-PRE_tTll[2]*v)*HCalib->fyl(0);
	return drescale*sqrtf(dxdd*dxdd + dydd*dydd);
}


float ImmaturePoint::calcResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->fh0->targetPrecalc[tmpRes->target->fh0->idx]);// 2022.1.11

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll[0];
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll[0];
	// 2021.12.18
	Vec2f affLL = precalc->PRE_aff_mode[0];
	//

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
		if(!projectPoint(this->u+patternP[idx][0], this->v+patternP[idx][1], idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv, 0)) // 2021.12.29
			{return 1e10;}

		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
		if(!std::isfinite((float)hitColor[0])) {return 1e10;}
		//if(benchmarkSpecialOption==5) hitColor = (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));

		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
	}

	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
	}
	return energyLeft;
}


// XTL：将未成熟残差项中的未成熟点投影至目标帧，计算残差，H矩阵，b，优化逆深度需要
// XTL：投影超出边界或投影到的颜色为无穷：设置state_NewState为OOB;残差过大：OUTLIER;
// XTL：设置state_NewEnergy为计算出来的残差
// XTL：只有非雷达点会用该函数处理
double ImmaturePoint::linearizeResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float &Hdd, float &bd,
		float idepth)
{
	if(tmpRes->state_state == ResState::OOB)
		{ tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }

	// xtl 2022.1.30
	int idx_i = host->_cam_idx;
	int idx_j = tmpRes->target->_cam_idx;
	// xtl 2022.1.30

	// 2021.11.16
	FrameFramePrecalc* precalc = &(host->fh0->targetPrecalc[tmpRes->target->fh0->idx]);
	// 2021.11.16

	// check OOB due to scale angle change.

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_RTll = precalc->PRE_RTll[idx_i*cam_num+idx_j];
	const Vec3f &PRE_tTll = precalc->PRE_tTll[idx_i*cam_num+idx_j];

	Vec2f affLL = precalc->PRE_aff_mode[idx_i*cam_num+idx_j]; // 2021.12.18

	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,
				PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth, idx_i, idx_j))
			{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}


		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

		if(!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

		// depth derivatives.
		// xtl 2022.1.30
		float dxInterp = hitColor[1]*HCalib->fxl(idx_j);
		float dyInterp = hitColor[2]*HCalib->fyl(idx_j);
		// xtl 2022.1.30
		float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale); // 对逆深度的导数

		hw *= weights[idx]*weights[idx];

		Hdd += (hw*d_idepth)*d_idepth; // 对逆深度的hessian
		bd += (hw*residual)*d_idepth; // 对逆深度的Jres
	}


	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
	}

	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}


double ImmaturePoint::linearizeResPixel(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		cv::Mat &Hdd, cv::Mat &bd, int level_ref,
		float idepth)
{
	if(tmpRes->state_state == ResState::OOB)
		{ tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }
	int idx_i = host->_cam_idx;
	int idx_j = tmpRes->target->_cam_idx;

	FrameFramePrecalc* precalc = &(host->fh0->targetPrecalc[tmpRes->target->fh0->idx]);

	// check OOB due to scale angle change.
	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_RTll = precalc->PRE_RTll[idx_i*cam_num+idx_j];
	const Vec3f &PRE_tTll = precalc->PRE_tTll[idx_i*cam_num+idx_j];

	Vec2f affLL = precalc->PRE_aff_mode[idx_i*cam_num+idx_j]; // 2021.12.18

	float mean_diff = 0;
	for(int idx=0;idx<patternNum;idx++)
	{
        cv::Mat matA(1, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, 1, CV_32F, cv::Scalar::all(0));
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,
				PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth, idx_i, idx_j))
			{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}


		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

		if(!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]) + mean_diff;

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

		// depth derivatives.
		// xtl 2022.1.30
		float derivex = hitColor[1];
		float derivey = hitColor[2];
		Vec3f ptc = getInterpolatedElement33BiLin(host->dI, this->u+dx, this->v+dy,wG[0]); 
		float derivex1 = ptc[1];
		float derivey1 = ptc[2];
		float fx = HCalib->fxl(idx_j);
		float fy = HCalib->fyl(idx_j);
		// xtl 2022.1.30
		Vec2f d_uv = derive_uv(PRE_RTll, u, v, fx, fy, derivex, derivey, derivex1, derivey1, drescale); // 对逆深度的导数
		matA.at<float>(0,0) = d_uv[0];
		matA.at<float>(0,1) = d_uv[1];
		matA.at<float>(0,2) = 1;
		matAt.at<float>(0,0) = d_uv[0];
		matAt.at<float>(1,0) = d_uv[1];
		matAt.at<float>(2,0) = 1;

		hw *= weights[idx]*weights[idx];

		Hdd += (hw*matAt)*matA; // 对u,v的hessian
		bd -= (hw*residual)*matAt; // 对u,v的Jres
	}


	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
	}

	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}

double ImmaturePoint::ResPixel(
		const float outlierTHSlack,
		FrameHessian* target, float u, float v,
		cv::Mat &Hdd, cv::Mat &bd, int level,
		float mean_diff,
		float idepth)
{
	int idx = host->_cam_idx;

	FrameFramePrecalc* precalc = &(host->fh0->targetPrecalc[target->idx]);

    Vec2f px_scaled(u,v);
	// check OOB due to scale angle change.
	float energyLeft=0;
	const Eigen::Vector3f* dIl = target->frame[idx]->dIp[level];
	const Mat33f &PRE_RTll = precalc->PRE_RTll[idx*cam_num+idx];
	const Vec3f &PRE_tTll = precalc->PRE_tTll[idx*cam_num+idx];

	int patch_size_ = 8;
	int halfpatch_size_ = 4;
	int dim = 2;
    for(int y=0; y<patch_size_; ++y)
    {
		int dy = y - halfpatch_size_;
		for(int x=0; x<patch_size_; ++x)
		{
			cv::Mat matA(1, dim, CV_32F, cv::Scalar::all(0));
			cv::Mat matAt(dim, 1, CV_32F, cv::Scalar::all(0));
			int dx = x - halfpatch_size_;
			Vec3f ref = getInterpolatedElement33(host->dIp[level], px_scaled[0]+dx, px_scaled[1]+dy,wG[level]); 
			float drescale, u, v, new_idepth;
			float Ku, Kv;
			Vec3f KliP;
			Vec3f hit;
			if(!projectPoint(px_scaled[0],px_scaled[1], idepth, level, dx, dy,
					PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth, idx))
			{
				hit.setZero();
			}
			else{
				hit = getInterpolatedElement33(dIl, Ku, Kv,wG[level]);
			}
			float residual = hit[0] - ref[0] /*+ mean_diff*/;
			float hw = 1/*fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual)*/;

			//! 点的权重 c^2 / ( c^2 + ||grad||^2 )
			// float weights = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ref.tail<2>().squaredNorm()));
			float weights = 1;
			energyLeft += weights*weights*hw *residual*residual*(2-hw);
			
			float derivex = hit[1];
			float derivey = hit[2];
			float derivex1 = ref[1];
			float derivey1 = ref[2];
			float fx = fxG[idx][level];
			float fy = fyG[idx][level];
			Vec2f d_uv = derive_uv(PRE_RTll, u, v, fx, fy, derivex, derivey, derivex1, derivey1, drescale); // 对逆深度的导数

			matA.at<float>(0,0) = d_uv[0];
			matA.at<float>(0,1) = d_uv[1];
			// matA.at<float>(0,2) = 0;
			matAt.at<float>(0,0) = d_uv[0];
			matAt.at<float>(1,0) = d_uv[1];
			// matAt.at<float>(2,0) = 0;

			hw *= weights*weights;
			Hdd += (hw*matAt)*matA; // 对u,v的hessian
			bd -= (hw*residual)*matAt; // 对u,v的Jres
		}
	}

	if(energyLeft > 64*setting_outlierTH*outlierTHSlack)
	{
		energyLeft = 64*setting_outlierTH*outlierTHSlack;
	}

	return energyLeft;
}
void ImmaturePoint::getWarpMatrixAffine(
    const double idepth,
	const Mat33f &KRKi, const Vec3f &Kt,
    Mat22f& A_cur_ref)
{
	Vec2 px_ref(u,v);
	// Compute affine warp matrix A_ref_cur
	const int halfpatch_size = 5;
	const Vec3f px_cur(KRKi*Vec3f(px_ref[0],px_ref[1],1)+Kt*idepth);
	const Vec3f px_du(KRKi*Vec3f(px_ref[0]+halfpatch_size,px_ref[1],1)+Kt*idepth);
	const Vec3f px_dv(KRKi*Vec3f(px_ref[0],px_ref[1]+halfpatch_size,1)+Kt*idepth);
	A_cur_ref.col(0) = (px_du.head<2>() - px_cur.head<2>())/halfpatch_size;
	A_cur_ref.col(1) = (px_dv.head<2>() - px_cur.head<2>())/halfpatch_size;
}


int ImmaturePoint::getBestSearchLevel(
    const Mat22f& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

double ImmaturePoint::calcResPixel(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth, float res)
{
	int idx_i = host->_cam_idx;
	int idx_j = tmpRes->target->_cam_idx;

	FrameFramePrecalc* precalc = &(host->fh0->targetPrecalc[tmpRes->target->fh0->idx]);

	// check OOB due to scale angle change.
	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_RTll = precalc->PRE_RTll[idx_i*cam_num+idx_j];
	const Vec3f &PRE_tTll = precalc->PRE_tTll[idx_i*cam_num+idx_j];

	Vec2f affLL = precalc->PRE_aff_mode[idx_i*cam_num+idx_j]; // 2021.12.18

	for(int dx=-1;dx<2;dx++)
	{
		for(int dy=-1;dy<2;dy++)
		{
			float drescale, u, v, new_idepth;
			float Ku, Kv;
			Vec3f KliP;

			if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,
					PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth, idx_i, idx_j))
			{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}

			Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

			Vec3f ptc = getInterpolatedElement33BiLin(host->dI, this->u+dx, this->v+dy,wG[0]); 

			if(!std::isfinite((float)hitColor[0]))	{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
			float residual = hitColor[0] - (affLL[0] * ptc[0] + affLL[1]);

			// float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
			energyLeft += fabsf(residual);
			if(energyLeft >= res)
				return 1.5*res;
		}
	}
	tmpRes->state_NewState = ResState::IN;
	tmpRes->state_NewEnergy = energyLeft;

	return energyLeft;
}

}