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
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "util/settings.h"

namespace dso
{

//@ 返回逆深度的导数值
EIGEN_STRONG_INLINE float derive_idepth(
		const Vec3f &t, const float &u, const float &v,
		const int &dx, const int &dy, const float &dxInterp,
		const float &dyInterp, const float &drescale)
{
	return (dxInterp*drescale * (t[0]-t[2]*u)
			+ dyInterp*drescale * (t[1]-t[2]*v))*SCALE_IDEPTH;
}

//@ 返回u,v的导数值
EIGEN_STRONG_INLINE Vec2f derive_uv(
		const Mat33f &R, const float &u, const float &v,
		const float &fx, const float &fy,
		const float &Ix, const float &Iy, 
		const float &Ix1, const float &Iy1, 
		const float &drescale)
{
	return Vec2f(drescale* (Ix*(R(0,0)-R(2,0)*u) + Iy*fy/fx*(R(1,0)-R(2,0)*v)) - Ix1,
			drescale* (Ix*fx/fy*(R(0,1)-R(2,1)*u) + Iy*(R(1,1)-v*R(2,1))) - Iy1);
}

//@ 把host上的点变换到target上
EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const Mat33f &KRKi, const Vec3f &Kt,
		float &Ku, float &Kv, int cam_idx)
{
	Vec3f ptp = KRKi * Vec3f(u_pt,v_pt, 1) + Kt*idepth; // host上点除深度
	Ku = ptp[0] / ptp[2];
	Kv = ptp[1] / ptp[2];
	// 2021.12.28
	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G && maskG[cam_idx][0][(int)Ku+(int)Kv*wG[0]]; // 不在边缘
}


//@ 将host帧投影到新的帧, 且可以设置像素偏移dxdy, 得到:
//@ 参数: [drescale 新比旧逆深度] [uv 新的归一化平面]
//@		[kukv 新的像素平面] [KliP 旧归一化平面] [new_idepth 新的逆深度]

EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const int &dx, const int &dy,
		CalibHessian* const &HCalib,
		const Mat33f &R, const Vec3f &t,
		float &drescale, float &u, float &v,
		float &Ku, float &Kv, Vec3f &KliP, float &new_idepth,
		int cam_idx_i,int cam_idx_j)// 2021.11.18
{
	// host上归一化平面点
	KliP = Vec3f(
			(u_pt+dx-HCalib->cxl(cam_idx_i))*HCalib->fxli(cam_idx_i),
			(v_pt+dy-HCalib->cyl(cam_idx_i))*HCalib->fyli(cam_idx_i),
			1);

	Vec3f ptp = R * KliP + t*idepth;
	drescale = 1.0f/ptp[2]; 		// target帧逆深度 比 host帧逆深度
	new_idepth = idepth*drescale;	// 新的帧上逆深度

	if(!(drescale>0)) return false;

	// 归一化平面
	u = ptp[0] * drescale;
	v = ptp[1] * drescale;
	// 像素平面
	Ku = u*HCalib->fxl(cam_idx_j) + HCalib->cxl(cam_idx_j);
	Kv = v*HCalib->fyl(cam_idx_j) + HCalib->cyl(cam_idx_j);

	// 2021.11.25
	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G && maskG[cam_idx_j][0][(int)Ku+(int)Kv*wG[0]];
	// 2021.11.25
}

//@ 将host帧投影到新的帧, 且可以设置像素偏移dxdy, 得到:
//@ 参数: [drescale 新比旧逆深度] [uv 新的归一化平面]
//@		[kukv 新的像素平面] [KliP 旧归一化平面] [new_idepth 新的逆深度]

EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth, int lvl,
		const int &dx, const int &dy,
		const Mat33f &R, const Vec3f &t,
		float &drescale, float &u, float &v,
		float &Ku, float &Kv, Vec3f &KliP, float &new_idepth,
		int cam_idx)// 2021.11.18
{
	// host上归一化平面点
	KliP = Vec3f(
			(u_pt+dx-cxG[cam_idx][lvl])/fxG[cam_idx][lvl],
			(v_pt+dy-cyG[cam_idx][lvl])/fyG[cam_idx][lvl],
			1);

	Vec3f ptp = R * KliP + t*idepth;
	drescale = 1.0f/ptp[2]; 		// target帧逆深度 比 host帧逆深度
	new_idepth = idepth*drescale;	// 新的帧上逆深度

	if(!(drescale>0)) return false;

	// 归一化平面
	u = ptp[0] * drescale;
	v = ptp[1] * drescale;
	// 像素平面
	Ku = u*fxG[cam_idx][lvl] + cxG[cam_idx][lvl];
	Kv = v*fyG[cam_idx][lvl] + cyG[cam_idx][lvl];

	// 2021.11.25
	return Ku>1.1f && Kv>1.1f && Ku<(wG[lvl]-2) && Kv<(hG[lvl]-2) && maskG[cam_idx][lvl][(int)Ku+(int)Kv*wG[lvl]];
	// 2021.11.25
}

EIGEN_STRONG_INLINE bool projectPoint(
		const Eigen::Vector3f pt, int lvl,
		Vec3f &pt_lb3,
		CalibHessian* const &HCalib,
		const Mat33f &R_lb3l, const Vec3f &t_lb3l,
		const Mat33f &R_cl, const Vec3f &t_cl,
		const Mat33f &R, const Vec3f &t,
		float &u_src, float &v_src,
		float &u, float &v,
		float &Ku_src, float &Kv_src,
		float &Ku, float &Kv, 
		float &idepth, float &new_idepth,
		int cam_idx)// 2021.11.18
{
	pt_lb3 = R_lb3l * pt + t_lb3l;
	Vec3f p_c = R_cl * pt + t_cl;
	// host上归一化平面点
	u_src = p_c[0]/p_c[2];
	v_src = p_c[1]/p_c[2];
	idepth = 1/p_c[2];

	// 像素平面
	Ku_src = u_src*fxG[cam_idx][lvl] + cxG[cam_idx][lvl];
	Kv_src = v_src*fyG[cam_idx][lvl] + cyG[cam_idx][lvl];

	if(!(Ku_src>1.1f && Kv_src>1.1f && Ku_src<wG[lvl]-3 && Kv_src<hG[lvl]-3 && maskG[cam_idx][lvl][(int)Ku_src+(int)Kv_src*wG[lvl]]))
		return false;

	Vec3f p_c2 = R * p_c + t;

	// target上归一化平面点
	u = p_c2[0]/p_c2[2];
	v = p_c2[1]/p_c2[2];
	new_idepth = 1/p_c2[2];

	if(!(new_idepth>0)) return false;
	
	// 像素平面
	Ku = u*fxG[cam_idx][lvl] + cxG[cam_idx][lvl];
	Kv = v*fyG[cam_idx][lvl] + cyG[cam_idx][lvl];

	return Ku>1.1f && Kv>1.1f && Ku<wG[lvl]-3 && Kv<hG[lvl]-3 && maskG[cam_idx][lvl][(int)Ku+(int)Kv*wG[lvl]];
}


}

