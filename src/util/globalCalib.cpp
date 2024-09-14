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



#include "util/globalCalib.h"
#include "stdio.h"
#include <iostream>

//! 后面带G的是global变量
namespace dso
{
	int wG[PYR_LEVELS], hG[PYR_LEVELS];
	float powG[PYR_LEVELS] = {1,2,4,8,16,32};

	bool firstG;
	float weightC[CAM_NUM];
	float lastRes;

	int _left=-1;
	int _right=10000;
	int _top=-1;
	int _bottom=10000;

	float fxG[CAM_NUM][PYR_LEVELS], fyG[CAM_NUM][PYR_LEVELS],
		  cxG[CAM_NUM][PYR_LEVELS], cyG[CAM_NUM][PYR_LEVELS];

	float fxiG[CAM_NUM][PYR_LEVELS], fyiG[CAM_NUM][PYR_LEVELS],
		  cxiG[CAM_NUM][PYR_LEVELS], cyiG[CAM_NUM][PYR_LEVELS];

	Eigen::Matrix3f KG[CAM_NUM][PYR_LEVELS], KiG[CAM_NUM][PYR_LEVELS];


	std::vector<SE3> T_lb_c(CAM_NUM);

	std::vector<SE3> T_c0_c(CAM_NUM);

	std::vector<SE3> T_c_c0(CAM_NUM);

	bool* maskG[CAM_NUM][PYR_LEVELS];

	float wM3G; // w-3 global
	float hM3G;

	int ids[CAM_NUM+2];

	Eigen::Matrix<unsigned char,3,1> LUT[256];


	void draw(cv::Mat& m, int Ku, int Kv, cv::Vec3b color, int margin)
	{
		for(int i = Ku-margin; i <= Ku+margin; i++)
		{
			for(int j = Kv-margin; j <= Kv+margin; j++)
			{
				// m.at<uchar>(j, i) = color[0];
				m.at<cv::Vec3b>(j, i) = color;
			}
		}
		/*
		for(int i = Ku-2; i <= Ku+2; i++)
		{
			m.at<cv::Vec3b>(Kv-2, i) = cv::Vec3b(0,0,0);
			m.at<cv::Vec3b>(Kv+2, i) = cv::Vec3b(0,0,0);
		}

		for(int j = Kv-1; j <= Kv+1; j++)
		{
			m.at<cv::Vec3b>(j, Ku-2) = cv::Vec3b(0,0,0);
			m.at<cv::Vec3b>(j, Ku+2) = cv::Vec3b(0,0,0);
		}
		*/
	}

	void setLUT(){
		int s;
		for (s = 0; s < 32; s++) {
			LUT[s][0] = 128 + 4 * s;
			LUT[s][1] = 0;
			LUT[s][2] = 0;
		}
		LUT[32][0] = 255;
		LUT[32][1] = 0;
		LUT[32][2] = 0;
		for (s = 0; s < 63; s++) {
			LUT[33+s][0] = 255;
			LUT[33+s][1] = 4+4*s;
			LUT[33+s][2] = 0;
		}
		LUT[96][0] = 254;
		LUT[96][1] = 255;
		LUT[96][2] = 2;
		for (s = 0; s < 62; s++) {
			LUT[97 + s][0] = 250 - 4 * s;
			LUT[97 + s][1] = 255;
			LUT[97 + s][2] = 6+4*s;
		}
		LUT[159][0] = 1;
		LUT[159][1] = 255;
		LUT[159][2] = 254;
		for (s = 0; s < 64; s++) {
			LUT[160 + s][0] = 0;
			LUT[160 + s][1] = 252 - (s * 4);
			LUT[160 + s][2] = 255;
		}
		for (s = 0; s < 32; s++) {
			LUT[224 + s][0] = 0;
			LUT[224 + s][1] = 0;
			LUT[224 + s][2] = 252-4*s;
		}
	}
	
	void setGlobalCalib(int w, int h,const Eigen::Matrix3f &K, int cam_idx, bool* mask)
	{
		// XTL nclt的cam顺序是0,1,2,3,4
		// XTL 1->0,2->1,3->2
		for(int i=0;i<cam_num;i++)
			ids[i+1] = i;
		// XTL 0->2,4->0
		ids[0] = cam_num-1;
		ids[cam_num+1] = 0;

		maskG[cam_idx][0] = new bool[w*h];
		for(int i=0;i<w*h;i++){
			maskG[cam_idx][0][i] = mask[i];
		}
		// for(int i=0;i<setting_margin;i++){
		// 	if(x+y*w-i>=0)
		// 		mask[x+y*w-i] = false;
		// 	if(x+y*w+i<w*h)
		// 		mask[x+y*w+i] = false;
		// 	if(x+(y-i)*w>=0)
		// 		mask[x+(y-i)*w] = false;
		// 	if(x+(y+i)*w<w*h)
		// 		mask[x+(y+i)*w] = false;						
		// }		
		// for(int i=0;i<cam_num;i++)
		// 	weightC[i]=1;
		firstG = false;
		lastRes = 0;

		int wlvl=w;
		int hlvl=h;
		pyrLevelsUsed=1;
		while(wlvl%2==0 && hlvl%2==0 && wlvl*hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)
		{
			wlvl /=2;
			hlvl /=2;
			pyrLevelsUsed++;
		}
		printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
				pyrLevelsUsed-1, wlvl, hlvl);
		if(wlvl>100 && hlvl > 100)
		{
			printf("\n\n===============WARNING!===================\n "
					"using not enough pyramid levels.\n"
					"Consider scaling to a resolution that is a multiple of a power of 2.\n");
		}
		if(pyrLevelsUsed < 3)
		{
			printf("\n\n===============WARNING!===================\n "
					"I need higher resolution.\n"
					"I will probably segfault.\n");
		}

		wM3G = w-3;
		hM3G = h-3;

		wG[0] = w;
		hG[0] = h;

		KG[cam_idx][0] = K;
		fxG[cam_idx][0] = K(0,0);
		fyG[cam_idx][0] = K(1,1);
		cxG[cam_idx][0] = K(0,2);
		cyG[cam_idx][0] = K(1,2);
		KiG[cam_idx][0] = KG[cam_idx][0].inverse();
		fxiG[cam_idx][0] = KiG[cam_idx][0](0,0);
		fyiG[cam_idx][0] = KiG[cam_idx][0](1,1);
		cxiG[cam_idx][0] = KiG[cam_idx][0](0,2);
		cyiG[cam_idx][0] = KiG[cam_idx][0](1,2);

		for (int level = 1; level < pyrLevelsUsed; ++ level)
		{
			wG[level] = w >> level;
			hG[level] = h >> level;

			fxG[cam_idx][level] = fxG[cam_idx][level-1] * 0.5;
			fyG[cam_idx][level] = fyG[cam_idx][level-1] * 0.5;
			cxG[cam_idx][level] = (cxG[cam_idx][0] + 0.5) / ((int)1<<level) - 0.5;
			cyG[cam_idx][level] = (cyG[cam_idx][0] + 0.5) / ((int)1<<level) - 0.5;

			KG[cam_idx][level]  << fxG[cam_idx][level], 0.0, cxG[cam_idx][level], 0.0, fyG[cam_idx][level], cyG[cam_idx][level], 0.0, 0.0, 1.0;	// synthetic
			KiG[cam_idx][level] = KG[cam_idx][level].inverse();

			fxiG[cam_idx][level] = KiG[cam_idx][level](0,0);
			fyiG[cam_idx][level] = KiG[cam_idx][level](1,1);
			cxiG[cam_idx][level] = KiG[cam_idx][level](0,2);
			cyiG[cam_idx][level] = KiG[cam_idx][level](1,2);

			// 4合1, 生成金字塔
			maskG[cam_idx][level] = new bool[wG[level]*hG[level]];
			bool* mask_ = maskG[cam_idx][level];
			bool* mask_l = maskG[cam_idx][level-1];
			int wlm1 = wG[level-1]; // 列数
			for(int y=0;y<hG[level];y++)
				for(int x=0;x<wG[level];x++)
				{
					int cnt = 4;
					if(!mask_l[2*x + 2*y*wlm1]){
						cnt --;
					}
					if(!mask_l[2*x+1 + 2*y*wlm1]){
						cnt --;
					}
					if(!mask_l[2*x + 2*y*wlm1+wlm1]){
						cnt --;
					}
					if(!mask_l[2*x+1 + 2*y*wlm1+wlm1]){
						cnt --;
					}
					if(cnt < 4){
						mask_[x + y*wG[level]] = false ;
					}
					else{
						mask_[x + y*wG[level]] = true ;
					}
				}
		}
	}

	Eigen::Matrix4d xyzrpy2T(float ssc[6]){

		float sr = sin(M_PI/180.0 * ssc[3]);
		float cr = cos(M_PI/180.0 * ssc[3]);

		float sp = sin(M_PI/180.0 * ssc[4]);
		float cp = cos(M_PI/180.0 * ssc[4]);

		float sh = sin(M_PI/180.0 * ssc[5]);
		float ch = cos(M_PI/180.0 * ssc[5]);

		Eigen::Vector3d t;
		t << ssc[0], ssc[1], ssc[2];
		Eigen::Matrix3d R;
		R << ch*cp, -sh*cr+ch*sp*sr, sh*sr+ch*sp*cr, sh*cp, ch*cr+sh*sp*sr, -ch*sr+sh*sp*cr, -sp, cp*sr, cp*cr;

		Eigen::Matrix4d T;
		T.setIdentity();

		T.block<3,3>(0,0) = R;
		T.topRightCorner<3, 1>() = t;

		return T;
	}

	void setGlobalExtrin(float ssc[6], int cam_idx){
		Eigen::Matrix4d T = xyzrpy2T(ssc);
		T_lb_c[cam_idx] = SE3(T.block<3,3>(0,0),T.topRightCorner<3, 1>());
	}

	void setGlobalExtrin_ijrr(Eigen::Matrix4d T_c_l, int cam_idx){
		T_lb_c[cam_idx] = SE3(T_c_l.block<3,3>(0,0),T_c_l.topRightCorner<3, 1>()).inverse();
	}

	void calGlobalExtrin(){
		for(int i=0;i<cam_num;i++){
			T_c0_c[i] = T_lb_c[0].inverse()*T_lb_c[i];
			T_c_c0[i] = T_c0_c[i].inverse();
		}
	}
}
