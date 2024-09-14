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


 
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "utility"
#include "iostream"
// 2021.12.29
#include <opencv2/imgproc.hpp>
//

using namespace std;

namespace dso
{

//@ 从ImmaturePoint构造函数, 不成熟点变地图点
PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
	instanceCounter++;
	host = rawPoint->host; // 主帧
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;  //似乎是显示用的

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5); //深度均值
	setPointStatus(PointHessian::INACTIVE);

	int n = patternNum;
	memcpy(color, rawPoint->color, sizeof(float)*n);// 一个点对应8个像素
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	efPoint=0; // 指针=0


}

//@ 释放residual
void PointHessian::release()
{
	for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
	residuals.clear();
}

void frame_hessian::release()
{
	// DELETE POINT
	// DELETE RESIDUAL
	for(unsigned int i=0;i<pointHessians.size();i++) delete pointHessians[i];
	for(unsigned int i=0;i<pointHessiansMarginalized.size();i++) delete pointHessiansMarginalized[i];
	for(unsigned int i=0;i<pointHessiansOut.size();i++) delete pointHessiansOut[i];
	for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];


	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();

	// 2020.10.20 yzk
	std::vector<PointHessian*>().swap(pointHessians);
	std::vector<PointHessian*>().swap(pointHessiansMarginalized);
	std::vector<PointHessian*>().swap(pointHessiansOut);
	std::vector<ImmaturePoint*>().swap(immaturePoints);
	// 2020.10.20 yzk
}

//* 计算各层金字塔图像的像素值和梯度
void frame_hessian::makeImages(float* color, CalibHessian* HCalib)
{
	// 每一层创建图像值, 和图像梯度的存储空间
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
		absSquaredGrad[i] = new float[wG[i]*hG[i]];
	}
	dI = dIp[0];


	// make d0
	int w=wG[0]; // 零层weight
	int h=hG[0]; // 零层height
	
	for(int i=0;i<w*h;i++)
	{
		dI[i][0] = color[i];
	}

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = wG[lvl], hl = hG[lvl]; // 该层图像大小
		Eigen::Vector3f* dI_l = dIp[lvl];

		float* dabs_l = absSquaredGrad[lvl];
		if(lvl>0)
		{
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1]; // 列数
			Eigen::Vector3f* dI_lm = dIp[lvlm1];


			// 像素4合1, 生成金字塔
			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					dI_l[x + y*wl][0] = /*0.25f **/ (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
					int cnt = 4;
					if(dI_lm[2*x + 2*y*wlm1][0] < 0.0001)
						cnt --;
					if(dI_lm[2*x+1 + 2*y*wlm1][0] < 0.0001)
						cnt --;
					if(dI_lm[2*x + 2*y*wlm1+wlm1][0] < 0.0001)
						cnt --;
					if(dI_lm[2*x+1 + 2*y*wlm1+wlm1][0] < 0.0001)
						cnt --;
					if(cnt == 0){
						dI_l[x + y*wl][0] = 0;
					}
					else{
						dI_l[x + y*wl][0] /= cnt ;
					}
				}
		}

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);

			if(dI_l[idx+1][0] < 0.0001 || dI_l[idx-1][0] < 0.0001){
				dx = 0;
			}
			if(dI_l[idx+wl][0] < 0.0001 || dI_l[idx-wl][0] < 0.0001){
				dy = 0;
			}
			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;

			dI_l[idx][1] = dx; // 梯度
			dI_l[idx][2] = dy;


			dabs_l[idx] = dx*dx+dy*dy; // 梯度平方

			if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
			{
				//! 乘上响应函数, 变换回正常的颜色, 因为光度矫正时 I = G^-1(I) / V(x)
				float gw = HCalib->getBGradOnly((float)(dI_l[idx][0])); 
				dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
			}
		}
	}
}

cv::Mat frame_hessian::getCvImages(int lvl)
{
	// 每一层创建图像值, 和图像梯度的存储空间
	float* temp = new float[wG[lvl]*hG[lvl]];

	for(int j = 0; j < wG[lvl] * hG[lvl]; j++)
		temp[j] = dIp[lvl][j][0];

	cv::Mat imgGray(hG[lvl], wG[lvl], CV_32FC1, temp);
	imgGray.convertTo(imgGray, CV_8UC1);
	// cv::Mat imgRGB(hG[lvl], wG[lvl],CV_8UC3);
	// cv::cvtColor(imgGray,imgRGB, cv::COLOR_GRAY2RGB);
	// std::cout<<imgRGB.cols<<" "<<imgRGB.rows<<std::endl;
	// cv::transpose(imgRGB, imgRGB);
	cv::Mat imgRGB(wG[0], hG[0],CV_8UC3);
	for(int i = 0; i < wG[0] * hG[0]; i++){
		Vec3b tmp = LUT[imgGray.at<uchar>(i%hG[0],i/hG[0])];
		imgRGB.at<cv::Vec3b>(i/hG[0],i%hG[0])=cv::Vec3b(tmp[0],tmp[1],tmp[2]);
	}
	// cv::imshow("test",imgRGB);
	// cv::waitKey(0);
	delete temp;
	imgGray.release();
	return imgRGB;
}
cv::Mat frame_hessian::getCvImages()
{
	// 每一层创建图像值, 和图像梯度的存储空间
	float* temp = new float[wG[0]*hG[0]];

	for(int j = 0; j < wG[0] * hG[0]; j++)
		temp[j] = dIp[0][j][0];

	cv::Mat imgGray(hG[0], wG[0], CV_32FC1, temp);
	imgGray.convertTo(imgGray, CV_8UC1);
	delete temp;
	return imgGray;
}

// XTL：求零空间，nullspaces_pose的求法是给worldToCam_evalPT一个SE3扰动，log后相减然后除小量
// XTL：nullspaces_scale的求法是给nullspaces_scale一个translation扰动，log后相减然后除小量
void FrameHessian::setStateZero(const Vec10 &state_zero)
{
	assert(state_zero.head<6>().squaredNorm() < 1e-20);

	this->state_zero = state_zero;

	//! 感觉这个nullspaces_pose就是 Adj_T
	//! Exp(Adj_T*zeta)=T*Exp(zeta)*T^{-1}
	for(int i=0;i<6;i++)
	{
		Vec6 eps; eps.setZero(); eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3::exp(eps);
		SE3 EepsM = Sophus::SE3::exp(-eps);
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	//? rethink
	// scale change
	SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_P_x0.translation() *= 1.00001;
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
	SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_M_x0.translation() /= 1.00001;
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);

	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0(0).a)*ab_exposure[0]);
};



void FrameHessian::release()
{
}


// 2019.11.07 yzk
bool cmp1(const pair<float, float> a, const pair<float, float> b) {
    return a.first<b.first;//自定义的比较函数
}
/*
void FrameHessian::makeDepthImages(float* depthImage)
{
	// 每一层创建图像值, 和图像梯度的存储空间
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		dDepth[i] = new float[wG[i]*hG[i]];
	}

	// make d0
	int w=wG[0]; // 零层weight
	int h=hG[0]; // 零层height
	for(int i=0;i<w*h;i++)
		dDepth[0][i] = depthImage[i] / 5000.0;

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = wG[lvl], hl = hG[lvl]; // 该层图像大小
		float* dDepth_l = dDepth[lvl];

		if(lvl>0)
		{
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1]; // 列数
			float* dDepth_lm = dDepth[lvlm1];

			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					vector<float> vNeighboursValue;

			        if(dDepth_lm[2 * x + 2 * y * wlm1] > 1e-5)
			        {
				        vNeighboursValue.push_back(dDepth_lm[2 * x + 2 * y * wlm1]);
			        }
			        if(dDepth_lm[2 * x + 1 + 2 * y * wlm1] > 1e-5)
			        {
				        vNeighboursValue.push_back(dDepth_lm[2 * x + 1 + 2 * y * wlm1]);
			        }
			        if(dDepth_lm[2 * x + 2 * y * wlm1 + wlm1] > 1e-5)
			        {
				        vNeighboursValue.push_back(dDepth_lm[2 * x + 2 * y * wlm1 + wlm1]);
			        }
			        if(dDepth_lm[2 * x + 1 + 2 * y * wlm1 + wlm1] > 1e-5)
			        {
				        vNeighboursValue.push_back(dDepth_lm[2 * x + 1 + 2 * y * wlm1 + wlm1]);
			        }

			        if(vNeighboursValue.size() == 0)
				        dDepth_l[x + y * wl] = 0;
			        else
			        {
				        float sum = 0.0;

				        for(int j = 0; j < vNeighboursValue.size(); j++)
				        {
					        sum = sum + vNeighboursValue[j];
				        }

				        float average = sum / vNeighboursValue.size();

				        vector<pair<float, float> > vNeighbours;

				        for(int j = 0; j < vNeighboursValue.size(); j++)
				        {
					        vNeighbours.push_back(make_pair(fabs(vNeighboursValue[j] - average) ,vNeighboursValue[j]));
				        }

				        sort(vNeighbours.begin(), vNeighbours.end(), cmp1);

                        //std::cout << "vNeighbours[0].second = " << vNeighbours[0].second << std::endl;
				        dDepth_l[x + y * wl] = vNeighbours[0].second;
			        }
				}
		}
	}
}

void FrameHessian::showDepthImages()
{
	// 每一层创建图像值, 和图像梯度的存储空间
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		cv::Mat m(hG[i], wG[i], CV_32FC1);
		memcpy(m.data, dDepth[i], sizeof(float)*wG[i]*hG[i]);
		m.convertTo(m, CV_16UC1, 5000.0);
		char str[20];
		sprintf(str, "depthImage-%d", i);
		cv::imshow(string(str), m);
	}
}
// 2019.11.07 yzk

// 2020.06.22 yzk
void FrameHessian::showRosImages()
{
	// 每一层创建图像值, 和图像梯度的存储空间
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		float* temp = new float[wG[i]*hG[i]];

		for(int j = 0; j < wG[i] * hG[i]; j++)
		    temp[j] = dIp[i][j][0];

		cv::Mat m(hG[i], wG[i], CV_32FC1);
		memcpy(m.data, temp, sizeof(float)*wG[i]*hG[i]);
		m.convertTo(m, CV_8UC1);
		char str[20];
		sprintf(str, "RosImage-%d", i);
		cv::imshow(string(str), m);
	}
}
// 2020.06.22 yzk

// 2021.12.29
cv::Mat FrameHessian::getCvImages()
{
	// 每一层创建图像值, 和图像梯度的存储空间
	float* temp = new float[wG[0]*hG[0]];

	for(int j = 0; j < wG[0] * hG[0]; j++)
		temp[j] = dIp[0][j][0];

	cv::Mat imgGray(hG[0], wG[0], CV_32FC1);
	memcpy(imgGray.data, temp, sizeof(float)*wG[0]*hG[0]);
	imgGray.convertTo(imgGray, CV_8UC1);


	cv::Mat imgGray(hG[0], wG[0], CV_32FC1, temp);
	imgGray.convertTo(imgGray, CV_8UC1);
	cv::Mat imgRGB(wG[0], hG[0],CV_8UC3);
	for(int i = 0; i < wG[0] * hG[0]; i++){
		Vec3b tmp = LUT[imgGray.at<uchar>(i%hG[0],i/hG[0])];
		imgRGB.at<cv::Vec3b>(i/hG[0],i%hG[0])=cv::Vec3b(tmp[0],tmp[1],tmp[2]);
	}

	// imgGray.release();
	return imgGray;
}*/


//@ 计算优化前和优化后的相对位姿, 相对光度变化, 及中间变量
void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib)
{
	this->host = host;
	this->target = target;
	
	for(int i=0;i<cam_num;i++)
	{
		// 乘上内参
		Mat33f Ki = Mat33f::Zero();
		Ki(0,0) = HCalib->fxl(i);
		Ki(1,1) = HCalib->fyl(i);
		Ki(0,2) = HCalib->cxl(i);
		Ki(1,2) = HCalib->cyl(i);
		Ki(2,2) = 1;
		for(int j=0;j<cam_num;j++)
		{
			// 优化前host target间位姿变换
			SE3 leftToLeft_0 = T_c_c0[j] * target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse() * T_c0_c[i];
			PRE_RTll_0[i*cam_num+j] = (leftToLeft_0.rotationMatrix()).cast<float>();
			PRE_tTll_0[i*cam_num+j] = (leftToLeft_0.translation()).cast<float>();


			// 优化后host到target间位姿变换
			SE3 leftToLeft = T_c_c0[j] * target->PRE_worldToCam * host->PRE_camToWorld * T_c0_c[i];
			PRE_RTll[i*cam_num+j] = (leftToLeft.rotationMatrix()).cast<float>();
			PRE_tTll[i*cam_num+j] = (leftToLeft.translation()).cast<float>();
			if(i==0&&j==0) distanceLL = leftToLeft.translation().norm();
			
			// 乘上内参
			Mat33f Kj = Mat33f::Zero();
			Kj(0,0) = HCalib->fxl(j);
			Kj(1,1) = HCalib->fyl(j);
			Kj(0,2) = HCalib->cxl(j);
			Kj(1,2) = HCalib->cyl(j);
			Kj(2,2) = 1;
			PRE_KRKiTll[i*cam_num+j] = Kj * PRE_RTll[i*cam_num+j] * Ki.inverse();
			PRE_RKiTll[i*cam_num+j] = PRE_RTll[i*cam_num+j] * Ki.inverse();
			PRE_KtTll[i*cam_num+j] = Kj * PRE_tTll[i*cam_num+j];	// 光度仿射值
			PRE_aff_mode[i*cam_num+j] = AffLight::fromToVecExposure(host->ab_exposure[i], target->ab_exposure[j], host->aff_g2l(i), target->aff_g2l(j)).cast<float>();
			PRE_aff_mode[i*cam_num+j][0] = 1;
			PRE_aff_mode[i*cam_num+j][1] = 0;
		}
		PRE_b0_mode[i] = host->aff_g2l_0(i).b;
		PRE_b0_mode[i] = 0;
	}
	/*
	PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
	PRE_b0_mode = host->aff_g2l_0().b;
	*/


}

}

