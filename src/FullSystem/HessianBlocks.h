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
#define MAX_ACTIVE_FRAMES 100

 
#include "util/globalCalib.h"
#include "vector"
 
#include <iostream>
#include <fstream>
#include "util/NumType.h"
#include "FullSystem/Residuals.h"
#include "util/ImageAndExposure.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>

namespace dso
{

//* 求得两个参考帧之间的光度仿射变换系数
// 设from是 i->j(ref->tar);  to是 k->j; 则结果是 i->k 的变换系数.
inline Vec2 affFromTo(const Vec2 &from, const Vec2 &to)	// contains affine parameters as XtoWorld.
{
	return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}


//提前声明下
struct FrameHessian;
struct frame_hessian;
struct PointHessian;

class ImmaturePoint;
class FrameShell;

class EFFrame;
class EFPoint;

//? 这是干什么用的? 是为了求解时候的数值稳定? 
#define SCALE_IDEPTH 1.0f			//!< 逆深度的比例系数  // scales internal value to idepth.
#define SCALE_XI_ROT 1.0f			//!< 旋转量(so3)的比例系数
#define SCALE_XI_TRANS 0.5f			//!< 平移量的比例系数
#define SCALE_F 50.0f   			//!< 相机焦距的比例系数
#define SCALE_C 50.0f				//!< 相机光心偏移的比例系数
#define SCALE_W 1.0f				//!< 不知道...
#define SCALE_A 10.0f				//!< 光度仿射系数a的比例系数
#define SCALE_B 1000.0f				//!< 光度仿射系数b的比例系数

//上面的逆
#define SCALE_IDEPTH_INVERSE (1.0f / SCALE_IDEPTH)
#define SCALE_XI_ROT_INVERSE (1.0f / SCALE_XI_ROT)
#define SCALE_XI_TRANS_INVERSE (1.0f / SCALE_XI_TRANS)
#define SCALE_F_INVERSE (1.0f / SCALE_F)
#define SCALE_C_INVERSE (1.0f / SCALE_C)
#define SCALE_W_INVERSE (1.0f / SCALE_W)
#define SCALE_A_INVERSE (1.0f / SCALE_A)
#define SCALE_B_INVERSE (1.0f / SCALE_B)

//* 其中带0的是FEJ用的初始状态, 不带0的是更新的状态
struct FrameFramePrecalc
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	static int instanceCounter;

	FrameHessian* host;	// defines row
	FrameHessian* target;	// defines column

	
	// precalc values
	Mat33f PRE_RTll[CAM_NUM*CAM_NUM];
	Mat33f PRE_KRKiTll[CAM_NUM*CAM_NUM];
	Mat33f PRE_RKiTll[CAM_NUM*CAM_NUM];
	Mat33f PRE_RTll_0[CAM_NUM*CAM_NUM];

	Vec3f PRE_tTll[CAM_NUM*CAM_NUM];
	Vec3f PRE_KtTll[CAM_NUM*CAM_NUM];
	Vec3f PRE_tTll_0[CAM_NUM*CAM_NUM];
	
	Vec2f PRE_aff_mode[CAM_NUM*CAM_NUM]; // 能量函数对仿射系数处理后的, 总系数
	float PRE_b0_mode[CAM_NUM]; // host的光度仿射系数b
	/*
	Vec2f PRE_aff_mode; // 能量函数对仿射系数处理后的, 总系数
	float PRE_b0_mode; // host的光度仿射系数b
	*/
	float distanceLL; // 两帧间距离

    inline ~FrameFramePrecalc() {}
    inline FrameFramePrecalc() {host=target=0;}
	void set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib);
};

//* 副帧
struct frame_hessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFFrame* efFrame;		//!< 帧的能量函数
	FrameShell* shell;		//!< 帧的"壳", 保存一些不变的,要留下来的量
	int _cam_idx;
	MinimalImageB3* debugImage;	//!< 小图???

	FrameHessian* fh0;

	//* 图像导数[0]:辐照度  [1]:x方向导数  [2]:y方向导数, （指针表示图像）
	Eigen::Vector3f* dI;				//!< 图像导数  // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
	Eigen::Vector3f* dIp[PYR_LEVELS];	//!< 各金字塔层的图像导数  // coarse tracking / coarse initializer. NAN in [0] only.
	float* absSquaredGrad[PYR_LEVELS];  //!< x,y 方向梯度的平方和 // only used for pixel select (histograms etc.). no NAN.
	
    // 2019.11.07 yzk
    float* dDepth[PYR_LEVELS];      //!< 深度图金字塔
    // 2019.11.07 yzk

	static int instanceCounter;		//!< 计数器

	// Photometric Calibration Stuff
	float frameEnergyTH;	//!< 阈值 // set dynamically depending on tracking residual
	bool flaggedForMarginalization;

	std::vector<PointHessian*> pointHessians;				//!< contains all ACTIVE points.
	std::vector<PointHessian*> pointHessiansMarginalized;	//!< contains all MARGINALIZED points (= fully marginalized, usually because point went OOB.)
	std::vector<PointHessian*> pointHessiansOut;			//!< contains all OUTLIER points (= discarded.).
	std::vector<ImmaturePoint*> immaturePoints;				//!< contains all OUTLIER points (= discarded.).

	//* 释放该帧内存
	void release();	
	
	inline ~frame_hessian()
	{
		assert(efFrame==0);
		release(); instanceCounter--;
		for(int i=0;i<pyrLevelsUsed;i++)
		{
			delete[] dIp[i];
			delete[]  absSquaredGrad[i];
		}

		if(debugImage != 0) delete debugImage;
	};
	inline frame_hessian()
	{
		fh0 = 0;
		efFrame = 0;
		instanceCounter++;  //! 若是发生拷贝, 就不会增加了
		flaggedForMarginalization=false;
		frameEnergyTH = 8*8*patternNum;
		debugImage=0;
	};

    void makeImages(float* color, CalibHessian* HCalib);
    // 2020.07.04 yzk
    cv::Mat getCvImages(int lvl);
    // 2020.07.04 yzk
	cv::Mat getCvImages();
};

//* 相机位姿+相机光度Hessian
struct FrameHessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFFrame* efFrame;		//!< 帧的能量函数

	frame_hessian* frame[CAM_NUM];

	// constant info & pre-calculated values
	FrameShell* shell;		//!< 帧的"壳", 保存一些不变的,要留下来的量

	int frameID;					//!< 所有关键帧的序号(FrameShell)	// incremental ID for keyframes only!
	static int instanceCounter;		//!< 计数器
	int idx;						//!< 激活关键帧的序号(FrameHessian)

	// Photometric Calibration Stuff
	float ab_exposure[CAM_NUM];

	//bool flaggedForMarginalization;
	bool w[CAM_NUM];

	Mat66 nullspaces_pose;
	Mat42 nullspaces_affine; // needn't !!
	Vec6 nullspaces_scale;

	// variable info.
	SE3 worldToCam_evalPT;		//!< 在估计的相机位姿
	Vec10 state_zero;   		//!< 固定的线性化点的状态增量, 为了计算进行缩放
	Vec10 state_scaled;			//!< 乘上比例系数的状态增量, 这个是真正求的值!!!
	// XTL：这个state（状态）是w2cam的更新量
	Vec10 state;				//!< 计算的状态增量
	//* step是与上一次优化结果的状态增量, [8 ,9]直接就设置为0了
	Vec10 step;					//!< 求解正规方程得到的增量
	Vec10 step_backup;			//!< 上一次的增量备份
	Vec10 state_backup;			//!< 上一次状态的备份

	//内联提高效率, 返回上面的值
    EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPT() const {return worldToCam_evalPT;}
    EIGEN_STRONG_INLINE const Vec10 &get_state_zero() const {return state_zero;}
    EIGEN_STRONG_INLINE const Vec10 &get_state() const {return state;}
    EIGEN_STRONG_INLINE const Vec10 &get_state_scaled() const {return state_scaled;}
    EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const {return get_state() - get_state_zero();} //x小量可以直接减


	// precalc values
	SE3 PRE_worldToCam;			//!< 预计算的, 位姿状态增量更新到位姿上
	SE3 PRE_camToWorld;
	std::vector<FrameFramePrecalc,Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc; //!< 对于其它帧的预运算值
	MinimalImageB3* debugImage;	//!<


    inline Vec6 w2c_leftEps() const {return get_state_scaled().head<6>();}  //* 返回位姿状态增量
	inline AffLight aff_g2l_0(int idx) const {return AffLight(get_state_zero()[6]*SCALE_A, get_state_zero()[7]*SCALE_B);} //* 返回线性化点处的仿射系数增量
    inline AffLight aff_g2l(int idx) const {return AffLight(get_state_scaled()[6], get_state_scaled()[7]);}

	//* 设置FEJ点状态增量
	void setStateZero(const Vec10 &state_zero);
	// XTL：这里的PRE_worldToCam是世界到该帧的SE3变换，get_worldToCam_evalPT返回的是估计的位姿，w2c_leftEps是更新的小量
	// XTL：迭代优化时调用，SCALE_A，SCALE_B可以看作步长，求解出来的增量乘以这些常数才是真正的增量
	inline void setState(const Vec10 &state)
	{
		this->state = state;
		state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
		state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);

		state_scaled[6] = SCALE_A * state[6];
		state_scaled[7] = SCALE_B * state[7];
		state_scaled[8] = SCALE_A * state[8];
		state_scaled[9] = SCALE_B * state[9];

		//位姿更新
		PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
		PRE_camToWorld = PRE_worldToCam.inverse();
		//setCurrentNullspace();
	};
	// XTL：只在setEvalPT_scaled当中用到了这个函数，其输入参数只有[6][7]有值
	// XTL：该函数只会在setEvalPT_scaled中调用，也就是每次估计完初值，进行进一步优化之前，给state_scaled和state的[6][7]赋值
	// XTL：那么这时候PRE_worldToCam=worldToCam_evalPT，也就是刚刚估计出来的初值
	inline void setStateScaled(const Vec10 &state_scaled)
	{

		this->state_scaled = state_scaled;
		state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
		state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);

		state[6] = SCALE_A_INVERSE * state_scaled[6];
		state[7] = SCALE_B_INVERSE * state_scaled[7];
		state[8] = SCALE_A_INVERSE * state_scaled[8];
		state[9] = SCALE_B_INVERSE * state_scaled[9];

		PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
		PRE_camToWorld = PRE_worldToCam.inverse();
		//setCurrentNullspace();
	};
	// XTL:这里的输入参数worldToCam_evalPT为PRE_worldToCam，意味着worldToCam_evalPT就是PRE_worldToCam
	// XTL:这里只会在优化完毕后调用，state前6维=0
	inline void setEvalPT(const SE3 &worldToCam_evalPT, const Vec10 &state)
	{
		this->worldToCam_evalPT = worldToCam_evalPT;
		setState(state);
		setStateZero(state);
	};


	// XTL：该函数只会在每次估计完初值，进行进一步优化之前调用（makekeyframe、makenokeyframe开头）
	// XTL：这个函数传入的参数worldToCam_evalPT是估计的WorldToCam
	// XTL：这里调用了setStateZero，即在进一步优化之前将状态(其实只有光度参数)保存在了state_zero中
	inline void setEvalPT_scaled(const SE3 &worldToCam_evalPT, const AffLight aff_g2l[CAM_NUM])
	{
		Vec10 initial_state = Vec10::Zero();

		initial_state[6] = aff_g2l[0].a;
		initial_state[7] = aff_g2l[0].b;
		
		this->worldToCam_evalPT = worldToCam_evalPT;
		setStateScaled(initial_state);
		setStateZero(this->get_state());
	};

	//* 释放该帧内存
	void release();

	inline ~FrameHessian()
	{
		assert(efFrame==0);
		release(); instanceCounter--;

		if(debugImage != 0) delete debugImage;
	};
	inline FrameHessian()
	{
		instanceCounter++;  //! 若是发生拷贝, 就不会增加了
		// flaggedForMarginalization=false;
		frameID = -1;
		efFrame = 0;
		for(int i=0;i<cam_num;i++)
			w[i] = 1;

		debugImage=0;
	};
	

	inline Vec10 getPrior()
	{
		Vec10 p =  Vec10::Zero();
		
		if(frameID==0)
		{
			p.head<3>() = Vec3::Constant(setting_initialTransPrior);
			p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);

			if(setting_solverMode & SOLVER_REMOVE_POSEPRIOR) p.head<6>().setZero();

			p[6] = setting_initialAffAPrior; // 1e14
			p[7] = setting_initialAffBPrior; // 1e14
		}
		else
		{
			// XTL 固定时
			if(setting_affineOptModeA < 0)
				p[6] = setting_initialAffAPrior;  //1e14
			else // XTL 优化时
				p[6] = setting_affineOptModeA;   // 自己设置

			if(setting_affineOptModeB < 0)
				p[7] = setting_initialAffBPrior; //1e14
			else
				p[7] = setting_affineOptModeB;  // 自己设置
		}
		p[8] = setting_initialAffAPrior;
		p[9] = setting_initialAffBPrior;
		return p;
	}


	inline Vec10 getPriorZero()
	{
		return Vec10::Zero();
	}

};


struct dataIMG
{
	cv::Mat sobel, canny;
	int threshold_canny;
	dataIMG(int th):threshold_canny(th){};
};

struct Description
{
	cv::Point2i kp_p11,kp_p22;
	cv::Point2i pair_p11,pair_p22;
    cv::Point2i kp;
    cv::Point2i _pair;
	cv::Point2f center;
	std::vector<cv::Point2i> pixels;
	std::vector<cv::Point2i> pixels_center;
    float distance;
	short cnt[100][100];
	float entropy;
    float rotation;
	inline Description()
	{
		_pair.x = 0;
		_pair.y = 0;
		pair_p11 = _pair;
		pair_p22 = _pair;
		kp_p11 = _pair;
		kp_p22 = _pair;
		center.x = 0;
		center.y = 0;
	}
	inline void calCenter()
	{
		for(int i=0;i<pixels.size();i++)
		{
			center.x += pixels[i].x;
			center.y += pixels[i].y;
		}
		center.x = 1.0*center.x/pixels.size();
		center.y = 1.0*center.y/pixels.size();
	}
	inline void findKp()
	{
		float min = 1000; cv::Point2i minIdx;
		for(auto pt:pixels)
		{
			float diffX = pt.x-center.x;
			float diffY = pt.y-center.y;
			float dist = sqrt(diffX*diffX+diffY*diffY);
			if(dist<min)
			{
				dist = min;
				minIdx = pt;
			}
		}
		kp = minIdx;
	}
	inline bool findPair(int min_rho, int max_rho)
	{
		float min = 1000;
		for(auto pt:pixels)
		{
			float diffX = pt.x-kp.x;
			float diffY = pt.y-kp.y;
			float dist = sqrt(diffX*diffX+diffY*diffY);
			if(dist<max_rho&&dist>min_rho)
			{
				_pair = pt;
				distance = dist;
				rotation = atan2(diffY, diffX) * 180.0 / M_PI;
				return true;
			}
			else if(dist<min&&dist>0)
			{
				dist = min;
				rotation = atan2(diffY, diffX) * 180.0 / M_PI;
				_pair = pt;
			}
		}
		return true;
	}
	inline void calcEntropy()
	{
		memset(cnt,0,sizeof(short)*10000);
		int min_x=2000,min_y=2000,max_x=-1,max_y=-1;
		for(auto pt:pixels)
		{
			if(pt.x<min_x)
				min_x = pt.x;
			if(pt.x>max_x)
				max_x = pt.x;
			if(pt.y<min_y)
				min_y = pt.y;
			if(pt.y>max_y)
				max_y = pt.y;
		}
		int grid = 1;
		int grid_x = (max_x-min_x);
		int grid_y = (max_y-min_y);
		int sum = 0;
		entropy = 0;
		if(grid_x==0||grid_y==0)
			return ;

		for(auto pt:pixels)
		{
			int u = (int)round((1.0*(pt.x-min_x)/grid_x)*99);
			int v = (int)round((1.0*(pt.y-min_y)/grid_y)*99);
			// std::cout<<"min_y="<<min_y<<std::endl;
			// std::cout<<"pt.y="<<pt.y<<std::endl;
			// std::cout<<"grid_y="<<grid_y<<std::endl;
			// std::cout<<"u="<<u<<",v="<<v<<std::endl;
			cnt[u][v] ++;
			sum ++;
		}
		for(int i=0;i<100;i++)
			for(int j=0;j<100;j++)
			{
				if(cnt[i][j]==0)
					continue;
				float prob = 1.0 * cnt[i][j] / sum;
				entropy -= prob * log2(prob);
			}

		std::cout<<"entropy is:"<<entropy<<std::endl;
	}
	inline void transform(float s,float yaw,cv::Point _kp)
	{
		pixels_center.resize(pixels.size());
		for(int i=0;i<pixels_center.size();i++)
		{
			float x,y;
			x = s*(pixels[i].x - kp.x);
			y = s*(pixels[i].y - kp.y);
			pixels_center[i].x = (int)round(x*cos(yaw) - y*sin(yaw) + _kp.x);
			pixels_center[i].y = (int)round(y*cos(yaw) + x*sin(yaw) + _kp.y);
		}
	}
	inline void calCROI_kp(int param)
	{
		int min_x = std::max(kp.x - (int)round(2*param/2),1);
		int min_y = std::max(kp.y - param/2,1);
		int max_x = std::min(kp.x + (int)round(2*param/2),wG[0]);
		int max_y = std::min(kp.y + param/2,wG[0]);
		kp_p11.x = min_x;
		kp_p11.y = min_y;
		kp_p22.x = max_x;
		kp_p22.y = max_y;
	}
	inline void calCROI_pair(int param)
	{
		int min_x = std::max(_pair.x - (int)round(2*param/2),1);
		int min_y = std::max(_pair.y - param/2,1);
		int max_x = std::min(_pair.x + (int)round(2*param/2),wG[0]);
		int max_y = std::min(_pair.y + param/2,wG[0]);
		pair_p11.x = min_x;
		pair_p11.y = min_y;
		pair_p22.x = max_x;
		pair_p22.y = max_y;
	}
	void draw_kpROI(cv::Mat& m, cv::Vec3b color)
	{
		int p1_u = kp_p11.x; int p1_v = kp_p11.y; int p2_u = kp_p22.x; int p2_v = kp_p22.y;
		for(int u=p1_u;u<p2_u;u++)
			draw(m,u,p1_v,color,0);
		for(int u=p1_u;u<p2_u;u++)
			draw(m,u,p2_v,color,0);
		for(int v=p1_v;v<p2_v;v++)
			draw(m,p1_u,v,color,0);
		for(int v=p1_v;v<p2_v;v++)
			draw(m,p2_u,v,color,0);
	}
	void drawLine_kp_Pair(cv::Mat& m)
	{
		cv::Point p1(kp.x,kp.y);
		cv::Point p2(_pair.x,_pair.y);
		cv::Scalar colour;
		if(entropy>5)
			colour = cv::Scalar(0);
		else if(entropy>3)
			colour = cv::Scalar(128);
		else
			colour = cv::Scalar(255);
		cv::line(m,p1,p2,colour,1);
	}
	void drawLine_aligned_Pair(cv::Mat& m,cv::Point p3,cv::Point p4)
	{
		cv::Point p1(kp.x,kp.y);
		cv::Point p2(_pair.x,_pair.y);
		cv::Scalar colour;
		colour = cv::Scalar(255);
		cv::line(m,p1,p2,colour,1);
		colour = cv::Scalar(0);
		cv::line(m,p3,p4,colour,1);
	}
	void draw_pairROI(cv::Mat& m, cv::Vec3b color)
	{
		int p1_u = pair_p11.x; int p1_v = pair_p11.y; int p2_u = pair_p22.x; int p2_v = pair_p22.y;
		for(int u=p1_u;u<p2_u;u++)
			draw(m,u,p1_v,color,0);
		for(int u=p1_u;u<p2_u;u++)
			draw(m,u,p2_v,color,0);
		for(int v=p1_v;v<p2_v;v++)
			draw(m,p1_u,v,color,0);
		for(int v=p1_v;v<p2_v;v++)
			draw(m,p2_u,v,color,0);
	}
};


//* 相机内参Hessian, 响应函数
struct CalibHessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	static int instanceCounter;
	// * 4×1的向量
	VecC value_zero[CAM_NUM];
	VecC value_scaled[CAM_NUM];
	VecCf value_scaledf[CAM_NUM];
	VecCf value_scaledi[CAM_NUM];
	VecC value[CAM_NUM];
	VecC step[CAM_NUM];
	VecC step_backup[CAM_NUM];
	VecC value_backup[CAM_NUM];
	VecC value_minus_value_zero[CAM_NUM];

    inline ~CalibHessian() {instanceCounter--;}
	inline CalibHessian()
	{

		VecC initial_value = VecC::Zero();		//typedef Eigen::Matrix<double, 4, 1>
		//* 初始化内参
		/*
		initial_value[0] = fxG[0];
		initial_value[1] = fyG[0];
		initial_value[2] = cxG[0];
		initial_value[3] = cyG[0];

		setValueScaled(initial_value);	
		value_zero = value;
		value_minus_value_zero.setZero();
		*/
		for(int i=0;i<cam_num;i++){
			initial_value[0] = fxG[i][0];
			initial_value[1] = fyG[i][0];
			initial_value[2] = cxG[i][0];
			initial_value[3] = cyG[i][0];
			setValueScaled(initial_value,i);
			value_zero[i] = value[i];
			value_minus_value_zero[i].setZero();
		}

		instanceCounter++;
		//响应函数
		for(int i=0;i<256;i++)
			Binv[i] = B[i] = i;		// set gamma function to identity
	};


	// normal mode: use the optimized parameters everywhere!
    inline float& fxl(int cam_idx) {return value_scaledf[cam_idx][0];}
    inline float& fyl(int cam_idx) {return value_scaledf[cam_idx][1];}
    inline float& cxl(int cam_idx) {return value_scaledf[cam_idx][2];}
    inline float& cyl(int cam_idx) {return value_scaledf[cam_idx][3];}
    inline float& fxli(int cam_idx) {return value_scaledi[cam_idx][0];}
    inline float& fyli(int cam_idx) {return value_scaledi[cam_idx][1];}
    inline float& cxli(int cam_idx) {return value_scaledi[cam_idx][2];}
    inline float& cyli(int cam_idx) {return value_scaledi[cam_idx][3];}

	//* 通过value设置
	inline void setValue(const VecC &value, int cam_idx)
	{
		// [0-3: Kl, 4-7: Kr, 8-12: l2r] what's this, stereo camera???
		/*
		this->value = value;
		value_scaled[0] = SCALE_F * value[0];
		value_scaled[1] = SCALE_F * value[1];
		value_scaled[2] = SCALE_C * value[2];
		value_scaled[3] = SCALE_C * value[3];

		this->value_scaledf = this->value_scaled.cast<float>();
		this->value_scaledi[0] = 1.0f / this->value_scaledf[0];
		this->value_scaledi[1] = 1.0f / this->value_scaledf[1];
		this->value_scaledi[2] = - this->value_scaledf[2] / this->value_scaledf[0];
		this->value_scaledi[3] = - this->value_scaledf[3] / this->value_scaledf[1];
		this->value_minus_value_zero = this->value - this->value_zero;
		*/
		this->value[cam_idx] = value;

		value_scaled[cam_idx][0] = SCALE_F * value[0];
		value_scaled[cam_idx][1] = SCALE_F * value[1];
		value_scaled[cam_idx][2] = SCALE_C * value[2];
		value_scaled[cam_idx][3] = SCALE_C * value[3];

		this->value_scaledf[cam_idx] = this->value_scaled[cam_idx].cast<float>();
		this->value_scaledi[cam_idx][0] = 1.0f / this->value_scaledf[cam_idx][0];
		this->value_scaledi[cam_idx][1] = 1.0f / this->value_scaledf[cam_idx][1];
		this->value_scaledi[cam_idx][2] = - this->value_scaledf[cam_idx][2] / this->value_scaledf[cam_idx][0];
		this->value_scaledi[cam_idx][3] = - this->value_scaledf[cam_idx][3] / this->value_scaledf[cam_idx][1];
		this->value_minus_value_zero[cam_idx] = this->value[cam_idx] - this->value_zero[cam_idx];
	};
	//* 通过value_scaled赋值
	inline void setValueScaled(const VecC &value_scaled, int cam_idx)
	{
		this->value_scaled[cam_idx] = value_scaled;
		this->value_scaledf[cam_idx] = this->value_scaled[cam_idx].cast<float>();

		value[cam_idx][0] = SCALE_F_INVERSE * value_scaled[0];
		value[cam_idx][1] = SCALE_F_INVERSE * value_scaled[1];
		value[cam_idx][2] = SCALE_C_INVERSE * value_scaled[2];
		value[cam_idx][3] = SCALE_C_INVERSE * value_scaled[3];

		this->value_minus_value_zero[cam_idx] = this->value[cam_idx] - this->value_zero[cam_idx];
		this->value_scaledi[cam_idx][0] = 1.0f / this->value_scaledf[cam_idx][0];
		this->value_scaledi[cam_idx][1] = 1.0f / this->value_scaledf[cam_idx][1];
		this->value_scaledi[cam_idx][2] = - this->value_scaledf[cam_idx][2] / this->value_scaledf[cam_idx][0];
		this->value_scaledi[cam_idx][3] = - this->value_scaledf[cam_idx][3] / this->value_scaledf[cam_idx][1];
		
	};

	//* gamma函数, 相机的响应函数G和G^-1, 映射到0~255
	float Binv[256];
	float B[256];

	//* 响应函数的导数
	EIGEN_STRONG_INLINE float getBGradOnly(float color)
	{
		int c = color+0.5f;
		if(c<5) c=5;
		if(c>250) c=250;
		return B[c+1]-B[c];
	}
	//* 响应函数逆的导数
	EIGEN_STRONG_INLINE float getBInvGradOnly(float color)
	{
		int c = color+0.5f;
		if(c<5) c=5;
		if(c>250) c=250;
		return Binv[c+1]-Binv[c];
	}
};

//* 点Hessian
// hessian component associated with one point.
struct PointHessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	static int instanceCounter;
	EFPoint* efPoint; 						//!< 点的能量函数

	// static values
	float color[MAX_RES_PER_POINT];			// colors in host frame
	float weights[MAX_RES_PER_POINT];		// host-weights for respective residuals.



	float u,v;							//!< 像素点的位置
	int idx;							//!< 
	float energyTH;						//!< 光度误差阈值
	frame_hessian* host;					//!< 主帧
	bool hasDepthPrior;					//!< 初始化得到的点是有深度先验的, 其它没有

	float idepth_fromSensor;

	bool isFromSensor;

	float my_type;//不同类型点, 显示用

	float idepth_scaled;				//!< host上点逆深度
	float idepth_zero_scaled;			//!< FEJ使用, 点在host上x=0初始逆深度
	float idepth_zero;					//!< 缩放了scale倍的固定线性化点逆深度
	float idepth;						//!< 缩放scale倍的逆深度
	float step;							//!< 迭代优化每一步增量
	float step_backup;					//!< 迭代优化上一步增量的备份
	float idepth_backup;				//!< 上一次的逆深度值

	float nullspaces_scale;				//!< 零空间 ?
	float idepth_hessian;				//!< 对应的hessian矩阵值
	float maxRelBaseline;				//!< 衡量该点的最大基线长度
	int numGoodResiduals;
	
	enum PtStatus {ACTIVE=0, INACTIVE, OUTLIER, OOB, MARGINALIZED};  // 这些状态都没啥用.....
	PtStatus status;

    inline void setPointStatus(PtStatus s) {status=s;}

	//* 各种设置逆深度
	inline void setIdepth(float idepth) {
		this->idepth = idepth;
		this->idepth_scaled = SCALE_IDEPTH * idepth;
    }
	inline void setIdepthScaled(float idepth_scaled) {
		this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
		this->idepth_scaled = idepth_scaled;
    }
	inline void setIdepthZero(float idepth) {
		idepth_zero = idepth;
		idepth_zero_scaled = SCALE_IDEPTH * idepth;
		nullspaces_scale = -(idepth*1.001 - idepth/1.001)*500; //? 为啥这么求
    }

	//* 点的残差值
	std::vector<PointFrameResidual*> residuals;					// only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
	std::pair<PointFrameResidual* , ResState > lastResiduals[10/*2*/]; 	// contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).
	
	void release();

	PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib);
    inline ~PointHessian() {assert(efPoint==0); release(); instanceCounter--;}

	// XTL：对于该点的每一个残差项，若状态为IN，且其目标帧将要被边缘化，则visInToMarg++;若残差项的数量大于阈值，好的残差项数量大于阈值，减去即将要被边缘化的残差项数量后小于阈值，则返回true;
	// XTL：若在最新帧上的残差状态为OOB，或者在最新两帧上的状态都为OUTLIER，就返回true
	// XTL：若残差数量小于2,返回false;
	inline bool isOOB(const std::vector<FrameHessian*>& toKeep, const std::vector<frame_hessian*>& toMarg) const
	{

		int visInToMarg = 0;
		for(PointFrameResidual* r : residuals)
		{
			if(r->state_state != ResState::IN) continue;  
			for(frame_hessian* k : toMarg)
				if(r->target == k) visInToMarg++;  // 在要边缘化掉的帧被观测的数量
		}
		//[1]: 原本是很好的一个点，但是边缘化一帧后，残差变太少了, 边缘化or丢掉
		if((int)residuals.size() >= setting_minGoodActiveResForMarg &&  // 残差数大于一定数目
				numGoodResiduals > setting_minGoodResForMarg+10 &&
				(int)residuals.size()-visInToMarg < setting_minGoodActiveResForMarg) //剩余残差足够少
			return true;



		//[2]: 最新一帧的投影在图像外了, 看不见了, 边缘化or丢掉
		// 或者满足以下条件,
		if(setting_useMultualPBack)
		{
			if(lastResiduals[0].second == ResState::OOB && 
				lastResiduals[2].second == ResState::OOB &&
				lastResiduals[4].second == ResState::OOB &&
				lastResiduals[6].second == ResState::OOB &&
				lastResiduals[8].second == ResState::OOB ) return true;   // XTL 点在最新关键帧上的投影为OOB
		}
		else
			if(lastResiduals[0].second == ResState::OOB) return true;
		//[3]: 残差比较少, 新加入的, 不边缘化
		if(residuals.size() < 2) return false;	//观测较少不设置为OOB
		// xtl 2022.1.30
		//[4]: 前两帧投影都是外点, 边缘化or丢掉
		if(setting_useMultualPBack)
		{
			if(lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER
			&&lastResiduals[2].second == ResState::OUTLIER && lastResiduals[3].second == ResState::OUTLIER
			&&lastResiduals[4].second == ResState::OUTLIER && lastResiduals[5].second == ResState::OUTLIER
			&&lastResiduals[6].second == ResState::OUTLIER && lastResiduals[7].second == ResState::OUTLIER
			&&lastResiduals[8].second == ResState::OUTLIER && lastResiduals[9].second == ResState::OUTLIER) return true;
		}
		else
			if(lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER) return true;
		return false;
	}

	//内点条件
	inline bool isInlierNew()
	{
		return (int)residuals.size() >= setting_minGoodActiveResForMarg
                    && numGoodResiduals >= setting_minGoodResForMarg;
	}

};





}

