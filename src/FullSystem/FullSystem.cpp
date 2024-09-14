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
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

#include "opencv2/highgui/highgui.hpp"
#include <fstream>

#include <dirent.h>

#include <random>

namespace dso
{
// Hessian矩阵计数, 有点像 shared_ptr
int FrameHessian::instanceCounter=0;
int frame_hessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;


const double mean = 0;
const double stddev = 0.001;
std::default_random_engine generator;
std::normal_distribution<double> dist(mean,stddev);
/********************************
 * @ function: 构造函数
 * 
 * @ param: 
 * 
 * @ note:
 *******************************/
// 2020.06.22 yzk
FullSystem::FullSystem()
{
	initializationValue();
}
// 2020.06.22 yzk

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	// 删除new的ofstream
	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}
	delete[] selectionMap;
	// delete[] selectionMapFromLidar;


	idxUseLast.clear();
	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames){
		for(int i=0;i<cam_num;i++)
			delete fh->frame[i];
		delete fh;
	}
	
	while(!Clouds.empty()) { Clouds.pop(); }

	while(!Cloud_ijrr.empty()) 
	{
		//for(int i=0;i<cam_num;i++)
			Cloud_ijrr.pop(); 
	}
	if(setting_useKFselection)
	{
		while(!fhsWindow.empty())
		{
			if(fhsWindow.front()!=0)
			{
				for(int i=0;i<cam_num;i++)
					delete fhsWindow.front()->frame[i];
				delete fhsWindow.front();
			}
			fhsWindow.pop();
            cloudsWindow.pop();
			// if(setting_useNCLT)
			// 	cloudsWindow.pop();
			// else
			// 	for(int i=0;i<cam_num;i++)
			// 		cloudsWindow_ijrr[i].pop();
		}
	}
	while(!qImg[0].empty())
	{
        for(int i=0;i<cam_num;i++){
            qImg[i].front().release();
            qImg[i].pop();
        }
	}

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}
void FullSystem::loadTimeStamp(std::vector<std::string> timeLidar)
{
    // DIR *dp;
    // struct dirent *dirp;
    // std::string::size_type position;
	// std::string lidar_path;
	// lidar_path = dataset_folder;
	// printf("Reading timestamp from file %s",lidar_path.c_str());

	// if((dp  = opendir(lidar_path.c_str())) == NULL)
	// {
	// 	std::cout<<"no such file!!"<<std::endl;
	// 	return ;
	// }
	// printf(" ... found!\n");

	// while ((dirp = readdir(dp)) != NULL) {
	// 	std::string name = std::string(dirp->d_name);
	// 	position = name.find(".");
	// 	name = name.substr(0,position);
	// 	if(name != "" && name != " ")
	// 		time_lidar.push_back(name);
	// 	// printf("%s\n",name.c_str());
	// }
	// closedir(dp);
	// std::sort(time_lidar.begin(), time_lidar.end());
	time_lidar = timeLidar;
}
void FullSystem::loadTimeStamp_ijrr(std::string time_stamp_path)
{
	printf("Reading timestamp from file %s",time_stamp_path.c_str());

	std::ifstream f(time_stamp_path.c_str());
	std::string tmp, time;
	std::string offset="4";
	while(1)
	{
		f >> tmp;
		f >> time;
		f >> time;
		f >> time;
		if(tmp==offset)
			break;
	}
	while(!f.eof())
	{
		f >> tmp;
		f >> time;
		f >> tmp;
		f >> tmp;
		time_lidar.push_back(time);
	}
	f.close();
	std::sort(time_lidar.begin(), time_lidar.end());
}

// 2020.06.22 yzk
void FullSystem::loadSensorPrameters(const std::string &pathSensorParameter)
{

	Eigen::Matrix4d T_lidar_lb3 = xyzrpy2T(x_lidar_lb3);
	
	std::string extrinsic_path[CAM_NUM];
	for(int i=0;i<cam_num;i++){
		extrinsic_path[i] = pathSensorParameter + std::to_string(cam_idx_array[i]) + ".csv";
		// printf("Reading Extrinsic from file %s",extrinsic_path[i].c_str());
		std::ifstream f(extrinsic_path[i].c_str());
		if (!f.good())
		{
			f.close();
			printf("%s ... not found. Cannot operate without Extrinsic, shutting down.\n",extrinsic_path[i].c_str());
			f.close();
			return ;
		}
		// printf(" ... found!\n");
		float rpy[6];
		char comma;
		for(int i=0;i<6;i++){
			f >> rpy[i];
			f >> comma;
		}
		f.close();
		Eigen::Matrix4d T_lb3_c = xyzrpy2T(rpy);
		Eigen::Matrix4d T_c_lidar = (T_lidar_lb3 * T_lb3_c).inverse();
		std::cout<<"T_c_lidar["<<i<<"]=\n"<<T_c_lidar<<std::endl;
		Rcl[i] = T_c_lidar.block<3,3>(0,0);
		tcl[i] = T_c_lidar.topRightCorner<3, 1>();
		setGlobalExtrin(rpy, i);
	}
	calGlobalExtrin();
	// 2021.11.14
}

void FullSystem::loadSensorPrameters_ijrr(const std::string &pathSensorParameter)
{
	std::string extrinsic_path[CAM_NUM];
	for(int i=0;i<cam_num;i++){
		extrinsic_path[i] = pathSensorParameter + std::to_string(cam_idx_array[i]) + ".csv";
		printf("Reading Extrinsic from file %s",extrinsic_path[i].c_str());
		std::ifstream f(extrinsic_path[i].c_str());
		if (!f.good())
		{
			f.close();
			printf(" ... not found. Cannot operate without Extrinsic, shutting down.\n");
			f.close();
			return ;
		}
		printf(" ... found!\n");
		float rpy[6];
		char comma;
		Eigen::Matrix4d T_c_lidar = Eigen::Matrix4d::Identity();
		for(int i=0;i<3;i++){
			for(int j=0;j<4;j++)
			{
				f >> T_c_lidar(i,j);
			}
		}
		f.close();
		std::cout<<T_c_lidar<<std::endl;
		setGlobalExtrin_ijrr(T_c_lidar, i);
	}
	calGlobalExtrin();
}

void FullSystem::initializationValue()
{
	int retstat =0;
	frameNum = 0; cnt =0; haveMakedKF = false;
	havePushed = 0;
	if(setting_logStuff)
	{
		//shell命令删除旧的文件夹, 创建新的
		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		// 打开读写log文件
		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847); // shell正常执行结束返回这么个值,填充8~15位bit, 有趣

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);

	selectionMap = new float[wG[0]*hG[0]];

	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100); //5维向量都=100

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this); // 建图线程单开
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;


	if(setting_useNCLT)
	{
		x_lidar_lb3_ori[0] = x_lidar_lb3[0] = 0.035; x_lidar_lb3_ori[1] = x_lidar_lb3[1] = 0.002;
		x_lidar_lb3_ori[2] = x_lidar_lb3[2] = -1.23; x_lidar_lb3_ori[3] = x_lidar_lb3[3] = -179.93; 
		x_lidar_lb3_ori[4] = x_lidar_lb3[4] = -0.23; x_lidar_lb3_ori[5] = x_lidar_lb3[5] = 0.50; 
	}
    else{
		x_lidar_lb3_ori[0] = x_lidar_lb3[0] = 0; x_lidar_lb3_ori[1] = x_lidar_lb3[1] = 0;
		x_lidar_lb3_ori[2] = x_lidar_lb3[2] = 0; x_lidar_lb3_ori[3] = x_lidar_lb3[3] = 0; 
		x_lidar_lb3_ori[4] = x_lidar_lb3[4] = 0; x_lidar_lb3_ori[5] = x_lidar_lb3[5] = 0; 
    }
	idxUseLast.clear();
	for(int i:idx_use)
		idxUseLast.push_back(i);
}
// 2020.06.22 yzk

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

//* 设置相机响应函数
void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	//myfile << std::setprecision(15);

	for(FrameShell* s : allFrameHistory)
	{
		if(!s->poseValid) continue;

		//if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

		Eigen::Quaterniond q;



	    q.w() = s->camToWorld.so3().unit_quaternion().w();
	    q.x() = s->camToWorld.so3().unit_quaternion().x();
	    q.y() = s->camToWorld.so3().unit_quaternion().y();
	    q.z() = s->camToWorld.so3().unit_quaternion().z();

	    Eigen::Vector3d t = s->camToWorld.translation();

		myfile.setf(std::ios::scientific, std::ios::floatfield);
		myfile.precision(6);

		myfile << std::fixed << s->timestamp << " " << (float)t(0, 0) << " " << (float)t(1, 0) << " " << (float)t(2, 0) << " " 
			   << (float)q.x() << " " << (float)q.y() << " " << (float)q.z() << " " << (float)q.w() << "\n";

	}
	myfile.close();
}

// XTL：尝试一系列的运动，进行最优初值的选择，更新camToTrackingRef，camToWorld
// 2020.07.13 yzk shiyong
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);

	FrameHessian* lastF = coarseTracker->lastRef;  // 参考帧

//[ ***step 1*** ] 设置不同的运动状态
	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	// XTL：若只有两帧（最开始调用该函数的情况），则认为这两帧之间无运动。
	if(allFrameHistory.size() == 2) {
		lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double,3,1>::Zero() ));
		initializeFromInitializer(fh);

		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
        {
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
            lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        }
		
		coarseTracker->makeK(&Hcalib);
		coarseTracker->setCTRefForFirstFrame(frameHessians);

		lastF = coarseTracker->lastRef;
	}
	else // XTL：将上一帧到大上帧的运动作为这帧到上帧的运动的粗略初值
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];   // 上一帧
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];  // 大上一帧

		double timeslast = slast->timestamp;
		double timesprelast = sprelast->timestamp;
		double timefh = fh->shell->timestamp;
		float ratio = (timefh-timeslast)/(timeslast-timesprelast);
		bool useConstVel = false;
		if(fabs(ratio-1)<0.01)	{useConstVel = true;}

		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			if(USE_GT){
				lastF_2_slast = slast->gtPose.inverse() * lastF->shell->gtPose;	// 参考帧到上一帧运动
			}
			else{
				slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;  // 上一帧和大上一帧的运动
				lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;	// 参考帧到上一帧运动
			}
		}
		SE3 fh_2_slast;
		if(USE_GT){
			fh_2_slast = slast->gtPose.inverse() * fh->shell->gtPose;
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,0.02), Vec3(0,0,0)));
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0.02,0), Vec3(0,0,0)));
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0.02,0,0), Vec3(0,0,0)));
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		}
		else{
			fh_2_slast = slast_2_sprelast;
		
			// get last delta-movement.
			if(useConstVel)
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
			else
				lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*ratio).inverse() * lastF_2_slast);	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
			lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
			lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
			lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

			// just try a TON of different initializations (all rotations). In the end,
			// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
			// also, if tracking rails here we loose, so we really, really want to avoid that.
			for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
			{
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			}

			if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) // 有不合法的
			{
				lastF_2_fh_tries.clear();
				lastF_2_fh_tries.push_back(SE3());
			}
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();

	//! as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	//! I'll keep track of the so-far best achieved residual for each level in achievedRes. 
	//! If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;

	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{

		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
		
		// XTL 当算出来的ab绝对值在正常范围内，且每层算出来的平均能量都不大于1.5倍的achievedRes时，trackingIsGood为true
		// XTL 否则认为这个位姿初值不对，尝试下一个位姿
		bool trackingIsGood;
        if(!setting_useMultualP)
        {
            trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this,
				pyrLevelsUsed-1,
				achievedRes,lastF_2_fh_tries[0],i);	// in each level has to be at least as good as the last try.
        }
        // xtl 2023.2.18
		else 
        {
            trackingIsGood = coarseTracker->trackNewestCoarse_ransac(
				fh, lastF_2_fh_this,
				pyrLevelsUsed-1,
				achievedRes,lastF_2_fh_tries[0]);	// in each level has to be at least as good as the last try.
        }
		tryIterations++;

		// XTL 这代表匀速运动模型失效了
		if(i != 0)
		{	
			// XTL lastResiduals为这次迭代中算出来的各层的残差项，achievedRes为有trackingIsGood时，求得的最小残差项
			printf("RE-TRACK ATTEMPT %d : %f -> %f \n",
					i, achievedRes[0],
					coarseTracker->lastResiduals[0]);
		}


//[ ***step 3*** ] XTL 若trackingIsGood, 并且0层残差比achievedRes小,则把该位姿放入lastF_2_fh, 该光度参数放入aff_g2l,置haveOneGood为true
		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]))
		{
			if(!(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
			{
				flowVecs = coarseTracker->lastFlowIndicators;
				lastF_2_fh = lastF_2_fh_this;
				haveOneGood = true;
			}
		}


		// take over achieved res (always).
		// XTL 若之前有算出一个good的结果，当前算出来的残差项比achievedRes小，则给achievedRes赋值
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
				{
					achievedRes[i] = coarseTracker->lastResiduals[i];	
				}
		}

        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
		{
			break;
		}
	}

	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
	
	if(USE_GT){
		fh->shell->camToTrackingRef = lastF->shell->gtPose.inverse() * fh->shell->gtPose;
		fh->shell->trackingRef = lastF->shell;
		fh->shell->camToWorld = fh->shell->gtPose;
	}
	else{
		fh->shell->camToTrackingRef = lastF_2_fh.inverse();
		fh->shell->trackingRef = lastF->shell;
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
	}


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];  // 第一次跟踪的平均能量值

    if(!setting_debugout_runquiet){
        printf("Coarse Tracker tracked Res %f!", achievedRes[0]);// 2021.12.18
		printf("\n");
	}



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure[0] << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}

	idxUseLast = coarseTracker->idxUse;

	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}
// 2020.07.13 yzk shiyong


void FullSystem::ProjectCloud(std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>>& cld, 
	SE3 T_lb3_lidar,bool debug,float res,bool before)
{
	std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> vCloud[CAM_NUM];

	cv::Mat imageFrame[CAM_NUM];
	for(int idx:idx_use)
	{
		SE3 Tcl =  T_lb_c[idx].inverse()*T_lb3_lidar;
		Rcl[idx] = Tcl.rotationMatrix().cast<double>();
		tcl[idx] = Tcl.translation().cast<double>();
		if(debug)
			imageFrame[idx] = frameHessians.back()->frame[idx]->getCvImages(0);
	}
	for (size_t i = 0; i < cld.size(); ++i)
	{
		for(int idx:idx_use)
		{
			Eigen::Vector4d temp;
			temp.head<3>() = cld[i].head<3>();

			temp.head<3>() = Rcl[idx] * temp.head<3>() + tcl[idx];

			if(temp(2, 0) < 0.1)
				continue;

			// 归一化平面
			float u = (float)(temp(0, 0) / temp(2, 0));
			float v = (float)(temp(1, 0) / temp(2, 0));
			// 像素平面
			float Ku = u * Hcalib.fxl(idx) + Hcalib.cxl(idx);
			float Kv = v * Hcalib.fyl(idx) + Hcalib.cyl(idx);

			if((int)Ku<4 || (int)Ku>=wG[0]-5 || (int)Kv<4 || (int)Kv>hG[0]-4 || !maskG[idx][0][(int)Ku+(int)Kv*wG[0]]) 
				continue;

			if(debug)
				draw(imageFrame[idx],(int)Kv,(int)Ku,cv::Vec3b(255,255,255),1);
			if(Ku<left) left=(int)Ku;
			if(Ku>right) right=(int)Ku;
			if(Kv<up) up=(int)Kv;
			if(Kv>down) down=(int)Kv;

			temp(0, 0) = (double)Ku;
			temp(1, 0) = (double)Kv;
			temp(2, 0) = temp(2, 0);
			temp(3, 0) = i;

			vCloud[idx].push_back(temp);
		}
	}
	for(int i:idx_use)
	{
		cloudPixel[i]=vCloud[i];
		if(debug)
		{
			char str[300];
			if(before)
				sprintf(str, "/media/jeff/DATA/extrinsic/%d[%d]BeforeOpt:res:%f.png", 
					frameHessians.back()->frame[i]->shell->id, i, res);
			else
				sprintf(str, "/media/jeff/DATA/extrinsic/%d[%d]AfterOpt:res:%f.png", 
					frameHessians.back()->frame[i]->shell->id, i, res);
			cv::imwrite(std::string(str), imageFrame[i]);
			imageFrame[i].release();
		}
	}
}


bool FullSystem::ProjectCloud(std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>>& cld)
{
	std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> vCloud[CAM_NUM];

	Eigen::Matrix4d T_lidar_lb3 = xyzrpy2T(x_lidar_lb3);
	SE3 T_lidar_lb = SE3(T_lidar_lb3.block<3,3>(0,0),T_lidar_lb3.topRightCorner<3, 1>());

	for(int idx:idx_use)
	{
		SE3 Tcl =  (T_lidar_lb * T_lb_c[idx]).inverse();
		Rcl[idx] = Tcl.rotationMatrix().cast<double>();
		tcl[idx] = Tcl.translation().cast<double>();
	}
	for (size_t i = 0; i < cld.size(); ++i)
	{
		for(int idx:idx_use)
		{
			Eigen::Vector4d temp;
			temp.head<3>() = cld[i].head<3>();

			temp.head<3>() = Rcl[idx] * temp.head<3>() + tcl[idx];

			if(temp(2, 0) < 0.1)
				continue;

			// 归一化平面
			float u = (float)(temp(0, 0) / temp(2, 0));
			float v = (float)(temp(1, 0) / temp(2, 0));
			// 像素平面
			float Ku = u * Hcalib.fxl(idx) + Hcalib.cxl(idx);
			float Kv = v * Hcalib.fyl(idx) + Hcalib.cyl(idx);

			if((int)Ku<4 || (int)Ku>=wG[0]-5 || (int)Kv<4 || (int)Kv>hG[0]-4 || !maskG[idx][0][(int)Ku+(int)Kv*wG[0]]) 
				continue;

			if(Ku<left) left=(int)Ku;
			if(Ku>right) right=(int)Ku;
			if(Kv<up) up=(int)Kv;
			if(Kv>down) down=(int)Kv;

			temp(0, 0) = (double)Ku;
			temp(1, 0) = (double)Kv;
			temp(2, 0) = temp(2, 0);
			temp(3, 0) = i;

			vCloud[idx].push_back(temp);
		}
	}
	for(int i:idx_use)
	{
		cloudPixel[i]=vCloud[i];
		if(cloudPixel[i].size()==0)
			return false;
	}
	return true;
}



// bool FullSystem::ProjectCloud_ijrr(std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> cld[])
// {
// 	std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloud[CAM_NUM];

// 	for(int idx:idx_use)
// 	{
// 		for(size_t i = 0; i < cld[idx].size(); i++)
// 		{
// 			Eigen::Vector3d temp = cld[idx][i];

// 			if(temp(2, 0) < 0.1)
// 				continue;

// 			// 归一化平面
// 			float u = (float)(temp(0, 0) / temp(2, 0));
// 			float v = (float)(temp(1, 0) / temp(2, 0));
// 			// 像素平面
// 			float Ku = u * Hcalib.fxl(idx) + Hcalib.cxl(idx);
// 			float Kv = v * Hcalib.fyl(idx) + Hcalib.cyl(idx);

// 			if((int)Ku<4 || (int)Ku>=wG[0]-5 || (int)Kv<4 || (int)Kv>hG[0]-4 || !maskG[idx][0][(int)Ku+(int)Kv*wG[0]]) 
// 				continue;

// 			if(Ku<left) left=(int)Ku;
// 			if(Ku>right) right=(int)Ku;
// 			if(Kv<up) up=(int)Kv;
// 			if(Kv>down) down=(int)Kv;

// 			temp(0, 0) = (double)Ku;
// 			temp(1, 0) = (double)Kv;
// 			temp(2, 0) = temp(2, 0);

// 			vCloud[idx].push_back(temp);
// 		}
// 	}
// 	for(int i:idx_use)
// 	{
// 		cloudPixel[i] = vCloud[i];
// 		if(cloudPixel[i].size()==0)
// 			return false;
// 	}
// }

void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	// 2021.11.13
	for(int i=0;i<cam_num;i++){

		K(0,0) = Hcalib.fxl(i);
		K(1,1) = Hcalib.fyl(i);
		K(0,2) = Hcalib.cxl(i);
		K(1,2) = Hcalib.cyl(i);

		// 遍历关键帧
		for(FrameHessian* host : frameHessians)		// go through all active frames
		{
			// 2021.11.15
			SE3 hostToNew_tmp = T_c_c0[i] * fh->PRE_worldToCam * host->PRE_camToWorld * T_c0_c[i];
			// SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
			Mat33f KRKi = K * hostToNew_tmp.rotationMatrix().cast<float>() * K.inverse();
			Vec3f Kt = K * hostToNew_tmp.translation().cast<float>();

			Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure[i], fh->ab_exposure[i], host->aff_g2l(i), fh->aff_g2l(i)).cast<float>(); // 2021.12.18

			for(ImmaturePoint* ph : host->frame[i]->immaturePoints)
			{
				ph->traceOn(fh->frame[i], KRKi, Kt, aff, &Hcalib, false );
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
				trace_total++;
			}
		}
	}
	// 2021.11.13
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}


//@ 激活提取出来的待激活的点
void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr;
	if(setting_useMultualPBack)
		tr = new ImmaturePointTemporaryResidual[frameHessians.size()*5];
	else
		tr = new ImmaturePointTemporaryResidual[frameHessians.size()];

	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}

void FullSystem::activatePointsMT()
{
	//currentMinActDist 初值为 2 

	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

	bool debugsave = false;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	FrameHessian* newestHs = frameHessians.back();


	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(cam_num*5000); // 待激活的点

	for(int idx=0;idx<cam_num;idx++)
	{
		coarseDistanceMap->makeK(&Hcalib,idx);
		// 将之前关键帧上的点投影到该帧上，构建距离地图。
		coarseDistanceMap->makeDistanceMap(frameHessians, newestHs, idx);

		for(FrameHessian* host : frameHessians)		// go through all active frames
		{
			// if(host->w[idx]==false)	continue;
			//if(host == newestHs) continue;  // 2020.07.18 yzk zhushi
			SE3 fhToNew = T_c_c0[idx] * newestHs->PRE_worldToCam * host->PRE_camToWorld * T_c0_c[idx];
			// 第0层到1层
			Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
			Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

			int impt_OUTLIER=0,impt_cantActivate=0,impt_canActivate=0,impt_large_dist=0;


			for(unsigned int i=0;i<host->frame[idx]->immaturePoints.size();i+=1)
			{
				ImmaturePoint* ph = host->frame[idx]->immaturePoints[i];
				ph->idxInImmaturePoints = i;

				if(ph->isFromSensor == false && host == newestHs)
					continue;
				
				// 删掉逆深度的最大值为无穷的点（创建point时，其逆深度最大值为无穷）或者IPS_OUTLIER的点
				// 这两种状态的改变均在traceon中
				if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
				{
					//	immature_invalid_deleted++;
					// remove point.
					delete ph;
					host->frame[idx]->immaturePoints[i]=0;
					continue;
				}
				
				//* 未成熟点的激活条件
				// 2019.12.06 yzk
				bool canActivate = (ph->lastTraceStatus == IPS_GOOD
						|| ph->lastTraceStatus == IPS_SKIPPED
						|| ph->lastTraceStatus == IPS_BADCONDITION
						|| ph->lastTraceStatus == IPS_OOB )
								&& ph->lastTracePixelInterval < 8
								&& ph->quality > setting_minTraceQuality
								&& (ph->idepth_max+ph->idepth_min) > 0;


				// if I cannot activate the point, skip it. Maybe also delete it.
				if(!canActivate)
				{
					//* 删除被边缘化帧上的, 和OOB点
					// if point will be out afterwards, delete it instead.
					if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
					{
						// immature_notReady_deleted++;
						delete ph;
						host->frame[idx]->immaturePoints[i]=0;
					}
					// immature_notReady_skipped++;
					continue;
				}


				// see if we need to activate point due to distance map.
				Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
				int u = ptp[0] / ptp[2] + 0.5f;
				int v = ptp[1] / ptp[2] + 0.5f;

				if((u > 0 && v > 0 && u < wG[1] && v < hG[1])&&(maskG[idx][1][u+v*wG[1]])) // 2021.11.25
				{
					float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

					// XTL my_type是在选点时确定的，代表点是在哪一层被选择的 currentMinActDist由预期的点数量和实际的ef->npoints数量决定
					if(dist>=currentMinActDist* ph->my_type) // 点越多, 距离阈值越大
					{
						coarseDistanceMap->addIntoDistFinal(u,v); // 2020.07.18 yzk shiyong
						toOptimize.push_back(ph);
					}
				}
				else
				{
					delete ph;
					host->frame[idx]->immaturePoints[i]=0;
				}
			}
		}
	}

		//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
		//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);
	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	// XTL：构建toOptimize中未成熟点对所有关键帧的残差，优化其逆深度，若好的残差足够（1个）则把该未成熟点放入optimized，并且把残差放入该点的residuals当中
	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);
	else
	{
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);
	}

	int cnt = 0;

	// 画出选择结果
	MinimalImageB3* img;
	bool plot = false;
	if(plot)
	{
		int w = wG[0];
		int h = hG[0];
		img = new MinimalImageB3(w,h);
		for(int i = 0; i < w; i++)
		{
			for(int j = 0; j < h; j++)
			{
				int idx = i + w*j;
				float c = frameHessians.back()->frame[0]->dI[idx][0]*0.7;
				if(c>255) c=255;
				img->at(idx) = Vec3b(c,c,c);
			}
		}
	}

	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];
		// XTL：若该点优化之后被认为是好的点，则删除原来的未成熟点，并将该点放入主帧的pointHessians当中，残差项加入ef
		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			cnt++;
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			ef->insertPoint(newpoint);		// 能量函数中插入点
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);		// 能量函数中插入残差
			assert(newpoint->efPoint != 0);
			if(plot)
				if(ph->host->_cam_idx==0)
					img->setPixelCirc(newpoint->u,newpoint->v,Vec3b(0,255,0));
			delete ph;
		}
		// XTL：若该点优化后被认为是不好的，或者上次追踪状态为OOB时，就将原来的未成熟点删掉
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
			delete ph;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}

	if(plot)
	{
		char str[100];
		sprintf(str, "delete point");
		IOWrap::displayImage(str, img);
		IOWrap::waitKey(0);
		IOWrap::closeAllWindows();
		delete img;
	}

	//
	// std::cout<<"toOptimize:"<<toOptimize.size()<<".Optimized ratio="<<100*cnt/toOptimize.size()<<"%"<<std::endl;

	// XTL 将要删除的点移到immaturePoints末尾，然后pop掉
	for(FrameHessian* host : frameHessians)
	{
		for(int idx=0;idx<cam_num;idx++){
			for(int i=0;i<(int)host->frame[idx]->immaturePoints.size();i++)
			{
				if(host->frame[idx]->immaturePoints[i]==0)
				{
					host->frame[idx]->immaturePoints[i] = host->frame[idx]->immaturePoints.back();
					host->frame[idx]->immaturePoints.pop_back();
					i--;
				}
			}
		}
	}


}





void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

//@ 标记要移除点的状态, 边缘化or丢掉
// TODO !!!!2021.11.17
void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<frame_hessian*> fhsToMargPoints;
	{
		// XTL：遍历关键帧，若标记为了边缘化帧，就放入fhsToMargPoints当中
		for(int i=0; i<(int)frameHessians.size();i++){
			if(frameHessians[i]->frame[0]->flaggedForMarginalization){
				for(int idx=0;idx<cam_num;idx++){
					fhsToMargPoints.push_back(frameHessians[i]->frame[idx]);
				}
			}
		}
	}

	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	// XTL：遍历关键帧上的关键点，若其逆深度小于0或者残差项数量为0,就删掉该点，若边缘化掉关键帧后点残差数量变得过少/该点在最近两帧的追踪状态都较差/该点的主帧被边缘化了，就
	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == frameHessians.back()) continue; // 2020.07.18 yzk shiyong

		for(unsigned int idx=0;idx<cam_num;idx++){

			for(unsigned int i=0;i<host->frame[idx]->pointHessians.size();i++)
			{
				PointHessian* ph = host->frame[idx]->pointHessians[i];
				if(ph==0) continue;

				//* 丢掉相机后面, 没有残差的点
				if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
				{
					host->frame[idx]->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					host->frame[idx]->pointHessians[i]=0;
					flag_nores++;
				}
				//* 把边缘化的帧上的点, 以及受影响较大的点标记为边缘化or删除
				else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->frame[idx]->flaggedForMarginalization)
				{
					flag_oob++;
					// XTL：若在这些帧还没有边缘化时，这个点残差项数量足够，则遍历这个点的残差项，重置（能量置零，state=IN,New_state=OUTLIER）,并进行残差、导数的计算
					// XTL：若其在计算之后发现点的残差是IN，则将残差项边缘化保留下来
					// XTL：若点的逆深度H矩阵大于阈值，则标记为边缘化的点，否则标记为PS_DROP
					if(ph->isInlierNew())
					{
						flag_in++;
						int ngoodRes=0;
						float Hdd_acc = 0;
						for(PointFrameResidual* r : ph->residuals)
						{
							r->resetOOB();
							r->linearize(&Hcalib);
							r->efResidual->isLinearized = false;
							r->applyRes(true);
							// 若残差的efResidual是激活状态，则进行边缘化
							if(r->efResidual->isActive())
							{
								r->efResidual->fixLinearizationF(ef);
								ngoodRes++;
							}
						}

						//* 如果逆深度的协方差很大直接扔掉, 小的边缘化掉
						if(ph->idepth_hessian > setting_minIdepthH_marg)
						{
							flag_inin++;
							ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
							host->frame[idx]->pointHessiansMarginalized.push_back(ph);
						}
						else
						{
							ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
							host->frame[idx]->pointHessiansOut.push_back(ph);
						}

					}
					//* 不是内点直接扔掉
					else
					{
						host->frame[idx]->pointHessiansOut.push_back(ph);
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
					}

					host->frame[idx]->pointHessians[i]=0; // 把点给删除
				}
			}

			//* 删除边缘化或者删除的点
			for(int i=0;i<(int)host->frame[idx]->pointHessians.size();i++)
			{
				if(host->frame[idx]->pointHessians[i]==0)
				{
					host->frame[idx]->pointHessians[i] = host->frame[idx]->pointHessians.back();
					host->frame[idx]->pointHessians.pop_back();
					i--;
				}
			}
			if(host->frame[idx]->flaggedForMarginalization)
				assert((int)host->frame[idx]->pointHessians.size()==0);
		}
	}
}

/********************************
 * @ function:
 * 
 * @ param: 	image		标定后的辐照度和曝光时间
 * @			id			
 * 
 * @ note: start from here
 *******************************/
void FullSystem::addActiveFrame(std::vector<ImageAndExposure*> image, int id )
{
	//[ ***step 1*** ] track线程锁
    if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);


	//[ ***step 2*** ] 创建FrameHessian和FrameShell, 并进行相应初始化, 并存储所有帧
	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();
	FrameShell* shell = new FrameShell();
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.

	for(int i=0;i<cam_num;i++){
		shell->aff_g2l[i] = AffLight(0,0);
	}

    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image[0]->timestamp;
    shell->incoming_id = id;
	fh->shell = shell;				//!< 帧的"壳", 保存一些不变的,要留下来的量
	allFrameHistory.push_back(shell);  // 只把简略的shell存起来

	//[ ***step 3*** ] 得到曝光时间, 生成金字塔, 计算整个图像梯度
	// =========================== make Images / derivatives etc. =========================
	for(int i=0;i<cam_num;i++)
		fh->ab_exposure[i] = image[i]->exposure_time;

	for(int i=0;i<cam_num;i++){
		frame_hessian* fh_tmp = new frame_hessian();
		fh_tmp->_cam_idx = i;
		fh_tmp->makeImages(image[i]->image, &Hcalib);
		fh_tmp->shell = shell;
		fh_tmp->fh0 = fh;
		fh->frame[i] = fh_tmp;
	}
	
	//[ ***step 4*** ] 进行初始化
	if(!initialized)
	{
		if(coarseInitializer->frameID < 0)
		{
			if(!useOnlyCamera)
			{
                if(1/*setting_useNCLT*/)
                {
                    if(ProjectCloud(Clouds.front()))
                    {
                        coarseInitializer->setFirstFromLidar(&Hcalib, fh, this, cloudPixel);
			            initialized = true;
                    }
                }
                else
                {
                    // std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> cloud_ijrr[CAM_NUM];
                    // for(int i=0;i<cam_num;i++)
                    // 	cloud_ijrr[i] = Cloud_ijrr[i].front();
                    // ProjectCloud_ijrr(cloud_ijrr);
                }
			}
		}
		return;
	}
	else	// do front-end operation.
	{
//[ ***step 5*** ] 对新来的帧进行跟踪, 得到位姿光度, 判断跟踪状态
		// =========================== SWAP tracking reference. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			// 交换参考帧和当前帧的coarseTracker
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker;
			coarseTracker=coarseTracker_forNewKF; 
			coarseTracker_forNewKF=tmp;
		}

		clock_t started = clock();
		Vec4 tres = trackNewCoarse(fh);
		clock_t ended = clock();
		// printf("Time spent in trackNewCoarse is %.2fms\n",1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC));
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
			isLost=true;
            return;
        }
//[ ***step 6*** ] 判断是否插入关键帧
		bool needToMakeKF = false;
		float sceneChange = setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) + 
							setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]);
		// printf("sceneChange=%.2f\n",sceneChange);
		if(setting_keyframesPerSecond > 0)  // 每隔多久插入关键帧
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{

			// BRIGHTNESS CHECK
			needToMakeKF = allFrameHistory.size()== 1 ||
					setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +  // 平移像素位移
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) + 	// 旋转像素位移, 设置为0???
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) > 1 ||// 旋转+平移像素位移
					2*coarseTracker->firstCoarseRMSE < tres[0];		// 误差能量变化太大(最初的两倍)
		}


        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);


		if(needToMakeKF)
		{
			cnt=0;
			if(!useOnlyCamera)
			{
				if(1/*setting_useNCLT*/)
				{
					if(ProjectCloud(Clouds.front()))
						setKeyFrameWeight(fh);
					else
						needToMakeKF=false;
				}
				else
				{
					// std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> cloud_ijrr[CAM_NUM];
					// for(int i=0;i<cam_num;i++)
					// 	cloud_ijrr[i] = Cloud_ijrr[i].front();
					// if(ProjectCloud_ijrr(cloud_ijrr))
					// 	setKeyFrameWeight(fh);
					// else
					// {
					// 	needToMakeKF=false;
					// }
				}
			}
		}
		lock.unlock();
		deliverTrackedFrame(fh, needToMakeKF);
		return;
	}
}

void FullSystem::setKeyFrameWeight(FrameHessian* fh)
{
	memset(fh->w,0,5*sizeof(bool));
	for(int i:idx_use)
		fh->w[i]=1;
	return;
}

//@ 把跟踪的帧, 给到建图线程, 设置成关键帧或非关键帧
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{
	assert(linearizeOperation);
	//! 顺序执行
	if(linearizeOperation) 
	{
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->frame[0]->dI);// 2022.1.11
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );



		if(needKF) makeKeyFrame(fh);
		else makeFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex); // 跟踪和建图同步锁
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);  // 当没有跟踪的图像, 就一直阻塞trackMapSyncMutex, 直到notify
		}

		lock.unlock();
	}
}

//@ 建图线程
void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);   // 没有图像等待trackedFrameSignal唤醒
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();


		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();		// 运行makeKeyFrame是不会影响unmappedTrackedFrames的, 所以解锁
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();  // 结束前唤醒
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)  // 太多了给处理掉
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				for(int i=0;i<cam_num;i++)
					delete fh->frame[i];
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)  // 后面需要关键帧
			{
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

//@ 结束建图线程
void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

//@ 设置成非关键帧
void FullSystem::makeFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex); // 生命周期结束后自动解锁
		assert(fh->shell->trackingRef != 0);
		// mapping时将它当前位姿取出来得到camToWorld
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		// 把此时估计的位姿取出来
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);  // 更新未成熟点(深度未收敛的点)
	if(!setting_useKFselection)
	{
		for(int i=0;i<cam_num;i++)
			delete fh->frame[i];
		delete fh;
	}
}
double FullSystem::ResPixel(
		const float outlierTHSlack,
		int idx, FrameHessian* target,
		float u, float v,
		cv::Mat &Hdd, cv::Mat &bd, int level,
		float mean_diff,
		float idepth)
{
	frame_hessian* host = frameHessians[frameHessians.size()-1]->frame[idx];

	FrameFramePrecalc* precalc = &(host->fh0->targetPrecalc[target->idx]);

    Vec2f px_scaled(u,v);
	// check OOB due to scale angle change.
	float energyLeft=0;
	const Eigen::Vector3f* dIl = target->frame[idx]->dIp[level];
	const Mat33f &PRE_RTll = precalc->PRE_RTll[idx*cam_num+idx];
	const Vec3f &PRE_tTll = precalc->PRE_tTll[idx*cam_num+idx];
	// printf("PRE_tTll.norm()=%f\n",PRE_tTll.norm());
	if(PRE_tTll.norm()<0.5 || PRE_tTll.norm()>1.5)
		return 0;

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

//@ 生成关键帧, 优化, 激活点, 提取点, 边缘化关键帧
void FullSystem::makeKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread
	{	
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l); // 待优化值
	}
	traceNewCoarse(fh); //用这一帧来极线搜索，更新之前帧的未成熟点的状态

	boost::unique_lock<boost::mutex> lock(mapMutex); // 建图锁

	// =========================== Flag Frames to be Marginalized. =========================
	// XTL：标记需要边缘化的帧
	flagFramesForMarginalization(fh);  // TODO 这里没用最新帧，可以改进下

	// =========================== add New Frame to Hessian Struct. =========================
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	setPrecalcValues();	// 每添加一个关键帧都会运行这个来设置位姿, 设置位姿线性化点

	// 2022.11.7
	if(1)//(setting_useNCLT)
	{
		SE3 T_lb3_lidar;
		Eigen::Matrix4d T_lidar_lb3 = xyzrpy2T(x_lidar_lb3_ori);
		SE3 T_lidar_lb = SE3(T_lidar_lb3.block<3,3>(0,0),T_lidar_lb3.topRightCorner<3, 1>());
		T_lb3_lidar = T_lidar_lb.inverse();
		// optExtrinsic(T_lb3_lidar);
		// Vec6 inc = Vec6::Zero();
		// for(int i=0;i<6;i++)
		// 	inc[i] = dist(generator);
		// std::cout<<"inc="<<inc.transpose()<<std::endl;
		// SE3 T_lb3_lidar_new = SE3::exp(inc) * T_lb3_lidar;
		// ProjectCloud(Clouds.front(),T_lb3_lidar_new,0);
		// ProjectCloud(Clouds.front(),T_lb3_lidar,0,0,1);
		ProjectCloud(Clouds.front());
		for(int i=0;i<cam_num;i++)
			bestCloud[i] = cloudPixel[i];
	}
	else
	{
		// std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> cloud_ijrr[CAM_NUM];
		// for(int i=0;i<cam_num;i++)
		// 	cloud_ijrr[i] = Cloud_ijrr[i].front();
		// ProjectCloud_ijrr(cloud_ijrr);
	}
	// 从雷达和图片中提取未成熟点加入impt
	makeNewTraces(fh);
	for(int i=0;i<cam_num;i++){
		for(ImmaturePoint* ph : fh->frame[i]->immaturePoints)
			if(ph->isFromSensor == true){
				ph->lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
			}
	}

	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	// 遍历所有的关键帧上的关键点，构建点与当前帧的残差项，初始化残差项（state_state为IN，点的lastResiduals前移一个单元，最后一个lastResiduals为新的残差项），PointFrameResidual插入到ef当中
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		// std::cout<<"Frame id is:"<<fh1->frameID<<std::endl;
		if(fh1 == fh) continue;
		for(int i=0;i<cam_num;i++)
		{
			for(PointHessian* ph : fh1->frame[i]->pointHessians) 
			{
				if(setting_useMultualPBack)
				{
                    for(int i=0;i<cam_num;i++)
                    {
						ph->lastResiduals[2*i+1] = ph->lastResiduals[2*i]; // 设置上上个残差
                        ph->lastResiduals[2*i].first = 0;
                    }
					FrameFramePrecalc* precalc = &(fh1->targetPrecalc.back());
					Mat33f &KRKi = precalc->PRE_RTll_0[i*cam_num+i];
					Vec3f &Kt = precalc->PRE_tTll_0[i*cam_num+i];
					Vec3f ptp = KRKi * Vec3f(ph->u,ph->v, 1) + Kt*ph->idepth; // host上点除深度
					float Ku = ptp[0] / ptp[2];
					float Kv = ptp[1] / ptp[2];
					bool isIn=false;
					if(Ku>1.1f && Kv>1.1f && Ku<wG[0] && Kv<hG[0] && maskG[i][0][(int)Ku+(int)Kv*wG[0]])
					{
						PointFrameResidual* r = new PointFrameResidual(ph, fh1->frame[i], fh->frame[i]);
						r->setState(ResState::IN);
						ph->residuals.push_back(r);
						ef->insertResidual(r);
						ph->lastResiduals[2*i] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
						numFwdResAdde+=1;
						isIn = true;
					}
					int j = 1, idx;
					bool isSecond = false;
					while(j<3)
					{
						if(isIn)
							break;
						if(!isSecond)
						{
							idx = i+j;
							if(idx>=5)
								idx-=5;
						}
						else
						{
							idx = i-j;
							j++;
							if(idx<0)
								idx+=5;
						}
						isSecond=!isSecond;
						if(!isIn)
						{
							KRKi = precalc->PRE_RTll_0[i*cam_num+idx];
							Kt = precalc->PRE_tTll_0[i*cam_num+idx];
							ptp = KRKi * Vec3f(ph->u,ph->v, 1) + Kt*ph->idepth; // host上点除深度
							Ku = ptp[0] / ptp[2];
							Kv = ptp[1] / ptp[2];
							if(Ku>1.1f && Kv>1.1f && Ku<wG[0] && Kv<hG[0] && maskG[idx][0][(int)Ku+(int)Kv*wG[0]])
							{
								PointFrameResidual* r = new PointFrameResidual(ph, fh1->frame[i], fh->frame[idx]);
								r->setState(ResState::IN);
								ph->residuals.push_back(r);
								ef->insertResidual(r);
								ph->lastResiduals[2*idx] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
								numFwdResAdde+=1;
								isIn=true;
							}
						}
					}
				}
				else
				{
					PointFrameResidual* r = new PointFrameResidual(ph, fh1->frame[i], fh->frame[i]);
					r->setState(ResState::IN);
					ph->residuals.push_back(r);
					ef->insertResidual(r);
					ph->lastResiduals[1] = ph->lastResiduals[0]; // 设置上上个残差
					ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
					numFwdResAdde+=1;
				}
			}
		}
	}

	// =========================== Activate Points (& flag for marginalization). =========================
	// XTL:激活合适的未成熟点
	activatePointsMT();
	// XTL:重新对ef中的frame进行编号;遍历各个frame上的点，放入allPoints，更新该点所有残差项的hostIDX与targetIDX
	ef->makeIDX();

	// =========================== OPTIMIZE ALL =========================
	//fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
	float rmse = optimize(setting_maxOptIterations);

	// =========================== Figure Out if INITIALIZATION FAILED =========================
	/*
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}
	*/

    if(isLost) return;


	// =========================== REMOVE OUTLIER =========================
	removeOutliers();


	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		// XTL：设置coarseTracker_forNewKF的内参
		coarseTracker_forNewKF->makeK(&Hcalib);  // 更新了内参, 因此重新make
		// XTL：设置lastRef为最新一个关键帧，refFrameID为该帧的shell的id，将所有与最新关键帧构建了残差的点的投影坐标记录下来，便于可视化
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians,idxUseLast);
		// XTL：深度图可视化的一些操作
        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}

	debugPlot("post Optimize");

//[ ***step 9*** ] 标记删除和边缘化的点, 并删除&边缘化
	// =========================== (Activate-)Marginalize Points =========================
	flagPointsForRemoval();

	ef->dropPointsF();  // 扔掉drop的点
	
	// 每次设置线性化点都会更新零空间
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
	// 边缘化掉点, 加在HM, bM上
	ef->marginalizePointsF();

//[ ***step 10*** ] 生成新的点
	// =========================== add new Immature points & new residuals =========================
	//makeNewTraces(fh, 0); // 2020.07.18 yzk zhushi

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }

	// =========================== Marginalize Frames =========================
//[ ***step 11*** ] 边缘化掉关键帧
	//* 边缘化一帧要删除or边缘化上面所有点
	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->frame[0]->flaggedForMarginalization)// 2022.1.11
			{marginalizeFrame(frameHessians[i]); i=0;}

	printLogLine();
    //printEigenValLine();

}

//@ 从初始化中提取出信息, 用于跟踪.
void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	//[ ***step 1*** ] 把第一帧设置成关键帧, 加入队列, 加入EnergyFunctional
	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;  // 第一帧增加进地图
	firstFrame->idx = frameHessians.size(); // 赋值给它id (0开始)
	frameHessians.push_back(firstFrame);  	// 地图内关键帧容器
	firstFrame->frameID = allKeyFramesHistory.size();  	// 所有历史关键帧id

	allKeyFramesHistory.push_back(firstFrame->shell); 	// 所有历史关键帧
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues();   		// 设置相对位姿预计算值

	int point_cnt=0;
	for(int idx=0;idx<cam_num;idx++)
		point_cnt += coarseInitializer->numPoints[idx][0];

	float keepPercentage = cam_num * setting_desiredPointDensity / point_cnt;

	if(!setting_debugout_runquiet)
		printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
				(int)(cam_num * setting_desiredPointDensity), point_cnt );

	
	for(int idx:idx_use)
	{
		firstFrame->frame[idx]->pointHessians.reserve(wG[0]*hG[0]*0.2f); // 20%的点数目
		firstFrame->frame[idx]->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f); // 被边缘化
		firstFrame->frame[idx]->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f); // 丢掉的点

		/*
		//[ ***step 2*** ] 求出平均尺度因子
		float sumID=1e-5, numID=1e-5;
		for(int i=0;i<coarseInitializer->numPoints[idx][0];i++)
		{
			//? iR的值到底是啥
			sumID += coarseInitializer->points[idx][0][i].iR; // 第0层点的中位值, 相当于
			numID++;
		}
		float rescaleFactor = 1 / (sumID / numID);  // 求出尺度因子 2020.07.14 yzk zhushi

		randomly sub-select the points I need.
		*/
		// 目标点数 / 实际提取点数
		// float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[idx][0];
		/*
		if(!setting_debugout_runquiet)
			printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
					(int)(setting_desiredPointDensity), coarseInitializer->numPoints[idx][0] );
		*/

		//[ ***step 3*** ] 创建PointHessian, 点加入关键帧, 加入EnergyFunctional
		for(int i=0;i<coarseInitializer->numPoints[idx][0];i++)
		{
			if(rand()/(float)RAND_MAX > keepPercentage) continue; // 如果提取的点比较少, 不执行; 提取的多, 则随机干掉

			Pnt* point = coarseInitializer->points[idx][0]+i;
			ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame->frame[idx],point->my_type, &Hcalib);

			// 2020.07.14 yzk shiyong
			if(point->isFromSensor == true)
			{
				pt->idepth_min = 1 / point->mdepth;
				pt->idepth_max = 1 / point->mdepth;
			}
			// 2020.07.14 yzk shiyong

			if(!std::isfinite(pt->energyTH)) { delete pt; continue; }  // 点值无穷大

			// 创建ImmaturePoint就为了创建PointHessian? 是为了接口统一吧
			//pt->idepth_max=pt->idepth_min=1; // 2020.07.14 yzk zhushi
			PointHessian* ph = new PointHessian(pt, &Hcalib);
			delete pt;
			if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

			ph->setIdepthScaled(point->midepth);
			ph->setIdepthZero(point->midepth);
			ph->hasDepthPrior=true;
			ph->setPointStatus(PointHessian::ACTIVE);

			if(point->isFromSensor == true)
			{
				ph->isFromSensor = true;
				ph->hasDepthPrior = true;
			}
			else
				ph->isFromSensor = false;

			firstFrame->frame[idx]->pointHessians.push_back(ph);
			ef->insertPoint(ph);
		}
	}
	
	//[ ***step 4*** ] 设置第一帧和最新帧的待优化量, 参考帧
	SE3 firstToNew = coarseInitializer->thisToNext;

	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();

		for(int i=0;i<cam_num;i++)
			firstFrame->shell->aff_g2l[i] = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		for(int i=0;i<cam_num;i++)
			newFrame->shell->aff_g2l[i] = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();
	}

	initialized=true;
	int cnt = 0;
	for(int i=0;i<cam_num;i++)
		cnt += firstFrame->frame[i]->pointHessians.size();
	
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)cnt);
}


void FullSystem::drawPointDistribution(int idx, FrameHessian* fh, std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> Clouds)
{
	cv::Mat imageFrame = fh->frame[idx]->getCvImages(0);

	int w = wG[0];
    int h = hG[0];

	for(int i = 0; i < Clouds.size(); i++)
        draw(imageFrame, (int)Clouds[i](0, 0), (int)Clouds[i](1, 0),cv::Vec3b(0,0,0),1);

	char str[300];
    sprintf(str, "/home/jeff/workspace/catkin_ws/src/omni_DSO_lidar/lidarPoint/[%d]%d.png", idx, fh->frame[idx]->shell->id);
    cv::imwrite(std::string(str), imageFrame);
    imageFrame.release();
}

// 2020.09.28 yzk
void FullSystem::setMask(cv::Mat &currentFrame, int Ku, int Kv)
{
	for(int i = Ku - pixelSelector->currentPotential; i <= Ku + pixelSelector->currentPotential; i++)
    //for(int i = Ku-15; i <= Ku+15; i++)
    {
        for(int j = Kv-1; j <= Kv+1; j++)
        {
        	if(j < hG[0] && j >= 0 && i < wG[0] && i>=0)
            	currentFrame.at<uchar>(j, i) = 1;
        }
    }

    /*
    for(int i = Ku-16; i <= Ku+16; i++)
    {
        currentFrame.at<uchar>(Kv-2, i) = 1;
        currentFrame.at<uchar>(Kv+2, i) = 1;
    }

    for(int j = Kv-15; j <= Kv+15; j++)
    {
        currentFrame.at<uchar>(j, Ku-2) = 1;
        currentFrame.at<uchar>(j, Ku+2) = 1;
    }
    */
}
// 2020.09.28 yzk

// XTL：遍历新帧，选点创建未成熟点
void FullSystem::makeNewTraces(FrameHessian* newFrame)
{
	pixelSelector->allowFast = true;
	int imageArea = wG[0] * hG[0];

	bool debugsave = false;

	int lidarArea = (right - left) * (down - up);			

	for(int idx=0;idx<cam_num;idx++)
	{
		int numPointsTotal = 0;
		int numPointLidar = 0;
		int numPointMonocular = 0;

		cv::Mat mask_ = cv::Mat::zeros(hG[0], wG[0], CV_8UC1);

		std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> vCloudPixel = bestCloud[idx];

		// assert(fabs(newFrame->shell->timestamp - qTimeLidarCloud.front()) < 0.01); TODO!!
		//if(fabs(newFrame->shell->timestamp - qTimeLidarCloud.front()) < 0.01)
		//{
			if(vCloudPixel.size()!=0&&newFrame->w[idx])
			{
				selectionMapFromLidar = new float[vCloudPixel.size()];
				numPointLidar = pixelSelector->makeMapsFromLidar(newFrame->frame[idx], selectionMapFromLidar, ((float)lidarArea/(float)imageArea) * setting_desiredImmatureDensity, 1, false, 1, vCloudPixel, newFrame->shell->id);
			}
			// drawPointDistribution(idx);
			// 2020.09.28 yzk
			_top = up; _bottom = down; _left = left; _right = right;
			if(addFeaturePoint&&newFrame->w[idx])
				numPointMonocular = pixelSelector->makeMaps(newFrame->frame[idx], selectionMap, setting_desiredImmatureDensity);
			numPointsTotal = numPointLidar + numPointMonocular;
		//}

		// printf("NumPoint Vision of [%d]=%d\n",idx,numPointMonocular);
		// numPointsTotal = numPointLidar + numPointMonocular;
			
		newFrame->frame[idx]->pointHessians.reserve(numPointsTotal*1.2f);
		newFrame->frame[idx]->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
		newFrame->frame[idx]->pointHessiansOut.reserve(numPointsTotal*1.2f);


		if(vCloudPixel.size()!=0&&newFrame->w[idx])
		{
			for(int i = 0; i < vCloudPixel.size(); i++)
			{
				if(selectionMapFromLidar[i]==0) continue;

				ImmaturePoint* impt = new ImmaturePoint(vCloudPixel[i](0, 0),vCloudPixel[i](1, 0),newFrame->frame[idx], selectionMapFromLidar[i], &Hcalib);

				impt->idepth_fromSensor = 1.0 / vCloudPixel[i](2, 0);
				impt->idepth_max = impt->idepth_fromSensor * 1.0;
				impt->idepth_min = impt->idepth_fromSensor * 1.0;

				impt->isFromSensor = true;

				if(!std::isfinite(impt->energyTH)) delete impt;  // 投影得到的不是有穷数
				else{ 
					newFrame->frame[idx]->immaturePoints.push_back(impt);
					setMask(mask_, vCloudPixel[i](0, 0), vCloudPixel[i](1, 0));
				}
			}
			delete[] selectionMapFromLidar;
		}


		// 2020.09.28 yzk
		if(addFeaturePoint&&newFrame->w[idx])
			for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
			for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
			{
				int i = x+y*wG[0];
				if(selectionMap[i]==0) continue;

				ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame->frame[idx], selectionMap[i], &Hcalib);

				impt->isFromSensor = false;

				if(!std::isfinite(impt->energyTH) || mask_.at<uchar>(y, x) == 1) 
					delete impt;  // 投影得到的不是有穷数
				else{ 
					newFrame->frame[idx]->immaturePoints.push_back(impt);
					setMask(mask_, x, y);
				}

			}

		mask_.release();
		// 2020.09.28 yzk

	}
}

//* 计算frameHessian的预计算值, 和状态的delta值
//@ 设置关键帧之间的关系
void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size()); // 每个目标帧预运算容器, 大小是关键帧数
		for(unsigned int i=0;i<frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib); // 计算Host 与 target之间的变换关系
	}

	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l[0].a,
                allKeyFramesHistory.back()->aff_g2l[0].b,// 2021.12.18
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l(0).a << " "  <<
				frameHessians.back()->aff_g2l(0).b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n"; // 2021.12.17
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}

}
