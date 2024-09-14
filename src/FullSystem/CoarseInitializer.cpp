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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	// 2021.11.14
	for(int i=0;i<cam_num;i++){
		for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
		{
			points[i][lvl] = 0;
			numPoints[i][lvl] = 0;
		}
	}
	// 2021.11.14

	frameID=-1;
	fixAffine=true;
	printDebug=false;

	//! 这是 		尺度偏好!! 经验值,代表作者认为不同的变量在每次迭代过程中的更新量
	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	// 2021.11.14
	for(int i=0;i<cam_num;i++){
		for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
		{
			if(points[i][lvl] != 0) delete[] points[i][lvl];
		}
	}
	// 2021.11.14
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;

	// 2021.11.23
	std::vector<MinimalImageB3*> _iRImg(CAM_NUM);
	int wl = w[lvl], hl = h[lvl];
	//MinimalImageB3 iRImg(hl,wl);

	for(int idx=0;idx<cam_num;idx++){
		_iRImg[idx] = new MinimalImageB3(hl,wl);
		_iRImg[idx]->setBlack();
		// iRImg.setBlack();

		Eigen::Vector3f* colorRef = firstFrame->frame[idx]->dIp[lvl];

		for(int x=0;x<hl;x++)
		{
			for(int y=0;y<wl;y++)
			{
				// iRImg.at(x,y) = Vec3b(colorRef[x*wl+y][0],colorRef[x*wl+y][0],colorRef[x*wl+y][0]);
				_iRImg[idx]->at(x,y) = Vec3b(colorRef[x*wl+y][0],colorRef[x*wl+y][0],colorRef[x*wl+y][0]);
			}
		}

		// 2021.11.16
		int npts = numPoints[idx][lvl];

		float nid = 0, sid=0;
		for(int i=0;i<npts;i++)
		{
			Pnt* point = points[idx][lvl]+i;
			if(point->isGood)
			{
				nid++;
				sid += point->iR;
			}
		}
		float fac = nid / sid;

		for(int i=0;i<npts;i++)
		{
			Pnt* point = points[idx][lvl]+i;

			if(!point->isGood)
				// iRImg.setPixel9(point->v+0.5f,point->u+0.5f,Vec3b(0,0,0));
				_iRImg[idx]->setPixel9(point->v+0.5f,point->u+0.5f,Vec3b(0,0,0));

			else
				// iRImg.setPixel9(point->v+0.5f,point->u+0.5f,makeRainbow3B(point->iR*fac));
				_iRImg[idx]->setPixel9(point->v+0.5f,point->u+0.5f,makeRainbow3B(point->iR*fac));

		}
		// _iRImg[idx] = iRImg.getClone();
	}
	//IOWrap::displayImage("idepth-R", &iRImg, false);
	for(IOWrap::Output3DWrapper* ow : wraps)
		ow->pushDepthImage(_iRImg);
	// 2021.11.10
	// 2021.12.02
	for(int i=0;i<cam_num;i++)
		delete 	_iRImg[i];
	// 2021.12.02
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
	//	float factori = 1.0f/factor;
	//	float factori2 = factori*factori;
	//
	//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	//	{
	//		int npts = numPoints[lvl];
	//		Pnt* ptsl = points[lvl];
	//		for(int i=0;i<npts;i++)
	//		{
	//			ptsl[i].iR *= factor;
	//			ptsl[i].idepth_new *= factor;
	//			ptsl[i].lastHessian *= factori2;
	//		}
	//	}
	//	thisToNext.translation() *= factori;

	return factor;
}

// IR是逆深度的均值，尺度收敛到IR


//* 低层计算高层, 像素值和梯度
void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];
		// 使用上一层得到当前层的值
		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
		// 根据像素计算梯度
		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}

// 2019.11.15 yzk
void CoarseInitializer::setFirstFromLidar(CalibHessian* HCalib, FrameHessian* newFrameHessian, FullSystem* fullSystem,
			std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> cloudPixel[])
{
	//[ ***step 1*** ] 计算图像每层的内参
	makeK(HCalib);

	firstFrame = newFrameHessian;

	PixelSelector sel(w[0],h[0]); // 像素选择

	int lidarArea = (fullSystem->right - fullSystem->left) * (fullSystem->down - fullSystem->up);

	for(int _idx:idx_use){
		std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> vCloudPixel = cloudPixel[_idx];

		// printf("Num points:%d\n",cloudPixel[_idx].size());

		float* statusMap = new float[vCloudPixel.size()];
		bool* statusMapB = new bool[vCloudPixel.size()];

		float densities[] = {0.03,0.05,0.15,0.5,1}; // 不同层取得点密度

		for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
		{
			std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloudPixelLvL;
		//[ ***step 2*** ] 针对不同层数选择大梯度像素, 第0层比较复杂1d, 2d, 4d大小block来选择3个层次的像素
			sel.currentPotential = 3; // 设置网格大小，3*3大小格
			int npts; // 选择的像素数目
			// TODO why < 10????
			if(lvl == 0) // 第0层提取特征像素
			{
				npts = sel.makeMapsFromLidar(firstFrame->frame[_idx], statusMap, densities[lvl]*lidarArea, 1, false, 2, vCloudPixel);
				if(npts<10)
					break;
			}
			else  // 其它层则选出goodpoints
			{
				for(int i = 0; i < vCloudPixel.size(); i++)
				{
					Eigen::Vector3d tempPoint = vCloudPixel[i].head<3>();
					int tempLvl = lvl;
					while(tempLvl>0)
					{
						tempPoint(0, 0) = tempPoint(0, 0)/2;
						tempPoint(1, 0) = tempPoint(1, 0)/2;
						tempLvl--;
					}
					vCloudPixelLvL.push_back(tempPoint);
				}
				//npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]); 2020.07.13 yzk zhushi
				npts = makePixelStatusFromLidar(firstFrame->frame[_idx]->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*lidarArea, 5, 1, vCloudPixelLvL);
			}

			// 如果点非空, 则释放空间, 创建新的
			if(points[_idx][lvl] != 0) delete[] points[_idx][lvl];
			points[_idx][lvl] = new Pnt[npts];

			// set idepth map to initially 1 everywhere.
			int wl = w[lvl], hl = h[lvl]; // 每一层的图像大小
			Pnt* pl = points[_idx][lvl];  // 每一层上的点
			int nl = 0;
			// 要留出pattern的空间, 2 border
			//[ ***step 3*** ] 在选出的像素中, 添加点信息
			for(int i = 0; i < vCloudPixel.size(); i++)
			{
				assert(vCloudPixel[i](2, 0)>0);

				if((lvl!=0 && statusMapB[i]) || (lvl==0 && statusMap[i] != 0))
				{
					if(lvl==0)
					{
						pl[nl].u = vCloudPixel[i](0, 0);
						pl[nl].v = vCloudPixel[i](1, 0);
					}
					else
					{
						pl[nl].u = vCloudPixelLvL[i](0, 0);
						pl[nl].v = vCloudPixelLvL[i](1, 0);
					}

					pl[nl].isGood = true;
					pl[nl].energy.setZero();
					pl[nl].lastHessian = 0;
					pl[nl].lastHessian_new = 0;
					pl[nl].my_type = (lvl!=0) ? 1 : statusMap[i];

					pl[nl].mdepth = vCloudPixel[i](2, 0);
					pl[nl].midepth = 1 / pl[nl].mdepth;
					pl[nl].idepth = pl[nl].midepth;
					pl[nl].iR = pl[nl].midepth;

					pl[nl].isFromSensor = true;

					Eigen::Vector3f* cpt;
					if(lvl==0)
						cpt = firstFrame->frame[_idx]->dIp[lvl] + (int)vCloudPixel[i](0, 0) + (int)vCloudPixel[i](1, 0)*w[lvl]; // 该像素梯度
					else
						cpt = firstFrame->frame[_idx]->dIp[lvl] + (int)vCloudPixelLvL[i](0, 0) + (int)vCloudPixelLvL[i](1, 0)*w[lvl];

					float sumGrad2=0;
					// 计算pattern内像素梯度和
					for(int idx=0;idx<patternNum;idx++)
					{
						int dx = patternP[idx][0]; // pattern 的偏移
						int dy = patternP[idx][1];
						float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
						sumGrad2 += absgrad;
					}

					// float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
					// pl[nl].outlierTH = patternNum*gth*gth;
					//! 外点的阈值与pattern的大小有关, 一个像素是12*12
					pl[nl].outlierTH = patternNum*setting_outlierTH;
					nl++;
					assert(nl <= npts);
				}
			}

			numPoints[_idx][lvl]=nl; // 点的数目,  去掉了一些边界上的点
		}
		delete[] statusMap;
		delete[] statusMapB;
		//[ ***step 4*** ] 计算点的最近邻和父点
		makeNN(_idx);
	}

	// 参数初始化

	thisToNext=SE3();
	snapped = false;
	frameID = snappedAt = 0;

	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();

}
// 2019.11.15 yzk

//@ 计算每个金字塔层的相机参数
void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	for(int i=0;i<cam_num;i++){
		fx[i][0] = HCalib->fxl(i);
		fy[i][0] = HCalib->fyl(i);
		cx[i][0] = HCalib->cxl(i);
		cy[i][0] = HCalib->cyl(i);
		// 求各层的K参数
		for (int level = 1; level < pyrLevelsUsed; ++ level)
		{
			
			w[level] = w[0] >> level;
			h[level] = h[0] >> level;
			fx[i][level] = fx[i][level-1] * 0.5;
			fy[i][level] = fy[i][level-1] * 0.5;
			//* 0.5 offset 看README是设定0.5到1.5之间积分表示1的像素值？
			cx[i][level] = (cx[i][0] + 0.5) / ((int)1<<level) - 0.5;
			cy[i][level] = (cy[i][0] + 0.5) / ((int)1<<level) - 0.5;
		}
		// 求K_inverse参数
		for (int level = 0; level < pyrLevelsUsed; ++ level)
		{
			K[i][level]  << fx[i][level], 0.0, cx[i][level], 0.0, fy[i][level], cy[i][level], 0.0, 0.0, 1.0;
			Ki[i][level] = K[i][level].inverse();
			fxi[i][level] = Ki[i][level](0,0);
			fyi[i][level] = Ki[i][level](1,1);
			cxi[i][level] = Ki[i][level](0,2);
			cyi[i][level] = Ki[i][level](1,2);
		}
	}
}



//@ 生成每一层点的KDTree, 并用其找到邻近点集和父点 
void CoarseInitializer::makeNN(int cam_idx)
{
	const float NNDistFactor=0.05;
	// 第一个参数为distance, 第二个是datasetadaptor, 第三个是维数
	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS]; // 每层建立一个点云
	KDTree* indexes[PYR_LEVELS]; // 点云建立KDtree
	//* 每层建立一个KDTree索引二维点云
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		// 2021.11.14
		pcs[i] = FLANNPointcloud(numPoints[cam_idx][i], points[cam_idx][i]); // 二维点点云
		// 2021.11.14
		// 参数: 维度, 点数据, 叶节点中最大的点数(越大build快, query慢)
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[cam_idx][lvl];
		int npts = numPoints[cam_idx][lvl];

		int ret_index[nn];  // 搜索到的临近点
		float ret_dist[nn]; // 搜索到点的距离
		// 搜索结果, 最近的nn个和1个
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v); // 当前点
			// 使用建立的KDtree, 来查询最近邻
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			//* 给每个点的neighbours赋值
			for(int k=0;k<nn;k++)
			{
				pts[i].neighbours[myidx]=ret_index[k]; // 最近的索引
				float df = expf(-ret_dist[k]*NNDistFactor); // 距离使用指数形式
				sumDF += df; // 距离和
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			// 对距离进行归10化,,,,,
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;

			//* 高一层的图像中找到该点的父节点
			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f); // 换算到高一层
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0]; // 父节点
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor); // 到父节点的距离(在高层中)

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[cam_idx][lvl+1]);
			}
			else  // 最高层没有父节点
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

