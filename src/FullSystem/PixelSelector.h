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


 

namespace dso
{


const float minUseGrad_pixsel = 10;

//@ 对于高层(0层以上)选择梯度最大的位置点
template<int pot>
inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, float THFac)
{

	memset(map_out, 0, sizeof(bool)*w*h);

	int numGood = 0;
	for(int y=1;y<h-pot;y+=pot)  // 每隔一个pot遍历
	{
		for(int x=1;x<w-pot;x+=pot)
		{
			int bestXXID = -1; // gradx 最大
			int bestYYID = -1; // grady 最大
			int bestXYID = -1; // gradx-grady 最大
			int bestYXID = -1; // gradx+grady 最大 

			float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

			Eigen::Vector3f* grads0 = grads+x+y*w; // 当前网格的起点
			// 分别找到该网格内上面4个best
			for(int dx=0;dx<pot;dx++)
				for(int dy=0;dy<pot;dy++)
				{
					int idx = dx+dy*w;
					Eigen::Vector3f g=grads0[idx]; // 遍历网格内的每一个像素
					float sqgd = g.tail<2>().squaredNorm(); // 梯度平方和
					float TH = THFac*minUseGrad_pixsel * (0.75f);  //阈值, 为什么都乘0.75 ? downweight		//min: setting : 10

					if(sqgd > TH*TH)
					{
						float agx = fabs((float)g[1]);
						if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

						float agy = fabs((float)g[2]);
						if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

						float gxpy = fabs((float)(g[1]-g[2]));
						if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

						float gxmy = fabs((float)(g[1]+g[2]));
						if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
					}
				}

			bool* map0 = map_out+x+y*w; // 选出来的像素为TRUE

			// 选上这些最大的像素
			if(bestXXID>=0)
			{
				if(!map0[bestXXID]) // 没有被选
					numGood++;
				map0[bestXXID] = true;

			}
			if(bestYYID>=0)
			{
				if(!map0[bestYYID])
					numGood++;
				map0[bestYYID] = true;

			}
			if(bestXYID>=0)
			{
				if(!map0[bestXYID])
					numGood++;
				map0[bestXYID] = true;

			}
			if(bestYXID>=0)
			{
				if(!map0[bestYXID])
					numGood++;
				map0[bestYXID] = true;

			}
		}
	}

	return numGood;
}

//* 同上, 只是把pot作为参数
inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, int pot, float THFac)
{

	memset(map_out, 0, sizeof(bool)*w*h);

	int numGood = 0;
	for(int y=1;y<h-pot;y+=pot)
	{
		for(int x=1;x<w-pot;x+=pot)
		{
			int bestXXID = -1;
			int bestYYID = -1;
			int bestXYID = -1;
			int bestYXID = -1;

			float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

			Eigen::Vector3f* grads0 = grads+x+y*w;
			for(int dx=0;dx<pot;dx++)
				for(int dy=0;dy<pot;dy++)
				{
					int idx = dx+dy*w;
					Eigen::Vector3f g=grads0[idx];
					float sqgd = g.tail<2>().squaredNorm();
					float TH = THFac*minUseGrad_pixsel * (0.75f);

					if(sqgd > TH*TH)
					{
						float agx = fabs((float)g[1]);
						if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

						float agy = fabs((float)g[2]);
						if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

						float gxpy = fabs((float)(g[1]-g[2]));
						if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

						float gxmy = fabs((float)(g[1]+g[2]));
						if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
					}
				}

			bool* map0 = map_out+x+y*w;

			if(bestXXID>=0)
			{
				if(!map0[bestXXID])
					numGood++;
				map0[bestXXID] = true;

			}
			if(bestYYID>=0)
			{
				if(!map0[bestYYID])
					numGood++;
				map0[bestYYID] = true;

			}
			if(bestXYID>=0)
			{
				if(!map0[bestXYID])
					numGood++;
				map0[bestXYID] = true;

			}
			if(bestYXID>=0)
			{
				if(!map0[bestYXID])
					numGood++;
				map0[bestYXID] = true;

			}
		}
	}

	return numGood;
}


inline int makePixelStatus(Eigen::Vector3f* grads, bool* map, int w, int h, float desiredDensity, int recsLeft=5, float THFac = 1)
{
	if(sparsityFactor < 1) sparsityFactor = 1; // 网格的大小, 在网格内选择最大的

	int numGoodPoints;


	if(sparsityFactor==1) numGoodPoints = gridMaxSelection<1>(grads, map, w, h, THFac);
	else if(sparsityFactor==2) numGoodPoints = gridMaxSelection<2>(grads, map, w, h, THFac);
	else if(sparsityFactor==3) numGoodPoints = gridMaxSelection<3>(grads, map, w, h, THFac);
	else if(sparsityFactor==4) numGoodPoints = gridMaxSelection<4>(grads, map, w, h, THFac);
	else if(sparsityFactor==5) numGoodPoints = gridMaxSelection<5>(grads, map, w, h, THFac);
	else if(sparsityFactor==6) numGoodPoints = gridMaxSelection<6>(grads, map, w, h, THFac);
	else if(sparsityFactor==7) numGoodPoints = gridMaxSelection<7>(grads, map, w, h, THFac);
	else if(sparsityFactor==8) numGoodPoints = gridMaxSelection<8>(grads, map, w, h, THFac);
	else if(sparsityFactor==9) numGoodPoints = gridMaxSelection<9>(grads, map, w, h, THFac);
	else if(sparsityFactor==10) numGoodPoints = gridMaxSelection<10>(grads, map, w, h, THFac);
	else if(sparsityFactor==11) numGoodPoints = gridMaxSelection<11>(grads, map, w, h, THFac);
	else numGoodPoints = gridMaxSelection(grads, map, w, h, sparsityFactor, THFac);


	/*
	 * #points is approximately proportional to sparsityFactor^2.
	 */

	float quotia = numGoodPoints / (float)(desiredDensity);

	int newSparsity = (sparsityFactor * sqrtf(quotia))+0.7f; // 更新网格大小


	if(newSparsity < 1) newSparsity=1;


	float oldTHFac = THFac;
	if(newSparsity==1 && sparsityFactor==1) THFac = 0.5;  // 已经是最小的了, 但是数目还是不够, 就减小阈值

	// 如果满足网格大小变化小且阈值是0.5 || 点数量在20%误差内 || 递归次数已到 , 则返回
	if((abs(newSparsity-sparsityFactor) < 1 && THFac==oldTHFac) ||
			( quotia > 0.8 &&  1.0f / quotia > 0.8) ||
			recsLeft == 0) 
	{

//		printf(" \n");
		//all good
		sparsityFactor = newSparsity;
		return numGoodPoints;
	}
	else // 否则进行递归
	{
//		printf(" -> re-evaluate! \n");
		// re-evaluate.
		sparsityFactor = newSparsity;
		return makePixelStatus(grads, map, w,h, desiredDensity, recsLeft-1, THFac);
	}
}

// 2020.07.13 yzk
template<int pot>
inline int gridMaxSelectionFromLidar(Eigen::Vector3f* grads, bool* map_out, int w, int h, float THFac, 
	std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &vCloudPixel)
{
	std::vector<std::vector<Vec2f,Eigen::aligned_allocator<Vec2f>>> vLvl;
	std::vector<std::vector<int>> vIndex;

	for(int i = 0; i < (h-1)/pot; i++)
		for(int j = 0; j < (w-1)/pot; j++)
		{
			std::vector<Vec2f,Eigen::aligned_allocator<Vec2f>> tempPt;
			vLvl.push_back(tempPt);
			std::vector<int> tempIndex;
			vIndex.push_back(tempIndex);
		}

	for(int i = 0; i < vCloudPixel.size(); i++)
	{
		if((int)vCloudPixel[i](0, 0) >= w-pot || (int)vCloudPixel[i](1, 0) >= h-pot) continue;

	    int indexX = (int)(vCloudPixel[i](0, 0)-1) / pot;
	    int indexY = (int)(vCloudPixel[i](1, 0)-1) / pot;

	    vLvl[indexY * ((w-1)/pot) + indexX].push_back(Vec2f((float)vCloudPixel[i](0, 0), (float)vCloudPixel[i](1, 0)));
	    vIndex[indexY * ((w-1)/pot) + indexX].push_back(i);
	}

	memset(map_out, 0, sizeof(bool)*vCloudPixel.size());

	int numGood = 0;
	for(int y=1;y<h-pot;y+=pot)  // 每隔一个pot遍历
	{
		for(int x=1;x<w-pot;x+=pot)
		{
			int bestXXID = -1; // gradx 最大
			int bestYYID = -1; // grady 最大
			int bestXYID = -1; // gradx-grady 最大
			int bestYXID = -1; // gradx+grady 最大 

			float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

			Eigen::Vector3f* grads0 = grads+x+y*w; // 当前网格的起点
			// 分别找到该网格内上面4个best
			int i = ((y-1)/pot)*((w-1)/pot) + ((x-1)/pot);
			for(int j = 0; j < vLvl[i].size(); j++)
			{
				int idx = vLvl[i][j](0, 0) + vLvl[i][j](1, 0) * w;
				Eigen::Vector3f g = grads[idx];
				float sqgd = g.tail<2>().squaredNorm();
				float TH = THFac*minUseGrad_pixsel * (0.75f);

				if(sqgd > TH*TH)
				{
					float agx = fabs((float)g[1]);
					if(agx > bestXX) {bestXX=agx; bestXXID=vIndex[i][j];}

					float agy = fabs((float)g[2]);
					if(agy > bestYY) {bestYY=agy; bestYYID=vIndex[i][j];}

					float gxpy = fabs((float)(g[1]-g[2]));
					if(gxpy > bestXY) {bestXY=gxpy; bestXYID=vIndex[i][j];}

					float gxmy = fabs((float)(g[1]+g[2]));
					if(gxmy > bestYX) {bestYX=gxmy; bestYXID=vIndex[i][j];}
				}
			}

			// 选上这些最大的像素
			if(bestXXID>=0)
			{
				if(!map_out[bestXXID]) // 没有被选
					numGood++;
				map_out[bestXXID] = true;

			}
			if(bestYYID>=0)
			{
				if(!map_out[bestYYID])
					numGood++;
				map_out[bestYYID] = true;

			}
			if(bestXYID>=0)
			{
				if(!map_out[bestXYID])
					numGood++;
				map_out[bestXYID] = true;

			}
			if(bestYXID>=0)
			{
				if(!map_out[bestYXID])
					numGood++;
				map_out[bestYXID] = true;

			}
		}
	}

	return numGood;
}

inline int gridMaxSelectionFromLidar(Eigen::Vector3f* grads, bool* map_out, int w, int h, int pot, float THFac, 
	std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &vCloudPixel)
{

	std::vector<std::vector<Vec2f,Eigen::aligned_allocator<Vec2f>>> vLvl;
	std::vector<std::vector<int>> vIndex;

	for(int i = 0; i < (h-1)/pot; i++)
		for(int j = 0; j < (w-1)/pot; j++)
		{
			std::vector<Vec2f,Eigen::aligned_allocator<Vec2f>> tempPt;
			vLvl.push_back(tempPt);
			std::vector<int> tempIndex;
			vIndex.push_back(tempIndex);
		}

	for(int i = 0; i < vCloudPixel.size(); i++)
	{
		if(vCloudPixel[i](0, 0) >= w-pot || vCloudPixel[i](1, 0) >= h-pot) continue;

	    int indexX = (int)(vCloudPixel[i](0, 0)-1) / pot;
	    int indexY = (int)(vCloudPixel[i](1, 0)-1) / pot;

	    vLvl[indexY * ((w-1)/pot) + indexX].push_back(Vec2f((float)vCloudPixel[i](0, 0), (float)vCloudPixel[i](1, 0)));
	    vIndex[indexY * ((w-1)/pot) + indexX].push_back(i);
	}

	memset(map_out, 0, sizeof(bool)*vCloudPixel.size());

	int numGood = 0;
	for(int y=1;y<h-pot;y+=pot)  // 每隔一个pot遍历
	{
		for(int x=1;x<w-pot;x+=pot)
		{
			int bestXXID = -1; // gradx 最大
			int bestYYID = -1; // grady 最大
			int bestXYID = -1; // gradx-grady 最大
			int bestYXID = -1; // gradx+grady 最大 

			float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

			Eigen::Vector3f* grads0 = grads+x+y*w; // 当前网格的起点
			// 分别找到该网格内上面4个best
			int i = ((y-1)/pot)*((w-1)/pot) + ((x-1)/pot);
			for(int j = 0; j < vLvl[i].size(); j++)
			{
				int idx = vLvl[i][j](0, 0) + vLvl[i][j](1, 0) * w;
				Eigen::Vector3f g = grads[idx];
				float sqgd = g.tail<2>().squaredNorm();
				float TH = THFac*minUseGrad_pixsel * (0.75f);

				if(sqgd > TH*TH)
				{
					float agx = fabs((float)g[1]);
					if(agx > bestXX) {bestXX=agx; bestXXID=vIndex[i][j];}

					float agy = fabs((float)g[2]);
					if(agy > bestYY) {bestYY=agy; bestYYID=vIndex[i][j];}

					float gxpy = fabs((float)(g[1]-g[2]));
					if(gxpy > bestXY) {bestXY=gxpy; bestXYID=vIndex[i][j];}

					float gxmy = fabs((float)(g[1]+g[2]));
					if(gxmy > bestYX) {bestYX=gxmy; bestYXID=vIndex[i][j];}
				}
			}

			// 选上这些最大的像素
			if(bestXXID>=0)
			{
				if(!map_out[bestXXID]) // 没有被选
					numGood++;
				map_out[bestXXID] = true;

			}
			if(bestYYID>=0)
			{
				if(!map_out[bestYYID])
					numGood++;
				map_out[bestYYID] = true;

			}
			if(bestXYID>=0)
			{
				if(!map_out[bestXYID])
					numGood++;
				map_out[bestXYID] = true;

			}
			if(bestYXID>=0)
			{
				if(!map_out[bestYXID])
					numGood++;
				map_out[bestYXID] = true;

			}
		}
	}

	return numGood;
}

inline int makePixelStatusFromLidar(Eigen::Vector3f* grads, bool* map, int w, int h, float desiredDensity, int recsLeft, float THFac, 
	std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> &vCloudPixel)
{
	if(sparsityFactor < 1) sparsityFactor = 1; // 网格的大小, 在网格内选择最大的

	int numGoodPoints;

	if(sparsityFactor==1) numGoodPoints = gridMaxSelectionFromLidar<1>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==2) numGoodPoints = gridMaxSelectionFromLidar<2>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==3) numGoodPoints = gridMaxSelectionFromLidar<3>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==4) numGoodPoints = gridMaxSelectionFromLidar<4>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==5) numGoodPoints = gridMaxSelectionFromLidar<5>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==6) numGoodPoints = gridMaxSelectionFromLidar<6>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==7) numGoodPoints = gridMaxSelectionFromLidar<7>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==8) numGoodPoints = gridMaxSelectionFromLidar<8>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==9) numGoodPoints = gridMaxSelectionFromLidar<9>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==10) numGoodPoints = gridMaxSelectionFromLidar<10>(grads, map, w, h, THFac, vCloudPixel);
	else if(sparsityFactor==11) numGoodPoints = gridMaxSelectionFromLidar<11>(grads, map, w, h, THFac, vCloudPixel);
	else numGoodPoints = gridMaxSelectionFromLidar(grads, map, w, h, sparsityFactor, THFac, vCloudPixel);

	/*
	 * #points is approximately proportional to sparsityFactor^2.
	 */

	float quotia = numGoodPoints / (float)(desiredDensity);

	int newSparsity = (sparsityFactor * sqrtf(quotia))+0.7f; // 更新网格大小


	if(newSparsity < 1) newSparsity=1;


	float oldTHFac = THFac;
	if(newSparsity==1 && sparsityFactor==1) THFac = 0.5;  // 已经是最小的了, 但是数目还是不够, 就减小阈值

	// 如果满足网格大小变化小且阈值是0.5 || 点数量在20%误差内 || 递归次数已到 , 则返回
	if((abs(newSparsity-sparsityFactor) < 1 && THFac==oldTHFac) ||
			( quotia > 0.8 &&  1.0f / quotia > 0.8) ||
			recsLeft == 0) 
	{

//		printf(" \n");
		//all good
		sparsityFactor = newSparsity;
		return numGoodPoints;
	}
	else // 否则进行递归
	{
//		printf(" -> re-evaluate! \n");
		// re-evaluate.
		sparsityFactor = newSparsity;
		return makePixelStatusFromLidar(grads, map, w,h, desiredDensity, recsLeft-1, THFac, vCloudPixel);
	}
}
// 2020.07.13 yzk

}

