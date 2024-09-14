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


#include "FullSystem/PixelSelector2.h"
 
// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
	randomPattern = new unsigned char[w*h];
	std::srand(3141592);	// want to be deterministic.
	for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF; // 随机数, 取低8位

	currentPotential=3;
	currentTH=700;

	// 32*32个块进行计算阈值
	gradHist = new int[100*(1+w/32)*(1+h/32)];
	ths = new float[((w/32)*(h/32)+100)];
	thsSmoothed = new float[((w/32)*(h/32)+100)];
	allowFast=false;
	gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
	delete[] randomPattern;
	delete[] gradHist;
	delete[] ths;
	delete[] thsSmoothed;
}

//* 占据 below% 的梯度值作为阈值
int computeHistQuantil(int* hist, float below ,int num = 90)
{
	int th = hist[0]*below+0.5f; // 最低的像素个数
	for(int i=0;i<num;i++) // 90? 这么随便....
	{
		th -= hist[i+1];  // 梯度值为0-i的所有像素个数占 below %
		if(th<0) return i;
	}
	return num;
}

void PixelSelector::makeHists(const frame_hessian* const fh)
{
	float * mapmax0 = fh->absSquaredGrad[0]; //第0层梯度
	gradHistFrame = fh;
	
	int w = wG[0];
	int h = hG[0];

	int w32 = w/32;
	int h32 = h/32;
	thsStep = w32;

	// XTL：将图像划分成32×32的小格，每个格子计算一个梯度的阈值，保存在ths当中
	// XTL：阈值为梯度中位数
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float* map0 = mapmax0+32*x+32*y*w;
			int* hist0 = gradHist;

			memset(hist0,0,sizeof(int)*50); // 分成50格

			for(int j=0;j<32;j++) for(int i=0;i<32;i++)
			{
				int it = i+32*x;
				int jt = j+32*y;
				if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;

				// 设置阈值时去掉mask之外的点
				if(!maskG[fh->_cam_idx][0][it+jt*w])	continue;

				int g = sqrtf(map0[i+j*w]); // 梯度平方和开根号
				if(g>48) g=48; 
				hist0[g+1]++; // 1-49 存相应梯度个数
				hist0[0]++;  // 所有的像素个数
			}
			// 得到每一block的阈值
			ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;//setting: 0.5/7 // 2021.11.30
		}
	// XTL：周围四个格子的阈值与该格子的阈值求平均值，放入thsSmoothed
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float sum=0,num=0;
			if(x>0)
			{
				if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
				num++; sum+=ths[x-1+(y)*w32];
			}

			if(x<w32-1)
			{
				if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
				num++; sum+=ths[x+1+(y)*w32];
			}

			if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
			num++; sum+=ths[x+y*w32];

			thsSmoothed[x+y*w32] = (sum/num) * (sum/num);

		}

}

/********************************
 * @ function:
 * 
 * @ param: 	fh				帧Hessian数据结构
 * @			map_out			选出的地图点
 * @			density		 	每一金字塔层要的点数(密度)
 * @			recursionsLeft	最大递归次数
 * @			plot			画图
 * @			thFactor		阈值因子
 * @
 * @ note:		使用递归
 *******************************/
int PixelSelector::makeMaps(
		const frame_hessian* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
	float numHave=0;
	float numWant=density;
	float quotia;
	int idealPotential = currentPotential;

//	if(setting_pixelSelectionUseFast>0 && allowFast)
//	{
//		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
//		std::vector<cv::KeyPoint> pts;
//		cv::Mat img8u(hG[0],wG[0],CV_8U);
//		for(int i=0;i<wG[0]*hG[0];i++)
//		{
//			float v = fh->dI[i][0]*0.8;
//			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
//		}
//		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
//		for(unsigned int i=0;i<pts.size();i++)
//		{
//			int x = pts[i].pt.x+0.5;
//			int y = pts[i].pt.y+0.5;
//			map_out[x+y*wG[0]]=1;
//			numHave++;
//		}
//
//		printf("FAST selection: got %f / %f!\n", numHave, numWant);
//		quotia = numWant / numHave;
//	}
//	else
	{

		// the number of selected pixels behaves approximately as
		// K / (pot+1)^2, where K is a scene-dependent constant.
		// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.
//[ ***step 1*** ] 没有计算直方图, 以及选点的阈值, 则调用函数生成block阈值
		if(fh != gradHistFrame) makeHists(fh); // 第一次进来，则生成直方图
		// select!
//[ ***step 2*** ] 在当前帧上选择符合条件的像素
		Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor);

		// sub-select!
		numHave = n[0]+n[1]+n[2]; // 选择得到的点
		quotia = numWant / numHave;  // 得到的 与 想要的 比例

//[ ***step 3*** ] 计算新的采像素点的, 范围大小, 相当于动态网格了, pot越小取得点越多
		// by default we want to over-sample by 40% just to be sure.
		float K = numHave * (currentPotential+1) * (currentPotential+1); // 相当于覆盖的面积, 每一个像素对应一个pot*pot
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

//[ ***step 4*** ] 想要的数目和已经得到的数目, 大于或小于0.25都会重新采样一次
		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential) // idealPotential应该小
				idealPotential = currentPotential-1; // 减小,多采点

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;

			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor); //递归
		}
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// re-sample to get less points!

			if(idealPotential<=currentPotential) // idealPotential应该大
				idealPotential = currentPotential+1; // 增大, 少采点

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

		}
	}

//[ ***step 5*** ] 现在提取的还是多, 随机删除一些点
	int numHaveSub = numHave;
	if(quotia < 0.95)
	{
		int wh=wG[0]*hG[0];
		int rn=0;
		unsigned char charTH = 255*quotia;
		for(int i=0;i<wh;i++)
		{
			if(map_out[i] != 0)
			{
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

	//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
	//			100*numHave/(float)(wG[0]*hG[0]),
	//			100*numWant/(float)(wG[0]*hG[0]),
	//			currentPotential,
	//			idealPotential,
	//			100*numHaveSub/(float)(wG[0]*hG[0]));
	currentPotential = idealPotential; //???

	// 画出选择结果
	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7; // 像素值
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}

		IOWrap::displayImage("Selector Image", &img);

		// 安照不同层数的像素, 画上不同颜色
		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)
					img.setPixelCirc(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)
					img.setPixelCirc(x,y,Vec3b(255,0,0));
				else if(map_out[i] == 4)
					img.setPixelCirc(x,y,Vec3b(0,0,255));
			}
		IOWrap::displayImage("Selector Pixels", &img);
	}

	return numHaveSub;
}


Eigen::Vector3i PixelSelector::select(const frame_hessian* const fh,
		float* map_out, int pot, float thFactor)
{
	Eigen::Vector3f const * const map0 = fh->dI;

	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];

	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

	//! 随机选这16个对应方向上的梯度和阈值比较
	//! 每个pot里面的方向随机选取的, 防止特征相同, 重复

	// 模都是1		sin(1/16) = 0.1951
	const Vec2f directions[16] = {
	         Vec2f(0,    1.0000),
	         Vec2f(0.3827,    0.9239),
	         Vec2f(0.1951,    0.9808),
	         Vec2f(0.9239,    0.3827),
	         Vec2f(0.7071,    0.7071),
	         Vec2f(0.3827,   -0.9239),
	         Vec2f(0.8315,    0.5556),
	         Vec2f(0.8315,   -0.5556),
	         Vec2f(0.5556,   -0.8315),
	         Vec2f(0.9808,    0.1951),
	         Vec2f(0.9239,   -0.3827),
	         Vec2f(0.7071,   -0.7071),
	         Vec2f(0.5556,    0.8315),
	         Vec2f(0.9808,   -0.1951),
	         Vec2f(1.0000,    0.0000),
	         Vec2f(0.1951,   -0.9808)};

	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));  // 不同选择状态的数目不同	//枚举类型就占一个整形4字节

	float dw1 = setting_gradDownweightPerLevel; // 第二层		setting: 0.75
	float dw2 = dw1*dw1; // 第三层


	int n3=0, n2=0, n4=0;
	// XTL：第一层步长为4×pot，第二层为2×pot，第三层为pot，第四层为1
	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
	{	
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;

		Vec2f dir4 = directions[randomPattern[n2] & 0xF];
		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			int x34 = x3+x4;
			int y34 = y3+y4;
			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];
			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
			{
				int x234 = x2+x34;
				int y234 = y2+y34;
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];
				// XTL：在pot×pot一个格子内部，选梯度最大的点（必须超过阈值），其下标放入bestIdx2，梯度放入bestVal2
				// XTL：若找不到梯度足够的点，则往上一层找，梯度要求会有所降低，找到的话序号放入bestIdx3,梯度放入bestVal3
				for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
				{
					assert(x1+x234 < w);
					assert(y1+y234 < h);
					int idx = x1+x234 + w*(y1+y234);
					int xf = x1+x234;
					int yf = y1+y234;

					if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;

					if(!(maskG[fh->_cam_idx][0][xf + yf*w])) continue;

					float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];		//xf/32 以及 yf/32是对应的x, y方向的格子id
					float pixelTH1 = pixelTH0*dw1;
					float pixelTH2 = pixelTH1*dw2;

					
					float ag0 = mapmax0[idx]; // 第0层梯度模
					// XTL：若第0层的梯度大于其所在区域的阈值，求一个随机方向上的图像梯度，若梯度足够大
					if(ag0 > pixelTH0*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();  // 后两位是图像导数 dx dy
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));   // 以这个方向上的梯度来判断
						if(!setting_selectDirectionDistribution) dirNorm = ag0;

						if(dirNorm > bestVal2) // 取梯度最大的
						{ bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
					}
					
					if(bestIdx3==-2) continue; // 有了则不在其它层选点, 但是还会在该pot里选最大的

					float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1]; // 第1层
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;

					float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2]; // 第2层

					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = idx; }
					}
				}

				// 第0层的pot循环完, 若有则添加标志
				if(bestIdx2>0)
				{
					map_out[bestIdx2] = 1;
					// 高层pot中有更好的了，满足更严格要求的，就不用满足pixelTH1的了
                    // bug bestVal3没有什么用，因为bestIdx3=-2直接continue了
					bestVal3 = 1e10;  // 第0层找到了, 就不在高层找了
					n2++; // 计数
				}
			}
			// 第0层没有, 则在第1层选
			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}
		// 第1层没有, 则在第2层选
		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}

	return Eigen::Vector3i(n2,n3,n4); // 第0, 1, 2层选点的个数
}


int PixelSelector::makeMapsFromLidar(const frame_hessian* const fh, float* map_out, float density, int recursionsLeft,
		bool plot, float thFactor, std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> &vCloudPixel, int id)
{
	float numHave=0;
	float numWant=density;
	float quotia;

	int idealPotential = currentPotential;
	{

		if(fh != gradHistFrame) makeHists(fh); // 第一次进来，则生成直方图
		Eigen::Vector3i n = this->selectFromLidar(fh, map_out, currentPotential, thFactor, vCloudPixel);

		// sub-select!
		numHave = n[0]+n[1]+n[2];

		quotia = numWant / numHave;  

		float K = numHave * (currentPotential+1) * (currentPotential+1); // 相当于覆盖的面积, 每一个像素对应一个pot*pot
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

//[ ***step 4*** ] 想要的数目和已经得到的数目, 大于或小于0.25都会重新采样一次
		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			// printf("voxel %d too big!!\n",currentPotential);
			if(idealPotential>=currentPotential)
				idealPotential = currentPotential-1;

			currentPotential = idealPotential;

			return makeMapsFromLidar(fh,map_out, density, recursionsLeft-1, plot,thFactor, vCloudPixel, id); //递归
		}
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// printf("voxel too small!!\n");
			// re-sample to get less points!

			if(idealPotential<=currentPotential)
				idealPotential = currentPotential+1;

			currentPotential = idealPotential;
			return makeMapsFromLidar(fh,map_out, density, recursionsLeft-1, plot,thFactor, vCloudPixel, id);

		}
	}

//[ ***step 5*** ] 现在提取的还是多, 随机删除一些点
	int numHaveSub = numHave;

	if(quotia < 0.95)
	{
		// printf("voxel %d small!!\n",currentPotential);
		int rn=0;
		unsigned char charTH = 255*quotia;
		for(int i=0;i<vCloudPixel.size();i++)
		{
			if(map_out[i] != 0)
			{
				// rn = (int)(vCloudPixel[i](0, 0) + vCloudPixel[i](1, 0) * wG[0]);
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

	currentPotential = idealPotential;

	// 画出选择结果
	if(plot)
	{
		int w = wG[0];
		int h = hG[0];

		MinimalImageB3 img(w,h);

		for(int i = 0; i < w; i++)
		{
			for(int j = 0; j < h; j++)
			{
				int idx = i + w*j;
				float c = fh->dI[idx][0]*0.7;
				if(c>255) c=255;
				img.at(idx) = Vec3b(c,c,c);
			}
		}

		for(int i = 0; i < vCloudPixel.size(); i++)
		{
			if(map_out[i] == 1)
				img.setPixelCirc(vCloudPixel[i](0, 0),vCloudPixel[i](1, 0),Vec3b(0,255,0));
			else if(map_out[i] == 2)
				img.setPixelCirc(vCloudPixel[i](0, 0),vCloudPixel[i](1, 0),Vec3b(255,0,0));
			else if(map_out[i] == 4)
				img.setPixelCirc(vCloudPixel[i](0, 0),vCloudPixel[i](1, 0),Vec3b(0,0,255));
		}
		char str[100];
		sprintf(str, "Selector Pixels %d",id);
		IOWrap::displayImage(str, &img);
		IOWrap::waitKey(0);
		IOWrap::closeAllWindows();
	}

	return numHaveSub;
}


Eigen::Vector3i PixelSelector::selectFromLidar(const frame_hessian* const fh, float* map_out, int pot, float thFactor,
	std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> &vCloudPixel, int _idx)
{
	float th = 50;
	std::vector<std::vector<int>> vIndex0;
	std::vector<std::vector<Vec2f,Eigen::aligned_allocator<Vec2f>>> vLvl0;
	// std::vector<int*> IntensityCnt;

	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

	// 模都是1		sin(1/16) = 0.1951
	const Vec2f directions[16] = {
			Vec2f(0,    1.0000),
			Vec2f(0.3827,    0.9239),
			Vec2f(0.1951,    0.9808),
			Vec2f(0.9239,    0.3827),
			Vec2f(0.7071,    0.7071),
			Vec2f(0.3827,   -0.9239),
			Vec2f(0.8315,    0.5556),
			Vec2f(0.8315,   -0.5556),
			Vec2f(0.5556,   -0.8315),
			Vec2f(0.9808,    0.1951),
			Vec2f(0.9239,   -0.3827),
			Vec2f(0.7071,   -0.7071),
			Vec2f(0.5556,    0.8315),
			Vec2f(0.9808,   -0.1951),
			Vec2f(1.0000,    0.0000),
			Vec2f(0.1951,   -0.9808)};

	int numPotH = (h%pot==0) ? h/pot : h/pot+1;
	int numPotW = (w%pot==0) ? w/pot : w/pot+1;

	float ths_[numPotH*numPotW];
	for(int i = 0; i < numPotH; i++)
		for(int j = 0; j < numPotW; j++)
		{
			std::vector<Vec2f,Eigen::aligned_allocator<Vec2f>> tempPt;
			vLvl0.push_back(tempPt);
			std::vector<int> tempIndex;
			vIndex0.push_back(tempIndex);
			// int* temp;
			// temp = new int[257];
			// IntensityCnt.push_back(temp);
		}


	int n3=0, n2=0, n4=0;
	//const 在*左, 指针内容不可改, 在*右指针不可改
	Eigen::Vector3f const * const map0 = fh->dI;

	// 0, 1, 2层的梯度平方和
	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];

	memset(map_out, 0, (vCloudPixel.size()) * sizeof(float));  // 不同选择状态的数目不同
	// XTL vLvl0的下标代表点所在block位置，里面的元素是具体的点坐标 vIndex0是该点在vCloudPixel中的索引
	bool haveChose[(int)w/pot*(int)h/pot]={0};
	for(int i = 0; i < vCloudPixel.size(); i++)
	{
		int X = (int)vCloudPixel[i](0, 0) / pot;
		int Y = (int)vCloudPixel[i](1, 0) / pot;

		vLvl0[Y * numPotW + X].push_back(Vec2f((float)vCloudPixel[i](0, 0), (float)vCloudPixel[i](1, 0)));
		vIndex0[Y * numPotW + X].push_back(i);
	}
	// for(int i = 0; i < numPotH; i++)
	// 	for(int j = 0; j < numPotW; j++)
	// 		ths_[i * numPotW + j] = computeHistQuantil(IntensityCnt[i*numPotW+j],setting_minGradHistCut,256) + setting_minGradHistAdd;
	
	// printf("Select %d points\n",n2);

	// 金字塔层阈值的减小倍数
	float dw1 = setting_gradDownweightPerLevel; // 第二层		setting: 0.75
	float dw2 = dw1*dw1; // 第三层

	// 第0层的4个pot里面只要选一个像素, 就不在对应高层的pot里面选了,
	// 但是还会在第0层的每个pot里面选大于阈值的像素
	// 阈值随着层数增加而下降
	// int n3=0, n2=0, n4=0;

	// XTL：第一层步长为4×pot，第二层为2×pot，第三层为pot，第四层为1
	int cnt1=0; int cnt2=0; int cnt3=0;
	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot)) // TODO
	{	
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;
		Vec2f dir4 = directions[randomPattern[n2] & 0xF];//randomPattern 是unsigned char 的随机种子取到的值
		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			int x34 = x3+x4;
			int y34 = y3+y4;
			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];  
			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
			{
				int x234 = x2+x34;
				int y234 = y2+y34;
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				int vBestIdx2=-1; float vBestVal2=0;
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];
				int i = (y4/pot + y3/pot + y2/pot)*numPotW + (x4/pot + x3/pot + x2/pot);
				// XTL：在pot×pot一个格子内部，选梯度最大的点（必须超过阈值），其下标放入bestIdx2，梯度放入bestVal2
				// XTL：若找不到梯度足够的点，则往上一层找，梯度要求会有所降低，找到的话序号放入bestIdx3,梯度放入bestVal3
				// if(addfeature)
				// {
				//	int _idx = fh->_cam_idx;
				// 	for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
				// 	{
				// 		assert(x1+x234 < w);
				// 		assert(y1+y234 < h);
				// 		int idx = x1+x234 + w*(y1+y234);
				// 		int xf = x1+x234;
				// 		int yf = y1+y234;

				// 		if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;

				// 		if(!(maskG[_idx][0][xf + yf*w])) continue;

				// 		float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];		//xf/32 以及 yf/32是对应的x, y方向的格子id
				// 		float pixelTH1 = pixelTH0*dw1;
				// 		float pixelTH2 = pixelTH1*dw2;
				
				// 		float ag0 = mapmax0[idx]; // 第0层梯度模
				// 		// XTL：若第0层的梯度大于其所在区域的阈值，求一个随机方向上的图像梯度，若梯度足够大
				// 		if(ag0 > pixelTH0*thFactor)
				// 		{
				// 			Vec2f ag0d = map0[idx].tail<2>();  // 后两位是图像导数 dx dy
				// 			float dirNorm = fabsf((float)(ag0d.dot(dir2)));   // 以这个方向上的梯度来判断
				// 			if(!setting_selectDirectionDistribution) dirNorm = ag0;

				// 			if(dirNorm > vBestVal2) // 取梯度最大的
				// 			{ vBestVal2 = dirNorm; vBestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
				// 		}
				// 	}
				// 	int Ku = idx % w;
				// 	int Kv = idx / w;
				// 	// XTL find nearest 3 points
				// 	float bestdist1 = 100; int bestidx1 = -1; 
				// 	float bestdist2 = 100; int bestidx2 = -1; 
				// 	float bestdist3 = 100; int bestidx3 = -1;
				// 	if(vBestIdx2 != -2)
				// 	{
				// 		for(int k = 0; k < vIndex0[i].size(); k++)
				// 		{
				// 			Eigen::Vector3d tripod = vCloudPixel[vIndex0[i][k]];
				// 			/*
				// 			float Kv_ = tripod(1,0); float Ku_ = tripod(0,0);
				// 			for(int i=0;i<lvl;i++){ Kv_ /= 2; Ku_ /= 2; }
				// 			Vec3f hitColor = getInterpolatedElement33(dINewl, Ku_, Kv_, wl);
				// 			float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
				// 			if(fabs(residual)>9) continue; // TODO adjust
				// 			*/
				// 			float dist = sqrt(pow(tripod(0,0)-Ku,2)+pow(tripod(1,0)-Kv,2));

				// 			if(dist<bestdist1) {bestdist1 = dist; bestidx1 = k;}
				// 			else if(dist<bestdist2) {bestdist2 = dist; bestidx2 = k;}
				// 			else if(dist<bestdist3) 
				// 			{
				// 				if(vcloudRowIdn[idx][vIndex0[i][bestidx1]] == vcloudRowIdn[idx][vIndex0[i][bestidx2]] &&
				// 					vcloudRowIdn[idx][vIndex0[i][bestidx1]] == vcloudRowIdn[idx][vIndex0[i][k]])
				// 					continue;
				// 				bestdist3 = dist; bestidx3 = k;
				// 			}
				// 		}
				// 		assert(bestdist1!=-1);
				// 		if(bestidx2==-1)
				// 			for(int k = vIndex0[i].size()-1; k >= 0 && k != bestidx1; k--)
				// 			{
				// 				Eigen::Vector3d tripod = vCloudPixel[vIndex0[i][k]];
				// 				float dist = sqrt(pow(tripod(0,0)-Ku,2)+pow(tripod(1,0)-Kv,2));
				// 				if(dist<bestdist2) {bestdist2 = dist; bestidx2 = k;}
				// 				else if(dist<bestdist3) 
				// 				{
				// 					if(vcloudRowIdn[idx][vIndex0[i][bestidx1]] == vcloudRowIdn[idx][vIndex0[i][bestidx2]] &&
				// 						vcloudRowIdn[idx][vIndex0[i][bestidx1]] == vcloudRowIdn[idx][vIndex0[i][k]])
				// 						continue;
				// 					bestdist3 = dist; bestidx3 = k;
				// 					// break;
				// 				}
				// 			}
				// 		if(bestidx2!=-1&&bestidx3==-1)
				// 			for(int k = 0; k < vIndex0[i].size() && k != bestidx1 && k != bestidx2; k++)
				// 			{
				// 				Eigen::Vector3d tripod = vCloudPixel[vIndex0[i][k]];
				// 				float dist = sqrt(pow(tripod(0,0)-Ku,2)+pow(tripod(1,0)-Kv,2));
				// 				if(dist<bestdist3) 
				// 				{
				// 					if(vcloudRowIdn[idx][vIndex0[i][bestidx1]] == vcloudRowIdn[idx][vIndex0[i][bestidx2]] &&
				// 						vcloudRowIdn[idx][vIndex0[i][bestidx1]] == vcloudRowIdn[idx][vIndex0[i][k]])
				// 						continue;
				// 					bestdist3 = dist; bestidx3 = k;
				// 				}
				// 			}

				// 	}
				// }
				// float x1 = (vCloudPixel[vIndex0[i][bestidx1]](0,0)*vCloudPixel[vIndex0[i][bestidx1]](2,0)-cxG[_idx][0])/fxG[_idx][0];
				// float y1 = (vCloudPixel[vIndex0[i][bestidx1]](1,0)*vCloudPixel[vIndex0[i][bestidx1]](2,0)-cyG[_idx][0])/fyG[_idx][0];
				// float z1 = vCloudPixel[vIndex0[i][bestidx1]](2,0);
				// float x2 = (vCloudPixel[vIndex0[i][bestidx2]](0,0)*vCloudPixel[vIndex0[i][bestidx2]](2,0)-cxG[_idx][0])/fxG[_idx][0];
				// float y2 = (vCloudPixel[vIndex0[i][bestidx2]](1,0)*vCloudPixel[vIndex0[i][bestidx2]](2,0)-cyG[_idx][0])/fyG[_idx][0];
				// float z2 = vCloudPixel[vIndex0[i][bestidx2]](2,0);
				// float x3 = (vCloudPixel[vIndex0[i][bestidx3]](0,0)*vCloudPixel[vIndex0[i][bestidx3]](2,0)-cxG[_idx][0])/fxG[_idx][0];
				// float y3 = (vCloudPixel[vIndex0[i][bestidx3]](1,0)*vCloudPixel[vIndex0[i][bestidx3]](2,0)-cyG[_idx][0])/fyG[_idx][0];
				// float z3 = vCloudPixel[vIndex0[i][bestidx3]](2,0);
				// Eigen::Vector3d tripod1 = vCloudPixel[vIndex0[i][bestidx1]];
				// Eigen::Vector3d tripod2 = vCloudPixel[vIndex0[i][bestidx2]];
				// Eigen::Vector3d tripod3 = vCloudPixel[vIndex0[i][bestidx3]];
                // float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                //          - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
                // float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                //          - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
                // float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                //          - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
				// float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

				// // 直线上点的归一化坐标
				// float x = (Ku-cxG[_idx][0])/fxG[_idx][0];
				// float y = (Kv-cyG[_idx][0])/fyG[_idx][0];
				// // 线面交点逆深度
				// float z = -pd/(pa*x+pb*y+pc);
				// printf("Depth value is:%f\n",z);
				// float z = (x1*y2*z3 - x1*y3*z2 - x2*y1*z3 + x2*y3*z1 + x3*y1*z2 - x3*y2*z1) 
				//                  / (x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2 + u*y1*z2 - u*y2*z1
				//                  - v*x1*z2 + v*x2*z1 - u*y1*z3 + u*y3*z1 + v*x1*z3 - v*x3*z1 + u*y2*z3 
				//                  - u*y3*z2 - v*x2*z3 + v*x3*z2);
				for(int j = 0; j < vLvl0[i].size(); j++)
				{
					assert(vLvl0[i][j](0, 0) < w);
					assert(vLvl0[i][j](1, 0) < h);

					int idx = vLvl0[i][j](0, 0) + w * vLvl0[i][j](1, 0); // 像素id
					if(vLvl0[i][j](0, 0)<4 || vLvl0[i][j](0, 0)>=w-5 || vLvl0[i][j](1, 0)<4 || vLvl0[i][j](1, 0)>h-4) continue;

					// if(haveChose[i])
					// 	continue;
					float pixelTH0 = thsSmoothed[((int)vLvl0[i][j](0, 0)>>5) + ((int)vLvl0[i][j](1, 0)>>5) * thsStep];
					float pixelTH1 = pixelTH0*dw1;
					float pixelTH2 = pixelTH1*dw2;

					float ag0 = mapmax0[idx];
					// XTL：若第0层的梯度大于其所在区域的阈值，求一个随机方向上的图像梯度，若梯度足够大
					if(ag0 > pixelTH0*thFactor)
					{
						cnt1++;
						Vec2f ag0d = map0[idx].tail<2>();  // 后两位是图像导数 dx dy
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));   // 以这个方向上的梯度来判断

						if(!setting_selectDirectionDistribution) dirNorm = ag0;

						if(dirNorm > bestVal2) // 取梯度最大的
						{ bestVal2 = dirNorm; bestIdx2 = vIndex0[i][j]; bestIdx3 = -2; bestIdx4 = -2;}
					}
					if(bestIdx3==-2) continue;

					float ag1 = mapmax1[(int)(vLvl0[i][j](0, 0) * 0.5f + 0.25f) + (int)(vLvl0[i][j](1, 0) * 0.5f + 0.25f) * w1]; // 第1层
					if(ag1 > pixelTH1*thFactor)
					{
						cnt2++;
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = vIndex0[i][j]; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;

					float ag2 = mapmax2[(int)(vLvl0[i][j](0, 0) * 0.25f + 0.125) + (int)(vLvl0[i][j](1, 0) * 0.25f + 0.125) * w2]; // 第2层

					if(ag2 > pixelTH2*thFactor)
					{
						cnt3++;
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = vIndex0[i][j]; }
					}
				}
				if(bestIdx2 > 0)
				{
					map_out[bestIdx2] = 1;
					// bug bestVal3没有什么用，因为bestIdx3=-2直接continue了
					bestVal3 = 1e10;  // 第0层找到了, 就不在高层找了
					n2++; // 计数
				}
			}
			// 第0层没有, 则在第1层选
			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}
		// 第1层没有, 则在第2层选
		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}
	// printf("Num of gridient over the TH,lvl1:%d,lvl2:%d,lvl3:%d\n",cnt1,cnt2,cnt3);
    return Eigen::Vector3i(n2,n3,n4); // 第0, 1, 2层选点的个数
}
// 2020.07.02 yzk


Eigen::Matrix<int,5,1> PixelSelector::simpleMakeMapsFromLidar(const FrameHessian* const Fh, float* map_out, float density, int recursionsLeft,
		bool plot, std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloudPixel[], int id)
{
	float numHave=0;
	float numWant=cam_num*density;
	float quotia;

	Eigen::Matrix<int,5,1> n = this->simpleSelectFromLidar(Fh, map_out, numWant, vCloudPixel, id);

	numHave = n[0]+n[1]+n[2]+n[3]+n[4];

	quotia = numWant / numHave;

	if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
	{
		//re-sample to get more points!
		currentTH *= 0.9;
		// if(currentTH<300) currentTH=300;
		return simpleMakeMapsFromLidar(Fh, map_out, numWant, recursionsLeft-1, plot, vCloudPixel, id);
	}
	else if(recursionsLeft>0 && quotia < 0.25)
	{
		// re-sample to get less points!
		currentTH *= 1.1;
		return simpleMakeMapsFromLidar(Fh, map_out, numWant, recursionsLeft-1, plot, vCloudPixel, id);
	}

	if(quotia < 0.95)
	{
		int rn=0;	int cnt=0;
		unsigned char charTH = 255*quotia;
		for(int idx = 0; idx < cam_num; idx++)
		{
			for(int i=0;i<vCloudPixel[idx].size();i++)
			{
				if(map_out[cnt+i] != 0)
				{
					if(randomPattern[rn] > charTH )
					{
						map_out[cnt+i]=0;
						n[idx]--;
					}
					rn++;
				}
			}
			cnt+=vCloudPixel[idx].size();
		}
	}

	return n;
}

Eigen::Matrix<int,5,1> PixelSelector::simpleSelectFromLidar(const FrameHessian* const Fh, float* map_out, float numWant,     
	std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d>> vCloudPixel[], int _idx)
{
    int w = wG[0];    int h = hG[0];
	Eigen::Matrix<int,5,1> n;
	for(int i=0;i<5;i++)
		n[i] = 0; 
	std::vector<float> gredient;
	std::vector<int> index;
	int cnt = 0;
	int cutOff[cam_num];
	memset(cutOff,0,sizeof(int)*cam_num);

	for(int fidx = 0; fidx < cam_num; fidx++)
	{
		Eigen::Vector3f const * const map0 = Fh->frame[fidx]->dI;
		float * mapmax0 = Fh->frame[fidx]->absSquaredGrad[0];
		for(int i = 0; i < vCloudPixel[fidx].size(); i++)    
		{        
			int idx = vCloudPixel[fidx][i](0, 0) + w * vCloudPixel[fidx][i](1, 0);        
			gredient.push_back(mapmax0[idx]);        
			index.push_back(cnt+i);    
		}
		cnt += vCloudPixel[fidx].size();
		cutOff[fidx] = cnt;
	}
	std::sort(index.begin(),index.end(),[&gredient](int i1, int i2) {return gredient[i1] > gredient[i2]; }); 

    memset(map_out, 0, cnt * sizeof(float));

    for(int i:index)    
	{    
		printf("gredient=%f,currentTH=%f\n",gredient[i],currentTH);  
		if(gredient[i] < currentTH /*|| (n[0]+n[1]+n[2]+n[3]+n[4]) >= numWant*/)        
		{            
			printf("MIN gredient=%f,currentTH=%f,",gredient[i],currentTH);  
			printf("n1=%d,n2=%d,n3=%d,n4=%d,n5=%d\n",n[0],n[1],n[2],n[3],n[4]);            
			break;        
		}        
		map_out[i] = 1;      
		if(i<cutOff[0])
		{
			n[0]++;
		}
		else
		{
			for(int idx=0;idx<cam_num;idx++)
			{
				if(i>=cutOff[idx] && i<cutOff[idx+1])
				{
					n[idx+1]++;
					break;
				}
			}
		}
	}    
	return n;
}


}

