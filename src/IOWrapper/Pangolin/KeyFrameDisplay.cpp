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



#include <stdio.h>
#include "util/settings.h"

//#include <GL/glx.h>
//#include <GL/gl.h>
//#include <GL/glu.h>

#include <pangolin/pangolin.h>
#include "KeyFrameDisplay.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"



namespace dso
{
namespace IOWrap
{


KeyFrameDisplay::KeyFrameDisplay()
{
	// originalInputSparse = 0;
	// 2021.12.14
	for(int i=0;i<cam_num;i++){
		originalInputSparse[i] = 0;
		numSparseBufferSize[i]=0;
		numSparsePoints[i]=0;
	}
	// 2021.12.14

	id = 0;
	active= true;
	camToWorld = SE3();

	needRefresh=true;

	my_scaledTH =1e10;
	my_absTH = 1e10;
	my_displayMode = 1;
	my_minRelBS = 0;
	my_sparsifyFactor = 1;

	numGLBufferPoints=0;
	bufferValid = false;
}
void KeyFrameDisplay::setFromF(FrameShell* frame, CalibHessian* HCalib)
{
	id = frame->id;
	// 2021.12.14
	/*
	fx = HCalib->fxl();
	fy = HCalib->fyl();
	cx = HCalib->cxl();
	cy = HCalib->cyl();
	fxi = 1/fx;
	fyi = 1/fy;
	cxi = -cx / fx;
	cyi = -cy / fy;
	*/
	for(int i=0;i<cam_num;i++){
		fx[i] = HCalib->fxl(i);
		fy[i] = HCalib->fyl(i);
		cx[i] = HCalib->cxl(i);
		cy[i] = HCalib->cyl(i);
		fxi[i] = 1/fx[i];
		fyi[i] = 1/fy[i];
		cxi[i] = -cx[i] / fx[i];
		cyi[i] = -cy[i] / fy[i];
	}
	// 2021.12.14
	width = wG[0];
	height = hG[0];
	camToWorld = frame->camToWorld;
	// 2021.12.20
	camToWorld_gt = frame->gtPose;
	// 2021.12.20
	needRefresh=true;
}

void KeyFrameDisplay::setFromKF(FrameHessian* fh, CalibHessian* HCalib)
{
	setFromF(fh->shell, HCalib);

	// add all traces, inlier and outlier points.
	// 2021.12.14
	
	int npoints = 	fh->frame[0]->immaturePoints.size() +
					fh->frame[0]->pointHessians.size() +
					fh->frame[0]->pointHessiansMarginalized.size() +
					fh->frame[0]->pointHessiansOut.size();//2022.1.11

	if(numSparseBufferSize[0] < npoints)
	{
		if(originalInputSparse[0] != 0) delete originalInputSparse[0];
		numSparseBufferSize[0] = npoints+100;
		originalInputSparse[0] = new InputPointSparse<MAX_RES_PER_POINT>[numSparseBufferSize[0]];
	}
	InputPointSparse<MAX_RES_PER_POINT>* pc = originalInputSparse[0];
	numSparsePoints[0]=0;

	for(ImmaturePoint* p : fh->frame[0]->immaturePoints)
	{
		for(int i=0;i<patternNum;i++)
			pc[numSparsePoints[0]].color[i] = p->color[i];

		pc[numSparsePoints[0]].u = p->u;
		pc[numSparsePoints[0]].v = p->v;
		pc[numSparsePoints[0]].idpeth = (p->idepth_max+p->idepth_min)*0.5f;
		pc[numSparsePoints[0]].idepth_hessian = 1000;
		pc[numSparsePoints[0]].relObsBaseline = 0;
		pc[numSparsePoints[0]].numGoodRes = 1;
		pc[numSparsePoints[0]].status = 0;
		numSparsePoints[0]++;
	}

	for(PointHessian* p : fh->frame[0]->pointHessians)
	{
		for(int i=0;i<patternNum;i++)
			pc[numSparsePoints[0]].color[i] = p->color[i];
		pc[numSparsePoints[0]].u = p->u;
		pc[numSparsePoints[0]].v = p->v;
		pc[numSparsePoints[0]].idpeth = p->idepth_scaled;
		pc[numSparsePoints[0]].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints[0]].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints[0]].numGoodRes =  0;
		pc[numSparsePoints[0]].status=1;

		numSparsePoints[0]++;
	}

	for(PointHessian* p : fh->frame[0]->pointHessiansMarginalized)
	{
		for(int i=0;i<patternNum;i++)
			pc[numSparsePoints[0]].color[i] = p->color[i];
		pc[numSparsePoints[0]].u = p->u;
		pc[numSparsePoints[0]].v = p->v;
		pc[numSparsePoints[0]].idpeth = p->idepth_scaled;
		pc[numSparsePoints[0]].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints[0]].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints[0]].numGoodRes =  0;
		pc[numSparsePoints[0]].status=2;
		numSparsePoints[0]++;
	}

	for(PointHessian* p : fh->frame[0]->pointHessiansOut)
	{
		for(int i=0;i<patternNum;i++)
			pc[numSparsePoints[0]].color[i] = p->color[i];
		pc[numSparsePoints[0]].u = p->u;
		pc[numSparsePoints[0]].v = p->v;
		pc[numSparsePoints[0]].idpeth = p->idepth_scaled;
		pc[numSparsePoints[0]].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints[0]].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints[0]].numGoodRes =  0;
		pc[numSparsePoints[0]].status=3;
		numSparsePoints[0]++;
	}
	assert(numSparsePoints[0] <= npoints);

	camToWorld = fh->PRE_camToWorld;
	needRefresh=true;
}


KeyFrameDisplay::~KeyFrameDisplay()
{
	// 2021.12.14
	for(int i=0;i</*cam_num*/1;i++)
		if(originalInputSparse[i] != 0)
			delete[] originalInputSparse[i];
}

bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity)
{
	if(canRefresh)
	{
		needRefresh = needRefresh ||
				my_scaledTH != scaledTH ||
				my_absTH != absTH ||
				my_displayMode != mode ||
				my_minRelBS != minBS ||
				my_sparsifyFactor != sparsity;
	}

	if(!needRefresh) return false;
	needRefresh=false;

	my_scaledTH = scaledTH;
	my_absTH = absTH;
	my_displayMode = mode;
	my_minRelBS = minBS;
	my_sparsifyFactor = sparsity;

	// 2021.12.14
	// for(int idx=0;idx<cam_num;idx++){

	// if there are no vertices, done!
	if(numSparsePoints[0] == 0)
		return false;

	// make data
	Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints[0]*patternNum];
	Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints[0]*patternNum];
	int vertexBufferNumPoints=0;

	for(int i=0;i<numSparsePoints[0];i++)
	{
		/* display modes:
		* my_displayMode==0 - all pts, color-coded
		* my_displayMode==1 - normal points
		* my_displayMode==2 - active only
		* my_displayMode==3 - nothing
		*/

		if(my_displayMode==1 && originalInputSparse[0][i].status != 1 && originalInputSparse[0][i].status!= 2) continue;
		if(my_displayMode==2 && originalInputSparse[0][i].status != 1) continue;
		if(my_displayMode>2) continue;

		if(originalInputSparse[0][i].idpeth < 0) continue;


		float depth = 1.0f / originalInputSparse[0][i].idpeth;
		float depth4 = depth*depth; depth4*= depth4;
		float var = (1.0f / (originalInputSparse[0][i].idepth_hessian+0.01));

		if(var * depth4 > my_scaledTH)
			continue;

		if(var > my_absTH)
			continue;

		if(originalInputSparse[0][i].relObsBaseline < my_minRelBS)
			continue;


		for(int pnt=0;pnt<patternNum;pnt++)
		{

			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;
			int dx = patternP[pnt][0];
			int dy = patternP[pnt][1];

			tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[0][i].u+dx)*fxi[0] + cxi[0]) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[0][i].v+dy)*fyi[0] + cyi[0]) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi[0] * (rand()/(float)RAND_MAX-0.5f));

			// // float x_lb3_c1[6] = {0.014543, 0.014543, 0.014543,-138.449751, 89.703877,-66.518051};
			// // float x_lb3_c4[6] = {0.011238, 0.011238, -0.000393, -160.239278, 89.812338,127.472911};
			// // float x_lb3_c5[6] = {0.041862, -0.001905, -0.000212, 160.868615, 89.914152,160.619894};

			// Vec3 temp;
			
			// if(idx==0)
			// {
			// 	tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[idx][i].u+dx)*fxi[idx] + cxi[idx]) * depth;
			// 	tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[idx][i].v+dy)*fyi[idx] + cyi[idx]) * depth;
			// 	tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi[idx] * (rand()/(float)RAND_MAX-0.5f));
			// }
			// // else if (idx == 1)
			// // {
			// // 	temp[0] = ((originalInputSparse[idx][i].u+dx)*fxi[idx] + cxi[idx]) * depth;
			// // 	temp[1] = ((originalInputSparse[idx][i].v+dy)*fyi[idx] + cyi[idx]) * depth;
			// // 	temp[2] = depth*(1 + 2*fxi[idx] * (rand()/(float)RAND_MAX-0.5f));	
			// // 	SE3 project = T_lb_c[0].inverse()*T_lb_c[idx];
			// // 	temp = project*temp;		
			// // 	tmpVertexBuffer[vertexBufferNumPoints][0] = temp[0];
			// // 	tmpVertexBuffer[vertexBufferNumPoints][1] = temp[1];
			// // 	tmpVertexBuffer[vertexBufferNumPoints][2] = temp[2];	
			// // }
			// // else if (idx == 2)
			// // {
			// // 	temp[0] = ((originalInputSparse[idx][i].u+dx)*fxi[idx] + cxi[idx]) * depth;
			// // 	temp[1] = ((originalInputSparse[idx][i].v+dy)*fyi[idx] + cyi[idx]) * depth;
			// // 	temp[2] = depth*(1 + 2*fxi[idx] * (rand()/(float)RAND_MAX-0.5f));	
			// // 	SE3 project = T_lb_c[0].inverse()*T_lb_c[idx];
			// // 	temp = project*temp;		
			// // 	tmpVertexBuffer[vertexBufferNumPoints][0] = temp[0];
			// // 	tmpVertexBuffer[vertexBufferNumPoints][1] = temp[1];
			// // 	tmpVertexBuffer[vertexBufferNumPoints][2] = temp[2];			
			// // }
			// // else if (idx == 3)
			// // {
			// // 	temp[0] = ((originalInputSparse[idx][i].u+dx)*fxi[idx] + cxi[idx]) * depth;
			// // 	temp[1] = ((originalInputSparse[idx][i].v+dy)*fyi[idx] + cyi[idx]) * depth;
			// // 	temp[2] = depth*(1 + 2*fxi[idx] * (rand()/(float)RAND_MAX-0.5f));	
			// // 	SE3 project = T_lb_c[0].inverse()*T_lb_c[idx];
			// // 	temp = project*temp;			
			// // 	tmpVertexBuffer[vertexBufferNumPoints][0] = temp[0];
			// // 	tmpVertexBuffer[vertexBufferNumPoints][1] = temp[1];
			// // 	tmpVertexBuffer[vertexBufferNumPoints][2] = temp[2];			
			// // }
			// // else if (idx == 4)
			// // {
			// // 	temp[0] = ((originalInputSparse[idx][i].u+dx)*fxi[idx] + cxi[idx]) * depth;
			// // 	temp[1] = ((originalInputSparse[idx][i].v+dy)*fyi[idx] + cyi[idx]) * depth;
			// // 	temp[2] = depth*(1 + 2*fxi[idx] * (rand()/(float)RAND_MAX-0.5f));	
			// // 	SE3 project = T_lb_c[0].inverse()*T_lb_c[idx];
			// // 	temp = project*temp;		
			// // 	tmpVertexBuffer[vertexBufferNumPoints][0] = temp[0];
			// // 	tmpVertexBuffer[vertexBufferNumPoints][1] = temp[1];
			// // 	tmpVertexBuffer[vertexBufferNumPoints][2] = temp[2];			
			// // }
			// else
			// {
			// 	temp[0] = ((originalInputSparse[idx][i].u+dx)*fxi[idx] + cxi[idx]) * depth;
			// 	temp[1] = ((originalInputSparse[idx][i].v+dy)*fyi[idx] + cyi[idx]) * depth;
			// 	temp[2] = depth*(1 + 2*fxi[idx] * (rand()/(float)RAND_MAX-0.5f));	
			// 	SE3 project = T_lb_c[0].inverse()*T_lb_c[idx];
			// 	temp = project*temp;		
			// 	tmpVertexBuffer[vertexBufferNumPoints][0] = temp[0];
			// 	tmpVertexBuffer[vertexBufferNumPoints][1] = temp[1];
			// 	tmpVertexBuffer[vertexBufferNumPoints][2] = temp[2];	
			// }


			if(my_displayMode==0)
			{
				if(originalInputSparse[0][i].status==0)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}
				else if(originalInputSparse[0][i].status==1)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 0;
				}
				else if(originalInputSparse[0][i].status==2)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 0;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}
				else if(originalInputSparse[0][i].status==3)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 255;
					tmpColorBuffer[vertexBufferNumPoints][1] = 0;
					tmpColorBuffer[vertexBufferNumPoints][2] = 0;
				}
				else
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 255;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}

			}
			else
			{
				tmpColorBuffer[vertexBufferNumPoints][0] = originalInputSparse[0][i].color[pnt];
				tmpColorBuffer[vertexBufferNumPoints][1] = originalInputSparse[0][i].color[pnt];
				tmpColorBuffer[vertexBufferNumPoints][2] = originalInputSparse[0][i].color[pnt];
			}
			vertexBufferNumPoints++;

			assert(vertexBufferNumPoints <= numSparsePoints[0]*patternNum);
		}
	}

	if(vertexBufferNumPoints==0)
	{
		delete[] tmpColorBuffer;
		delete[] tmpVertexBuffer;
		return true;
	}

	numGLBufferGoodPoints = vertexBufferNumPoints;
	if(numGLBufferGoodPoints > numGLBufferPoints)
	{
		numGLBufferPoints = vertexBufferNumPoints*1.3;
		vertexBuffer[0].Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
		colorBuffer[0].Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
	}
	vertexBuffer[0].Upload(tmpVertexBuffer, sizeof(float)*3*numGLBufferGoodPoints, 0);
	colorBuffer[0].Upload(tmpColorBuffer, sizeof(unsigned char)*3*numGLBufferGoodPoints, 0);
	bufferValid=true;
	delete[] tmpColorBuffer;
	delete[] tmpVertexBuffer;
	// }
	// 2021.12.14

	return true;
}



void KeyFrameDisplay::drawCam(float lineWidth, float* color, float sizeFactor)
{
	if(width == 0)
		return;

	int cam_idx = 0;

	float sz=sizeFactor;

	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
		glMultMatrixf((GLfloat*)m.data());

		if(color == 0)
		{
			glColor3f(1,0,0);
		}
		else
			glColor3f(color[0],color[1],color[2]);

		glLineWidth(lineWidth);
		glBegin(GL_LINES);
		glVertex3f(0,0,0);
		// 2021.12.14
		glVertex3f(sz*(0-cx[cam_idx])/fx[cam_idx],sz*(0-cy[cam_idx])/fy[cam_idx],sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx[cam_idx])/fx[cam_idx],sz*(height-1-cy[cam_idx])/fy[cam_idx],sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx[cam_idx])/fx[cam_idx],sz*(height-1-cy[cam_idx])/fy[cam_idx],sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx[cam_idx])/fx[cam_idx],sz*(0-cy[cam_idx])/fy[cam_idx],sz);

		glVertex3f(sz*(width-1-cx[cam_idx])/fx[cam_idx],sz*(0-cy[cam_idx])/fy[cam_idx],sz);
		glVertex3f(sz*(width-1-cx[cam_idx])/fx[cam_idx],sz*(height-1-cy[cam_idx])/fy[cam_idx],sz);

		glVertex3f(sz*(width-1-cx[cam_idx])/fx[cam_idx],sz*(height-1-cy[cam_idx])/fy[cam_idx],sz);
		glVertex3f(sz*(0-cx[cam_idx])/fx[cam_idx],sz*(height-1-cy[cam_idx])/fy[cam_idx],sz);

		glVertex3f(sz*(0-cx[cam_idx])/fx[cam_idx],sz*(height-1-cy[cam_idx])/fy[cam_idx],sz);
		glVertex3f(sz*(0-cx[cam_idx])/fx[cam_idx],sz*(0-cy[cam_idx])/fy[cam_idx],sz);

		glVertex3f(sz*(0-cx[cam_idx])/fx[cam_idx],sz*(0-cy[cam_idx])/fy[cam_idx],sz);
		glVertex3f(sz*(width-1-cx[cam_idx])/fx[cam_idx],sz*(0-cy[cam_idx])/fy[cam_idx],sz);
		// 2021.12.14

		glEnd();
	glPopMatrix();
}


void KeyFrameDisplay::drawPC(float pointSize)
{

	if(!bufferValid || numGLBufferGoodPoints==0)
		return;


	glDisable(GL_LIGHTING);

	glPushMatrix();

		// 2021.12.14
		//for(int idx=0;idx<cam_num;idx++)
		//{
			Sophus::Matrix4f m = (T_c_c0[0]*camToWorld*T_c0_c[0]).matrix().cast<float>(); // 2021.12.14
			glMultMatrixf((GLfloat*)m.data());

			glPointSize(pointSize);


			colorBuffer[0].Bind();
			glColorPointer(colorBuffer[0].count_per_element, colorBuffer[0].datatype, 0, 0);
			glEnableClientState(GL_COLOR_ARRAY);

			vertexBuffer[0].Bind();
			glVertexPointer(vertexBuffer[0].count_per_element, vertexBuffer[0].datatype, 0, 0);
			glEnableClientState(GL_VERTEX_ARRAY);
			glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
			glDisableClientState(GL_VERTEX_ARRAY);
			vertexBuffer[0].Unbind();

			glDisableClientState(GL_COLOR_ARRAY);
			colorBuffer[0].Unbind();
		//}
		// 2021.12.14

	glPopMatrix();
}

}
}