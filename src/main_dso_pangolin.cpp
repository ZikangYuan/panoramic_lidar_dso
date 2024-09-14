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



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <mutex>
#include <condition_variable>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs/legacy/constants_c.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>


using namespace std;

float scaling = 0.005;// 5 mm
float offset = -100.0;

typedef pcl::PointXYZI  PointType;
struct PointXYZIR
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIR,  
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (uint16_t, ring, ring)
)

ImageFolderReader* reader[CAM_NUM];

std::string dataset = "/media/xiang/OneTouch/dataset/nclt/";
std::string lidar_dataset = "";
std::string image_dataset = "";
std::string vignette = "";
std::string gammaCalib = "";
std::string calib = "";
std::string undistortion = "";
std::string resultPath = "";
std::string pathSensorPrameter = "";
std::string sequence = "";
int seg=0;
double startT[13];
double endT[13];
double rescale = 1;
static int downSample = 1;
bool useOnlyCamera = false;
bool reverse = false;
bool disableROS = false;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;

// 2020.06.23 yzk
FullSystem* fullSystem = NULL;
bool firstFlag = true;
double firstFrameTime;
bool initialization = false;
double initialtimestamp = 0.0;
double timestamp = 0.0;
int currentId = 0;
struct timeval tv_start;
clock_t started;
double sInitializerOffset=0;
// 2020.06.23 yzk

// 2020.10.05 yzk
pcl::PointCloud<PointXYZIR> laser_cloud;
pcl::PointCloud<pcl::PointXYZ> laser_cloud_ijrr;


std::vector<std::string> image_files[CAM_NUM];
std::vector<std::string> lidar_files;
std::vector<std::string> lidar_files_ijrr[CAM_NUM];
std::vector<std::string> utimes_dataset;

PointType nanPoint; // fill in fullCloud at each iteration


float startOrientation;
float endOrientation;


extern const bool loopClosureEnableFlag = false;
extern const double mappingProcessInterval = 0.3; /*0.3*/

extern const float scanPeriod = 0.1;
extern const int systemDelay = 0;

extern const float sensorMountAngle = 0.0;
extern const float segmentTheta = /*ori:60 ck:0.0*/60.0/180.0*M_PI; // decrese this value may improve accuracy 点云分割时的角度跨度上限（π/3）
extern const int segmentValidPointNum = 5/*ori:5 yzk:50*/; // 检查上下左右连续5个点做为分割的特征依据
extern const int segmentValidLineNum = 3/*ori:3 yzk:50*/; // 检查上下左右连续3线做为分割的特征依据
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI; //转成弧度
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI; //转成弧度
//Vel 64


int mode=1;

bool firstRosSpin=false;

using namespace dso;


void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}



void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}


void parseArgument(char* arg)
{
	int option;
	float foption;
	char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

    if(1==sscanf(arg,"IJRR=%d",&option))
    {
        if(option==1)
        {
            setting_useNCLT = false;
            printf("Using ijrr dataset!!\n");
        }
        return;
    }

    if(1==sscanf(arg,"optDepth=%d",&option))
    {
        if(option==1)
        {
            setting_optDepth = true;
            printf("Optimize depth!!\n");
        }
        else if(option==0)
        {
            setting_optDepth = false;
            printf("Don't optimize depth!!\n");
        }
        return;
    }

    if(1==sscanf(arg,"estimateType=%d",&option))
    {
        if(option==1)
        {
            estimateType = 1;
            printf("Using MLE!!\n");
        }
        else if(option==2)
        {
            estimateType = 2;
            printf("Using James-Stein type!!\n");
        }
        else if(option==3)
        {
            estimateType = 3;
            printf("Using Bayes estimator!!\n");
        }
        return;
    }
    if(1==sscanf(arg,"weightOfMotion=%f",&foption))
    {
        setting_weightMotion = foption;
        return;
    }

    if(1==sscanf(arg,"seg=%d",&option))
    {
        seg = option;
        return;
    }

    if(1==sscanf(arg,"KF_selection=%d",&option))
    {
        setting_useKFselection = option;
        if(option)
            printf("Using KF selection!!\n");
        return;
    }

	if(1==sscanf(arg,"preset=%d",&option))
	{
		settingsDefault(option);
		return;
	}


	if(1==sscanf(arg,"rec=%d",&option))
	{
		if(option==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}



	if(1==sscanf(arg,"noros=%d",&option))
	{
		if(option==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}
	if(1==sscanf(arg,"useOnlyCamera=%d",&option))
	{
		if(option==1)
		{
			useOnlyCamera = true;
			printf("useOnlyCamera!\n");
		}
		return;
	}
	// if(1==sscanf(arg,"reverse=%d",&option))
	// {
	// 	if(option==1)
	// 	{
	// 		reverse = true;
	// 		printf("REVERSE!\n");
	// 	}
	// 	return;
	// }
	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"prefetch=%d",&option))
	{
		if(option==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
		return;
	}

	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}   

	if(1==sscanf(arg,"undistort=%s",buf))
	{
		undistortion = buf;
		printf("loading undistortion from %s!\n", undistortion.c_str());
		return;
	} 

	if(1==sscanf(arg,"dataset=%s",buf))
	{
		dataset = buf;
		printf("loading dataset from %s!\n", dataset.c_str());
		return;
	}  

	if(1==sscanf(arg,"dataset_lidar=%s",buf))
	{
		lidar_dataset = buf;
		return;
	}  

	if(1==sscanf(arg,"dataset_image=%s",buf))
	{
		image_dataset = buf;
		return;
	}  

	if(1==sscanf(arg,"pathSensorPrameter=%s",buf))
	{
		pathSensorPrameter = buf;
		printf("loading extrinsic from %s!\n", pathSensorPrameter.c_str());
		return;
	}  

	if(1==sscanf(arg,"resultPath=%s",buf))
	{
		resultPath = buf;
		printf("save result in %s!\n", resultPath.c_str());
		return;
	}  

	if(1==sscanf(arg,"sequence=%s",buf))
	{
		sequence = buf;
		return;
	}  

	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
		return;
	}

	if(1==sscanf(arg,"rescale=%f",&foption))
	{
		rescale = foption;
		printf("RESCALE %f!\n", rescale);
		return;
	}

	if(1==sscanf(arg,"speed=%f",&foption))
	{
		playbackSpeed = foption;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
		return;
	}

	if(1==sscanf(arg,"save=%d",&option))
	{
		if(option==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	if(1==sscanf(arg,"mode=%d",&option))
	{

		mode = option;
		if(option==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(option==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		if(option==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
		}
		return;
	}

}

void getMeasurements(int n)
{
	std::cout << "sequence: " << sequence << ",seg: " << seg <<std::endl;
	std::cout << "n: " << n <<std::endl;
    // ---------------------------process img------------------------------ //
    cv::Mat currentFrame[CAM_NUM];
    for(int i=0;i<cam_num;i++)
    {
        // std::cout << image_files[i][n] << std::endl;
        cv::Mat image_cam = cv::imread(image_files[i][n], CV_LOAD_IMAGE_GRAYSCALE);
        // cv::imshow("image",image_cam);
        // cv::waitKey(0);
        if(image_cam.cols*image_cam.rows==0){
            cout<<image_files[i][n]<<" doesn't exits!!!continue."<<endl;
            return;
        }
        if(!setting_useNCLT)
			cv::resize(image_cam, image_cam, cv::Size(image_cam.cols, image_cam.rows * 2), 0, 0, cv::INTER_LINEAR); 
        //  cv::imshow("image",image_cam);
        // cv::waitKey(0);                
        
        currentFrame[i] = image_cam;
    }

    // -----------------------process lidar cloud-------------------------- //
    if(!useOnlyCamera)
    {
        if(setting_useNCLT)
        {
            Eigen::Vector4d lidarCloudTemp;
            PointXYZIR point;
            std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> Cloud_;
            std::vector<uchar> intensity;
            double timeLidarCloud = stod(utimes_dataset[n]);

            std::ifstream lidar_data_file(lidar_files[n], std::ifstream::in | std::ifstream::binary);

            if(!lidar_data_file.is_open())
            {
                cout<<lidar_files[n]<<" doesn't exits!!!continue."<<endl;
                return;
            }

            laser_cloud.clear();
            while(!lidar_data_file.eof()){
                unsigned char indensity,ring;
                unsigned short x,y,z;
                lidar_data_file.read((char*)&x,sizeof(unsigned short));
                lidar_data_file.read((char*)&y,sizeof(unsigned short));
                lidar_data_file.read((char*)&z,sizeof(unsigned short));
                lidar_data_file.read((char*)&indensity,sizeof(unsigned char));
                lidar_data_file.read((char*)&ring,1);
                if(!isfinite(x))
                    continue;
                point.x = x*scaling+offset; 
                point.y = y*scaling+offset;
                point.z = z*scaling+offset;
                point.intensity = indensity;
                point.ring = ring;
                // horizontalAngle = atan2(point.y,point.x) * 180 / M_PI;
                laser_cloud.push_back(point);
            }     
            lidar_data_file.close(); 

            // Remove Nan points
            // std::vector<int> indices;
            // pcl::removeNaNFromPointCloud(laser_cloud, laser_cloud, indices);

            // printf("point intensity");
            for (size_t i = 0; i < laser_cloud.points.size(); ++i){
                lidarCloudTemp(0, 0) = laser_cloud.points[i].x;
                lidarCloudTemp(1, 0) = laser_cloud.points[i].y;
                lidarCloudTemp(2, 0) = laser_cloud.points[i].z;
                lidarCloudTemp(3, 0) = laser_cloud.points[i].ring;
                intensity.push_back(laser_cloud.points[i].intensity);
                // printf(" %d, ",intensity[i]);
                Cloud_.push_back(lidarCloudTemp);
            }
            fullSystem->Clouds.push(Cloud_);
            fullSystem->Intensities.push(intensity);
            // resetParameters();
            fullSystem->qTimeLidarCloud.push(timeLidarCloud/1e6);
        }
        else
        {
            // if (lidar_files_ijrr[0].empty())
            //     return ;
            for(int i=0;i<cam_num;i++){ if(image_files[i].empty()) return ; }

            double timeLidarCloud = stod(utimes_dataset[n]);
            // for(int idx=0;idx<cam_num;idx++)
            // {
                laser_cloud_ijrr.clear();

                if (pcl::io::loadPCDFile<pcl::PointXYZ>(lidar_files[n], laser_cloud_ijrr) == -1) {
                    std::cout << "Couldn't read file "<<lidar_files[n]<< "!" << std::endl;
                    return;
                }

                // Remove Nan points
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(laser_cloud_ijrr, laser_cloud_ijrr, indices);

                Eigen::Vector4d lidarCloudTemp;
                std::vector<Eigen::Vector4d,Eigen::aligned_allocator<Eigen::Vector4d>> cloud;

                int numGround = 0; int numAll = 0;
                for (size_t i = 0; i < laser_cloud_ijrr.points.size(); ++i)
                {
                    lidarCloudTemp(0, 0) = laser_cloud_ijrr.points[i].x;
                    lidarCloudTemp(1, 0) = laser_cloud_ijrr.points[i].y;
                    lidarCloudTemp(2, 0) = laser_cloud_ijrr.points[i].z;
                    lidarCloudTemp(3, 0) = 1;

                    cloud.push_back(lidarCloudTemp);
                }
                fullSystem->Clouds.push(cloud);
            //}
            fullSystem->qTimeLidarCloud.push(timeLidarCloud/1e6);
        }
        std::cout << "laser_cloud_ijrr.size: " <<  laser_cloud_ijrr.size() << std::endl;
        
    }
    else{
        double timeLidarCloud = stod(utimes_dataset[n]);
        fullSystem->qTimeLidarCloud.push(timeLidarCloud/1e6);
    }
    for(int i=0;i<cam_num;i++)
        fullSystem->qImg[i].push(currentFrame[i]);
    fullSystem->addFeaturePoint = false;
    if(useOnlyCamera)
    {
        fullSystem->addFeaturePoint = true;
        fullSystem->useOnlyCamera = true;
    }
    /*
    if(float(numGround)/(float)numAll > 0.8)
        fullSystem->addFeaturePoint = false;
    else
        fullSystem->addFeaturePoint = false;
    */
    // -----------------------process lidar cloud-------------------------- //
    
    return ;
}


void process(int n)
{   
	std::cout << "process!!!" <<std::endl;
    getMeasurements(n);
	std::cout << "读入一帧!!!" <<std::endl;
    if(fullSystem->qImg[0].size()==0)
        return;
    if(!useOnlyCamera)
        assert(fullSystem->qImg[0].size()==1);
    else
        assert(fullSystem->qImg[0].size()==1&&fullSystem->Clouds.size()==1);
        // if (setting_useNCLT)
        //     assert(fullSystem->qImg[0].size()==1&&fullSystem->Clouds.size()==1);
        // else
        //     assert(fullSystem->qImg[0].size()==1&&fullSystem->Cloud_ijrr[0].size()==1);
    // printf("Size of img is %d,",fullSystem->qImg[0].size());
    while(1)
    {
        for(int i=0;i<cam_num;i++)
            if(fullSystem->qImg[i].empty()) 
                return;

        std::vector<cv::Mat> currentFrame(cam_num);
        for(int i=0;i<cam_num;i++)
            currentFrame[i] = fullSystem->qImg[i].front();
        
        timestamp = fullSystem->qTimeLidarCloud.front();

        if(currentId == 0)
            initialtimestamp = timestamp;

        if(!fullSystem->initialized)	// if not initialized: reset start time.
        {
            gettimeofday(&tv_start, NULL);
            started = clock();
            sInitializerOffset = timestamp - initialtimestamp;
        }

        if(firstFlag)
        {
            firstFrameTime = timestamp;
            firstFlag = false;
        }

        std::vector<ImageAndExposure*> img(cam_num);
        for(int i=0;i<cam_num;i++)
            img[i] = reader[i]->getRosImage(currentFrame[i], timestamp);
        
        
        bool skipFrame=false;

        if(!skipFrame) fullSystem->addActiveFrame(img, currentId);

        currentId++;

        for(int i=0;i<cam_num;i++)
        {
            delete img[i];
            fullSystem->qImg[i].pop();
        }
        
        if(!useOnlyCamera)
        {
            fullSystem->Clouds.pop();
        // if(setting_useNCLT)
        //     fullSystem->Clouds.pop();
        // else
        //     // for(int i=0;i<cam_num;i++)
        //         fullSystem->Cloud_ijrr.pop();
        fullSystem->qTimeLidarCloud.pop();
        //fullSystem->Intensities.pop();
        }

        if(fullSystem->initFailed || setting_fullResetRequested)
        {
            if(currentId < 250 || setting_fullResetRequested)
            {
                printf("RESETTING!\n");

                std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                delete fullSystem;

                for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                fullSystem = new FullSystem();
                fullSystem->setGammaFunction(reader[0]->getPhotometricGamma());
                fullSystem->linearizeOperation = (playbackSpeed==0);
                fullSystem->outputWrapper = wraps;

                setting_fullResetRequested=false;
            }
        }

        if(fullSystem->isLost)
        {
            printf("LOST!!\n");
            return;
        }
    }
}

void loaddataset(std::string image_path, std::string lidar_path, int idx)
{
    DIR *dp;
    struct dirent *dirp;
    std::string::size_type position;
	printf("loading lidar from %s!\n", lidar_path.c_str());
	printf("loading img from %s!\n", image_path.c_str());

    if((dp  = opendir(lidar_path.c_str())) == NULL)
    {
        cout<<"no such file!!"<<lidar_path.c_str()<<endl;
        return ;
    }
    utimes_dataset.clear();
    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);
        position = name.find(".");
        name = name.substr(0,position);
        if(name != "" && name != " ")
        {
            if(stod(name)<=startT[idx])
                continue;
            else if(stod(name)>=endT[idx])
                continue;
            utimes_dataset.push_back(name);
        }
    }
    closedir(dp);
    sort(utimes_dataset.begin(), utimes_dataset.end());
    for(int i=0;downSample*i<utimes_dataset.size();i++)
        utimes_dataset[i] = utimes_dataset[downSample*i];
    utimes_dataset.resize(utimes_dataset.size()/downSample);

    for (int j=0;j<utimes_dataset.size();j++)
    {
        string name = utimes_dataset[j]+".bin";
        name = lidar_path + name;
        lidar_files.push_back(name);
    }

    for (int i=0;i<CAM_NUM;i++)
    {
        for (int j=0;j<utimes_dataset.size();j++)
        {
            string name = "Cam"+to_string(cam_idx_array[i])+"/"+utimes_dataset[j]+".tiff";
            name = image_path + name;
            image_files[i].push_back(name);
        }
    }
}

void loaddataset(std::string image_path, int idx)
{
    DIR *dp;
    struct dirent *dirp;
    std::string::size_type position;
    string cam1 = image_path + "Cam1";
	printf("loading timestamp from %s!\n", cam1.c_str());

    if((dp  = opendir(cam1.c_str())) == NULL)
    {
        cout<<"no such file!!"<<cam1.c_str()<<endl;
        return ;
    }
    utimes_dataset.clear();

    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);
        position = name.find(".");
        name = name.substr(0,position);
        //if(name != "" && name != " ")
        if(name != "" && name != " " )
        {
            if(stod(name)<=startT[idx])
                continue;
            else if(stod(name)>=endT[idx])
                continue;
            utimes_dataset.push_back(name);
        }
    }
    closedir(dp);
    sort(utimes_dataset.begin(), utimes_dataset.end());
    std::cout<<"Loaded "<<utimes_dataset.size()<<" timestamps"<<std::endl;
/* 
    for (int j=0;j<utimes_dataset.size();j++)
    {
        string name = utimes_dataset[j]+".bin";
        name = image_path + name;
        lidar_files.push_back(name);
    } */

    for (int i=0;i<CAM_NUM;i++)
    {
        for (int j=0;j<utimes_dataset.size();j++)
        {
            string name = "Cam"+to_string(cam_idx_array[i])+"/"+utimes_dataset[j]+".tiff";
            name = image_path + name;
            image_files[i].push_back(name);
        }
    }
}

void loaddataset_ijrr(std::string dataset_path)
{
	dataset_path = dataset_path + sequence + "/";
    std::cout << dataset_path << std::endl;
	std::string in_path;

	in_path = dataset_path + "Timestamp.log";

	printf("Reading timestamp from file %s",in_path.c_str());

	std::ifstream f(in_path.c_str());		
    if (!f.good())
    {
        f.close();
        printf(" ... not found. Cannot operate without timestamp, shutting down.\n");
        f.close();
        return ;
    }
	std::string tmp, time;
	std::string offset="1";
    std::string end="4729";
	while(1)
	{
		f >> tmp;
		f >> time;
        // std::cout<<tmp<<std::endl;
		f >> time;
		if(tmp==offset)
			break;
		f >> time;
	}
    utimes_dataset.clear();
	utimes_dataset.push_back(time);

	if(sequence=="1") { /*startT = 1335704132113151; endT = 1335708191119947;*/ }
	else if(sequence=="2") { /*startT = 1339759077460675; endT = 1339763829773952;*/ }
	
	while(!f.eof())
	{
		f >> tmp;
        f >> tmp;
		f >> time;
		f >> tmp;
		utimes_dataset.push_back(time);
		// if(tmp==end)
		// 	break;
	}
	f.close();
    sort(utimes_dataset.begin(), utimes_dataset.end());

    if (sequence == "ford_1")
    {
    	for (int j = 4; j < 2766; j++)
        {
            std::stringstream lidar_path;
            lidar_path << dataset_path << "pcd"<<"/" << std::setfill('0') << std::setw(4) << j << ".pcd";
            // lidar_path <<"/media/xiang/Xiang_P/2pcd/"<<"/" << std::setfill('0') << std::setw(4) << j << ".pcd";
            lidar_files.push_back(lidar_path.str());
            for(int i=0;i<cam_num;i++)
            {
                std::stringstream image_path;
                image_path << dataset_path << "IMAGES/Cam"<< to_string(cam_idx_array[i]) <<"/image" << std::setfill('0') << std::setw(4) << j << ".ppm";
                image_files[i].push_back(image_path.str());
            }
            std::cout << j <<std::endl;
        }
    }
    if (sequence == "ford_2")
    {
    	if (seg == 0)
    	{
    		for (int j = 4; j < 1464; j++)
	        {
	            std::stringstream lidar_path;
	            lidar_path << dataset_path << "pcd"<<"/" << std::setfill('0') << std::setw(4) << j << ".pcd";
	            // lidar_path <<"/media/xiang/Xiang_P/2pcd/"<<"/" << std::setfill('0') << std::setw(4) << j << ".pcd";
	            lidar_files.push_back(lidar_path.str());
	            for(int i=0;i<cam_num;i++)
	            {
	                std::stringstream image_path;
	                image_path << dataset_path << "IMAGES/Cam"<< to_string(cam_idx_array[i]) <<"/image" << std::setfill('0') << std::setw(4) << j << ".ppm";
	                image_files[i].push_back(image_path.str());
	            }
	        }
    	}
    	else if (seg == 1)
    	{
    		for (int j = 1464; j < 2983; j++)
	        {
	            std::stringstream lidar_path;
	            lidar_path << dataset_path << "pcd"<<"/" << std::setfill('0') << std::setw(4) << j << ".pcd";
	            // lidar_path <<"/media/xiang/Xiang_P/2pcd/"<<"/" << std::setfill('0') << std::setw(4) << j << ".pcd";
	            lidar_files.push_back(lidar_path.str());
	            for(int i=0;i<cam_num;i++)
	            {
	                std::stringstream image_path;
	                image_path << dataset_path << "IMAGES/Cam"<< to_string(cam_idx_array[i]) <<"/image" << std::setfill('0') << std::setw(4) << j << ".ppm";
	                image_files[i].push_back(image_path.str());
	            }
	        }
    	}
	    	
    }
}

void loaddataset_nclt()
{
    lidar_dataset = dataset+sequence+"/velodyne_sync/";
    image_dataset = dataset+sequence+"/lb3/";
    if(sequence=="2012-04-29") 
    { 
        startT[0] = 1335704132113151;
    }
    else if(sequence=="2012-01-08") 
    {
        startT[0]=1326030980126139;    
        endT[0]=1326033790977678;
        startT[1]=1326033821578255;    
        endT[1]=1326034114183833;
        startT[2]=1326034127184111;    
        endT[2]=1326034336588533;
        startT[3]=1326034341588639;    
        endT[3]=1326034464591192;
        startT[4]=1326034479191514;    
        endT[4]=1326034640594874;
        startT[5]=1326034642394916;    
        endT[5]=1326034856599384;
        startT[6]=1326034874399786;    
        endT[6]=1326034959001612;
        startT[4]=1326034479191514;       
        endT[4]=1326034959001612;
        startT[7]=1326034968801819;    
        endT[7]=1326035112604974;
        startT[8]=1326035132005432;    
        endT[8]=1326035899821169;
        startT[9]=1326035914421404;    
        endT[9]=1326035997423010;
        startT[10]=1326036005623171;    
        endT[10]=1326036461432068;
        startT[11]=1326036474432324;    
        endT[11]=1326036608635029;
        
        startT[12]=1326030980126139;
        endT[12]=1326036608635029;
    }
    else if(sequence=="2012-06-15") 
    { 
        startT[0] = 1339759077460675;
        endT[0]   = 1339761784010131;
        startT[1] = 1339761936607334;
        endT[1]   = 1339763106986538; 
    }        
    else if(sequence=="2012-11-04") 
    { 
        endT[0] = 1352040593994773; 
        startT[1] = 1352040679380633; 
        endT[1]   = 1352041575232725; 
        startT[2] = 1352041758002618; 
        endT[2]   = 1352042366902617; 
        startT[3] = 1352042506079746; 
        endT[3]   = 1352043287351321; 
        startT[4] = 1352043599700085; 
        endT[4]   = 1352044656127265;

        startT[5] = 1352040608592377; // same with 04-05_5 long coor 287
        endT[5]   = 1352040666182834; 
        startT[6] = 1352041601628365; // two yellow 677
        endT[6]   = 1352041737206027; 
        startT[7] = 1352042382100116; // 3 circle 6022
        endT[7]   = 1352043586302257; 

        startT[8] = 1352039974097463;
        endT[8]   = 1352044656127265; 
    }
    else if(sequence=="2012-12-01") 
    {  
        startT[0] = 1354396870671183; 
        endT[0]   = 1354397390216462; // 2022.8.6
        // endT[0]   = 1354397472407801;

        startT[1] = 1354397476407379;
        endT[1]   = 1354398299520501; 

        startT[2] = 1354398408708928;
        endT[2]   = 1354399807760347;

        startT[3] = 1354400020337815;
        endT[3]   = 1354400209117926;

        startT[4] = 1354400348903214; // night
        endT[4]   = 1354401328799333; //

        startT[5] = 1354397419213405; // long coor 264
        endT[5]   = 1354397472207819;
        startT[6] = 1354398316518698; // light 4896
        endT[6]   = 1354398396910176;
        startT[7] = 1354400231715549; // 3 circle. light 398
        endT[7]   = 1354400344903854;
        startT[8] = 1354401359596051; // different light 578
        endT[8]   = 1354401475383749;
    }
    else if(sequence=="2013-02-23") 
    { 
        startT[0] = 1361644442945901; 
        endT[0]   = 1361645008038307; 
        // 2022.8.6
        // startT[1] = 1361646512554222; 
        // endT[1]   = 1361648861509036; 
        // startT[2] = 1361648934095275; 
        // endT[2]   = 1361649134857262; 
        startT[1] = 1361645117617532; 
        endT[1]   = 1361645632920016; 
        startT[2] = 1361645758096369; 
        endT[2]   = 1361646282197525;
        startT[3] = 1361646512554222; 
        endT[3]   = 1361648861509036;
        startT[4] = 1361648934095275; 
        endT[4]   = 1361649134857262;
        // 2022.8.6

        startT[5] = 1361645026035191; // 2 yellow deng 374
        endT[5]   = 1361645100820713; 
        startT[6] = 1361645655115822; //  3 circle same 484
        endT[6]   = 1361645751897538; 
        startT[7] = 1361646307192803; // same with 04-05_6 994
        endT[7]   = 1361646506155440; 
        startT[8] = 1361648879705600; // 3 di tan 214
        endT[8]   = 1361648922497457; 

        startT[9] = 1361644442945901;
        endT[9]= 1361649134857262;

    }
    else if(sequence=="2013-04-05") 
    { 
        startT[0] = 1365177514828891; 
        endT[0]   = 1365178060527770; 
        startT[1] = 1365178156509931;
        endT[1]   = 1365179601244078; 
        startT[2] = 1365179829402209;
        endT[2]   = 1365180957994668; 
        startT[3] = 1365181081172010;
        endT[3]   = 1365181118365181; 
        startT[4] = 1365181239742953;
        endT[4]   = 1365181563683508; 

        startT[5] = 1365178093921489; // long coor
        endT[5]   = 1365178149711186; 
        startT[6] = 1365179626839339; // turn right
        endT[6]   = 1365179819404033; 
        startT[7] = 1365180976791203; // 3 circle
        endT[7]   = 1365181078372522; 
        startT[8] = 1365181143960482; // hu xing
        endT[8]   = 1365181230944551; 

        startT[9] = 1365177514828891; // hu xing
        endT[9]   = 1365181563683508;         
    }

    if(!useOnlyCamera)
        loaddataset(image_dataset,lidar_dataset,seg);
    else
        loaddataset(image_dataset,seg);
}

// 2020.06.22 yzk
int main(int argc, char** argv)
{

	for(int i=1; i<argc;i++)
		parseArgument(argv[i]);


    if(setting_useNCLT)
    {
        for(int i=0;i<10;i++)
        {
            startT[i] = 0;
            endT[i] = 1e20;
        }
        loaddataset_nclt();
    }
    else
        loaddataset_ijrr(dataset);

	// for(int i=1; i<utimes_dataset.size();i++)
	// 	std::cout << image_files[0][i] << std::endl;

    std::string intrinsic_path[CAM_NUM];
    for(int i=0;i<cam_num;i++){
        intrinsic_path[i] = calib + std::to_string(cam_idx_array[i]) + ".txt";
    }
    std::string undistort_path[CAM_NUM];
    for(int i=0;i<cam_num;i++){
        undistort_path[i] = undistortion + std::to_string(cam_idx_array[i]) + "_1616X1232.txt";
        reader[i] = new ImageFolderReader(intrinsic_path[i], gammaCalib, vignette, undistort_path[i]);
    }
    for(int i=0;i<cam_num;i++){
        reader[i]->setGlobalCalibration(i);
    }
    setLUT();

    // initial the system
    fullSystem = new FullSystem();
    fullSystem->loadTimeStamp(utimes_dataset);
	fullSystem->setGammaFunction(reader[0]->getPhotometricGamma());
	fullSystem->linearizeOperation = (playbackSpeed==0);

    if(setting_useNCLT)
	    fullSystem->loadSensorPrameters(pathSensorPrameter);
    else
        fullSystem->loadSensorPrameters_ijrr(pathSensorPrameter);

	IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

    std::thread runthread([&]() {

        gettimeofday(&tv_start, NULL);

        started = clock();

        std::cout << "utimes_dataset" << utimes_dataset.size() << std::endl;

        for(int ii=0;ii<utimes_dataset.size(); ii++)
        {
            if(fullSystem->isLost)
                break;
        	process(ii);
        }

        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);


        fullSystem->printResult(resultPath);


        int numFramesProcessed = currentId;
        double numSecondsProcessed = fabs(timestamp - firstFrameTime);
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*currentId) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)currentId << "\n";
            tmlog.flush();
            tmlog.close();
        }
    });

    if(viewer != 0)
        viewer->run();

    runthread.join();

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}


	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;

	printf("DELETE READER!\n");

    for(int i=0;i<cam_num;i++)
	    delete reader[i];

	printf("DSO OVER!\n");

	return 0;
}
// 2020.06.22 yzk