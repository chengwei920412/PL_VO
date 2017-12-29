//
// Created by rain on 17-12-28.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/highgui/highgui.hpp>


using namespace std;

// ./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml
// PATH_TO_SEQUENCE_FOLDER/mav0/cam0/data Examples/Monocular/EuRoC_TimeStamps/SEQUENCE.txt
void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{

    google::InitGoogleLogging(argv[0]);

    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;

    string strAssociationFilename("/home/rain/workspace/DataSets/rgbd_dataset_freiburg2_desk/associate.txt");
    string strSequenceFilename("/home/rain/workspace/DataSets/rgbd_dataset_freiburg2_desk");
    string strSettingsFile("../Example/TUM2.yaml");

    cout << "datasets asscociateion file: " << strAssociationFilename << endl;
    cout << "datasets sequence file: " << strSequenceFilename << endl;
    cout << "setting file: "<< strSettingsFile << endl;

    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    size_t imagesize = vstrImageFilenamesRGB.size();

    if (imagesize <= 0)
    {
        cerr << endl << "no images found in that path " << endl;
        return 1;
    }

    if (vstrImageFilenamesRGB.size() != vstrImageFilenamesD.size())
    {
        cerr << endl << "the number of the rgb image and depth image are different " << endl;
        return 1;
    }

    cv::Mat imRGB, imDepth;
    for (size_t i = 0; i < imagesize; i++)
    {
        imRGB = cv::imread(strSequenceFilename +"/"+ vstrImageFilenamesRGB[i], CV_LOAD_IMAGE_UNCHANGED);
        imDepth = cv::imread(strSequenceFilename +"/"+ vstrImageFilenamesD[i], CV_LOAD_IMAGE_UNCHANGED);
        double imagetimestams = vTimestamps[i];

        if (imRGB.empty())
        {
            cerr << "failed to load image file " << strSequenceFilename +"/"+ vstrImageFilenamesRGB[i] << endl;
        }
    }

}


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}