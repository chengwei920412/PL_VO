//
// Created by rain on 17-12-28.
//

#include "PointFeature.h"

namespace PL_VO
{

void PointFeature::detectPointfeature(const cv::Mat &img, vector<cv::KeyPoint> &vkeypoints, cv::Mat &pointdesc)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img, vkeypoints);

    if (vkeypoints.size() > Config::orbNFeatures() && Config::orbNFeatures() != 0)
    {
        sort(vkeypoints.begin(), vkeypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b)
        {
            return (a.response > b.response);
        });

        vkeypoints.resize((size_t)Config::orbNFeatures());

        for (int i = 0; i < Config::lsdNFeatures(); i++)
        {
            vkeypoints[i].class_id = i;
        }

        descriptor->compute(img, vkeypoints, pointdesc);
    }
    else
    {
        descriptor->compute(img, vkeypoints, pointdesc);
    }
}

void PointFeature::matchPointFeatures(const cv::Mat &pointdesc1, const cv::Mat &pointdesc2, vector<cv::DMatch> &vpointmatches12)
{
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create("BruteForce-Hamming");

    matcher->match(pointdesc1, pointdesc2, vpointmatches12);

    CHECK(vpointmatches12.size() > 8) << " the point matches's size is less than eight" << endl;
}

vector<cv::DMatch> PointFeature::refineMatchesWithDistance(const vector<cv::DMatch> &vmatches)
{
    vector<cv::DMatch> inliermatches;
    double mindist=10000, maxdist=0;

    for (int i = 0; i < vmatches.size(); i++)
    {
        double dist = vmatches[i].distance;
        if (dist < mindist) mindist = dist;
        if (dist > maxdist) maxdist = dist;
    }

    cout << "max dist " << maxdist << endl;
    cout << "min dist " << mindist << endl;

    for (int i = 0; i < vmatches.size(); i++)
    {
        if (vmatches[i].distance <= max(2*mindist, 30.0))
        {
            inliermatches.push_back(vmatches[i]);
        }
    }

    return inliermatches;
}

vector<cv::DMatch> PointFeature::refineMatchesWithFundamental(const vector<cv::KeyPoint> &vKeyPoints1,
                                                              const vector<cv::KeyPoint> &vKeyPoints2,
                                                              const vector<cv::DMatch> &vmathes)
{
    CHECK(vmathes.size() > 8) << " the keypoint's size is less than eight ";

    vector<cv::Point2f> vpoint2d1;
    vector<cv::Point2f> vpoint2d2;
    vector<uchar> vinliersMask(vmathes.size());

    for (size_t i = 0; i < vmathes.size(); i++)
    {
        vpoint2d1.emplace_back(vKeyPoints1[vmathes[i].queryIdx].pt);
        vpoint2d2.emplace_back(vKeyPoints2[vmathes[i].trainIdx].pt);
    }

    cv::findFundamentalMat(vpoint2d1, vpoint2d2, cv::FM_RANSAC, 5.0, 0.99, vinliersMask);

    vector<cv::DMatch> vinliersMatch;

    for (size_t i = 0; i < vinliersMask.size(); i++)
    {
        if (vinliersMask[i])
            vinliersMatch.push_back(vmathes[i]);
    }

    return vinliersMatch;
}

} // namespace PL_VO