//
// Created by rain on 17-12-21.
//

#include <glog/logging.h>

#include "LineFeature.h"
#include "TicToc.h"

namespace PL_VO
{

LineFeature::LineFeature()
{}

void LineFeature::detectLinefeature(const cv::Mat img, vector<cv::line_descriptor::KeyLine> &vkeylines,
                                    cv::Mat &linedesc, const double minLinelength)
{
    cv::Ptr<cv::line_descriptor::LSDDetectorC> lsd = cv::line_descriptor::LSDDetectorC::createLSDDetectorC();
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();

    cv::line_descriptor::LSDDetectorC::LSDOptions opts;

    opts.refine = Config::lsdRefine();
    opts.scale = Config::lsdScale();
    opts.sigma_scale = Config::lsdSigmaScale();
    opts.quant = Config::lsdQuant();
    opts.ang_th = Config::lsdAngTh();
    opts.log_eps = Config::lsdLogEps();
    opts.density_th = Config::lsdDensityTh();
    opts.n_bins = Config::lsdNBins();
    opts.min_length = minLinelength;

    TicToc tictoc1;
    // there also can use the mask to set the dected aera
    lsd->detect(img, vkeylines, (int)Config::lsdScale(), 1, opts);

    cout << "keyline: " << vkeylines.size() << endl;
    cout << "lsd detection times(ms): " << tictoc1.toc() << endl;

    TicToc tictoc2;
    if (vkeylines.size() > Config::lsdNFeatures() && Config::lsdNFeatures() != 0)
    {
        sort(vkeylines.begin(), vkeylines.end(), [](const cv::line_descriptor::KeyLine& a, const cv::line_descriptor::KeyLine& b)
        {
            return (a.response > b.response);
        });

        vkeylines.resize((size_t)Config::lsdNFeatures());

        for (int i = 0; i < Config::lsdNFeatures(); i++)
        {
            vkeylines[i].class_id = i;
        }

        lbd->compute(img, vkeylines, linedesc);
    }
    else
    {
        lbd->compute(img, vkeylines, linedesc);
    }

    cout << "lbd descriptor times(ms): " << tictoc2.toc() << endl;
}

void LineFeature::matchLineFeatures(const cv::Mat &linedesc1, const cv::Mat &linedesc2, vector<cv::DMatch> &vlinematches12)
{
    cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> bdm =
            cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    bdm->match(linedesc1, linedesc2, vlinematches12 );

}

vector<cv::DMatch> LineFeature::refineMatchesWithDistance(vector<cv::DMatch> &vlinematches12)
{
    vector<cv::DMatch> inliermatches;
    double min_dist=10000, max_dist=0;

    for (int i = 0; i < vlinematches12.size(); i++)
    {
        double dist = vlinematches12[i].distance;

        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    for ( int i = 0; i < vlinematches12.size(); i++ )
    {
        if ( vlinematches12[i].distance <= max ( 2*min_dist, 35.0 ) )
        {
            inliermatches.push_back(vlinematches12[i]);
        }
    }

    return inliermatches;
}

vector<cv::DMatch> LineFeature::refineMatchesWithKnn(vector<vector<cv::DMatch>> &vlinematches12)
{
    double nn12distth;

    nn12distth = 45;

    sort( vlinematches12.begin(), vlinematches12.end(), sort_descriptor_by_queryIdx() );

    vector<cv::DMatch> vmatches;

    for (int i = 0; i < vlinematches12.size(); i++)
    {
        double dist12 = vlinematches12[i][1].distance - vlinematches12[i][0].distance;

        if (dist12 > nn12distth)
            vmatches.push_back(vlinematches12[i][0]);
    }

    return vmatches;
}

vector<cv::DMatch> LineFeature::refineMatchesWithFundamental(const vector<cv::line_descriptor::KeyLine> &vqueryKeylines,
                                                             const vector<cv::line_descriptor::KeyLine> &vtrainKeylines,
                                                             const vector<cv::DMatch> &vmathes)
{

    vector<cv::Point2f> vqueryPts;
    vector<cv::Point2f> vtrainPts;
    vector<uchar> vinliersMask(vmathes.size());

    for (size_t i = 0; i < vmathes.size(); i++)
    {
        vqueryPts.emplace_back(cv::Point2f(vqueryKeylines[vmathes[i].queryIdx].startPointX,
                                        vqueryKeylines[vmathes[i].queryIdx].startPointY));
        vtrainPts.emplace_back(cv::Point2f(vtrainKeylines[vmathes[i].trainIdx].startPointX,
                                        vtrainKeylines[vmathes[i].trainIdx].startPointY));
    }

    cv::findFundamentalMat(vtrainPts, vqueryPts, cv::FM_RANSAC, 5.0, 0.99, vinliersMask);

    vector<cv::DMatch> vinliersMatch;

    for (size_t i = 0; i < vinliersMask.size(); i++)
    {
        if (vinliersMask[i])
            vinliersMatch.push_back(vmathes[i]);
    }

    return vinliersMatch;
}

} // namespace PL_VO
