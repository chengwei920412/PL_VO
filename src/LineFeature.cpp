//
// Created by rain on 17-12-21.
//

#include "LineFeature.h"
#include "TicToc.h"

namespace PL_VO
{

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

    cout << "lbd descriptor times(ms): " << tictoc2.toc() << endl;

}

void LineFeature::matchLineFeatures(cv::BFMatcher *bfmatcher, cv::Mat linedesc1, cv::Mat linedesc2,
                                    vector<vector<cv::DMatch>> &linematches12)
{
    bfmatcher->knnMatch(linedesc1, linedesc2, linematches12, 2);
}


}
