//
// Created by rain on 17-12-22.
//

#ifndef PL_VO_CONFIG_H
#define PL_VO_CONFIG_H

namespace PL_VO
{

class Config
{

public:

    Config();

    static Config& getInstance();

    static int& depthscale() {return getInstance().depth_scale;}

    // flag
    static bool&    hasPoints()          { return getInstance().has_points; }
    static bool&    hasLines()           { return getInstance().has_lines; }
    static bool&    lrInParallel()       { return getInstance().lr_in_parallel; }
    static bool&    plInParallel()       { return getInstance().pl_in_parallel; }

    // ORB
    static int&     orbNFeatures()       { return getInstance().orb_nfeatures; }

    // lines detection and matching
    static int&     lsdNFeatures()       { return getInstance().lsd_nfeatures; }
    static int&     lsdRefine()          { return getInstance().lsd_refine; }
    static double&  lsdScale()           { return getInstance().lsd_scale; }
    static double&  lsdSigmaScale()      { return getInstance().lsd_sigma_scale; }
    static double&  lsdQuant()           { return getInstance().lsd_quant; }
    static double&  lsdAngTh()           { return getInstance().lsd_ang_th; }
    static double&  lsdLogEps()          { return getInstance().lsd_log_eps; }
    static double&  lsdDensityTh()       { return getInstance().lsd_density_th; }
    static int&     lsdNBins()           { return getInstance().lsd_n_bins; }
    static double&  lineHorizTh()        { return getInstance().line_horiz_th; }
    static double&  minLineLength()      { return getInstance().min_line_length; }
    static double&  descThL()            { return getInstance().desc_th_l; }
    static double&  minRatio12L()        { return getInstance().min_ratio_12_l; }
    static double&  lineCovTh()          { return getInstance().line_cov_th; }

private:

    int depth_scale;

    // flags
    bool has_points;
    bool has_lines;
    bool lr_in_parallel;
    bool pl_in_parallel;
    bool best_lr_matches;
    bool adaptative_fast;

    int orb_nfeatures;

    // lines detection and matching
    int    lsd_nfeatures;
    int    lsd_refine;
    double lsd_scale;
    double lsd_sigma_scale;
    double lsd_quant;
    double lsd_ang_th;
    double lsd_log_eps;
    double lsd_density_th;
    int    lsd_n_bins;
    double line_horiz_th;
    double min_line_length;
    double desc_th_l;
    double min_ratio_12_l;
    double line_cov_th;


}; // class Config

} // namesapce PL_VO

#endif //PL_VO_CONFIG_H
