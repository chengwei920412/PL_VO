//
// Created by rain on 17-12-22.
//

#include "Config.h"

namespace PL_VO
{

Config::Config()
{
    depth_scale = 5000;
    image_RGBForm = true;
    // flag
    has_points         = true;      // true if using points
    has_lines          = true;      // true if using line segments
    lr_in_parallel     = true;     // true if detecting and matching features in parallel
    pl_in_parallel     = false;     // true if detecting points and line segments in parallel
    best_lr_matches    = true;      // true if double-checking the matches between the two images
    adaptative_fast    = true;      // true if using adaptative fast_threshold
    // ORB
    orb_nfeatures = 300;
    orb_scale_factor = 1.2;
    orb_nlevels      = 4;
    orb_edge_th      = 31;
    orb_wta_k        = 2;
    orb_score        = 1;           // 0 - HARRIS  |  1 - FAST
    orb_patch_size   = 31;
    orb_fast_th      = 20;          // default FAST threshold

    // LSD parameters
    lsd_nfeatures    = 300;         // set to 0 if keeping all lines
    lsd_refine       = 2;
    lsd_scale        = 1.2;
    lsd_sigma_scale  = 0.6;
    lsd_quant        = 2.0;
    lsd_ang_th       = 22.5;
    lsd_log_eps      = 1.0;
    lsd_density_th   = 0.6;
    lsd_n_bins       = 1024;

    min_line_length   = 0.025;       // min. line length (relative to img size)
    line_horiz_th     = 0.1;         // parameter to avoid horizontal lines
    desc_th_l         = 0.1;         // parameter to avoid outliers in line matching
    line_cov_th       = 10.0;        // parameter to remove noisy line segments

    // Optimization parameters
    // -----------------------------------------------------------------------------------------------------
    min_features     = 10;          // min. number of features to perform StVO
    max_iters        = 5;           // max. number of iterations in the first stage of the optimization
    inlier_k         = 2.0;         // factor to discard outliers before the refinement stage
}

Config& Config::getInstance()
{
    static Config instance; // Instantiated on first use and guaranteed to be destroyed
    return instance;
}

}