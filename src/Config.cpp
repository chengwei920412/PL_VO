//
// Created by rain on 17-12-22.
//

#include "Config.h"

namespace PL_VO
{

Config::Config()
{
    depth_scale = 1000;

    // flag
    has_points         = true;      // true if using points
    has_lines          = true;      // true if using line segments
    lr_in_parallel     = false;      // true if detecting and matching features in parallel
    pl_in_parallel     = false  ;      // true if detecting points and line segments in parallel
    best_lr_matches    = true;      // true if double-checking the matches between the two images
    adaptative_fast    = true;      // true if using adaptative fast_threshold
    // ORB
    orb_nfeatures = 300;

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
}

Config& Config::getInstance()
{
    static Config instance; // Instantiated on first use and guaranteed to be destroyed
    return instance;
}

}