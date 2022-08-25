#pragma once

#include <cstdint>

struct stereo_matching_configuration {
    uint32_t height = 1080;
    uint32_t width = 1920;
    uint32_t downscale_factor = 2;
    int32_t min_disparity = 75;
    int32_t max_disparity = 262;
    uint32_t ncc_patch_radius = 1;
    uint32_t sad_patch_radius = 5;
    uint32_t threshold = 5;
};