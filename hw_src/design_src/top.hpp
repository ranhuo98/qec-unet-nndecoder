#ifndef TOP_HPP
#define TOP_HPP

#include "function_wrapper.hpp"


#include "MVAU_params/MVAU_params12.h"
#include "Thresholding_params/Thresholding_thresh0.h"

void top(
		hls::stream<ap_uint<2*Input_precision>> &in_0,
		hls::stream<ap_uint<32>> &out_0,
		hls::stream<ap_uint<WS_TILE*SIMD*Input_precision>> &weight_stream_1,
		hls::stream<ap_uint<WS_TILE*SIMD*Input_precision>> &weight_stream_2,
		hls::stream<ap_uint<BW_THRESHOLDS*NUM_THRES>> &thresh_stream_1,
		hls::stream<ap_uint<BW_THRESHOLDS*NUM_THRES>> &thresh_stream_2,
		ap_uint<8> block,
		ap_uint<8> ee_flag
);



#endif



