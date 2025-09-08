#include "odma.hpp"

void odma(hls::stream<ap_uint<DataWidth> > &in0_V, ap_uint<DataWidth> *out_V)
{

#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out_V depth=128
#pragma HLS INTERFACE s_axilite port=out_V bundle=control
#pragma HLS INTERFACE axis port=in0_V depth=128
#pragma HLS DATAFLOW

	Stream2Mem<DataWidth, NumBytes>(in0_V, out_V);
}
