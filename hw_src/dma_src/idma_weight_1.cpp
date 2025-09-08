#include "bnn-library.h"

// includes for network parameters
#include "dma.h"
#include "streamtools.h"

#define DataWidth 8*32*4
#define NumBytes numWords*(DataWidth/8)
#define numWords 144

void idma_weight_1(ap_uint<DataWidth> *in0_V, hls::stream<ap_uint<DataWidth> > &out_V, ap_uint<8> actualWords)
{

#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=in0_V
#pragma HLS INTERFACE s_axilite port=in0_V bundle=control
#pragma HLS INTERFACE s_axilite port=actualWords bundle=control
#pragma HLS INTERFACE axis port=out_V
#pragma HLS DATAFLOW

	Mem2Stream_test<DataWidth, NumBytes>(in0_V, out_V, actualWords);
}
