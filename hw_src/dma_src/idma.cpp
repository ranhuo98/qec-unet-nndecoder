#include "idma.hpp"

void idma(ap_uint<DataWidth> *in0_V, hls::stream<ap_uint<2*precision> > &out_V)
{

#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=in0_V
#pragma HLS INTERFACE s_axilite port=in0_V bundle=control
#pragma HLS INTERFACE axis port=out_V
#pragma HLS DATAFLOW

	hls::stream<ap_uint<DataWidth>> dma2dwc;
#pragma HLS STREAM variable=dma2dwc depth=9
	Mem2Stream<DataWidth, NumBytes>(in0_V, dma2dwc);
	StreamingDataWidthConverter_Batch<DataWidth, 2*precision, NumWords>(dma2dwc, out_V, 1);
}
