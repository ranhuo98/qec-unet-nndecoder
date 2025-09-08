#ifndef ODMA_HPP
#define ODMA_HPP

#include "bnn-library.h"

// includes for network parameters
#include "dma.h"
#include "streamtools.h"

#define DataWidth 32
#define NumBytes 144
#define precision 4
#define NumWords NumBytes/(DataWidth/8)

void odma(hls::stream<ap_uint<DataWidth> > &in0_V, ap_uint<DataWidth> *out_V);

#endif
