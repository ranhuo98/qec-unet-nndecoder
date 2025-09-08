#ifndef IDMA_HPP
#define IDMA_HPP

#include "bnn-library.h"

// includes for network parameters
#include "dma.h"
#include "streamtools.h"

#define DataWidth 32
#define NumBytes 36
#define precision 4
#define NumWords NumBytes/(DataWidth/8)

void idma(ap_uint<DataWidth> *in0_V, hls::stream<ap_uint<2*precision> > &out_V);

#endif
