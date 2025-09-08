#ifndef FUNCTION_TEST_HPP
#define FUNCTION_TEST_HPP

#include "bnn-library.h"


/**
 * \brief   FM Padding - Padds the input with zeroes for when the sliding window is
 *          centered on border pixels
 *
 * Used to add padding with zeroes to multiple inputs in case the sliding window is
 * centered on border pixels - working on non-square images and padding
 *
 * \tparam	OutputDim_x	Padded width of the output feature map
 * \tparam	OutputDim_y	Padded height of the output feature map
 * \tparam	PaddingLeft		Left image padding on x-axis
 * \tparam	PaddingRight	Right image padding on x-axis
 * \tparam	PaddingTop		Top image padding on y-axis
 * \tparam	PaddingBottom	Bottom image padding on y-axis
 * \tparam	NumChannels		Number of channels of the input feature map
 * \tparam	SIMD			Input parallelism
 * \tparam	In_t			Input datatype
 *
 * \param	in		Input stream
 * \param	out		Output stream
 */

template<
	unsigned  OutputDim_x,
	unsigned  OutputDim_y,
	unsigned  PaddingLeft,
	unsigned  PaddingRight,
	unsigned  PaddingTop,
	unsigned  PaddingBottom,
	unsigned  NumChannels,
	unsigned  SIMD,
	typename  In_t
>
void FMPadding_nonsquare_test(
	hls::stream<ap_uint<SIMD*In_t::width>> &in,
	hls::stream<ap_uint<SIMD*In_t::width>> &out,
	ap_uint<4> outputdim,
	ap_uint<3> paddingleft,
	ap_uint<3> paddingright,
	ap_uint<3> paddingtop,
	ap_uint<3> paddingbottom
){
	static_assert(NumChannels%SIMD == 0, "Channel count must be a SIMD multiple.");
	constexpr unsigned  Folding = NumChannels/SIMD;

	for(unsigned  y = 0; y < OutputDim_y; y++) {
		for(unsigned  x = 0; x < OutputDim_x; x++) {
			for(unsigned  sf = 0; sf < Folding; sf++) {
#pragma HLS pipeline style=flp II=1
				ap_uint<SIMD*In_t::width>  outData = 0;

				// Read & forward real data only for non-padding image kernel
				if(
					/* rows */ (paddingtop  <= y) && (y < outputdim - paddingbottom) &&
					/* cols */ (paddingleft <= x) && (x < outputdim - paddingright)
				) {
					outData = in.read();
					out.write(outData);
				}
				else if(
						/* rows */ ((y < paddingtop) && (x < outputdim)) || ((y >= outputdim - paddingbottom) && (y < outputdim) && (x < outputdim)) ||
						/* cols */ ((x < paddingleft) && (y < outputdim)) || ((x >= outputdim - paddingright) && (x < outputdim) && (y < outputdim))
				) {
					out.write(outData);
				}
			}
		}
	}
}

template<
	unsigned  ImgDim,
	unsigned  OutputDim,
	unsigned  PaddingBefore,
	unsigned  PaddingBehind,
	unsigned  NumChannels,
	unsigned  SIMD,
	typename  In_t
>
void FMPadding_test(
	hls::stream<ap_uint<SIMD*In_t::width>> &in,
	hls::stream<ap_uint<SIMD*In_t::width>> &out,
	ap_uint<4> outputdim,
	ap_uint<3> paddingleft,
	ap_uint<3> paddingright,
	ap_uint<3> paddingtop,
	ap_uint<3> paddingbottom
){
#pragma HLS inline
	FMPadding_nonsquare_test<
		OutputDim, OutputDim,
		PaddingBefore, PaddingBehind, PaddingBefore, PaddingBehind,
		NumChannels, SIMD, In_t
	>(in, out, outputdim, paddingleft, paddingright, paddingtop, paddingbottom);
}

template<
	unsigned  ImgDim,
	unsigned  OutputDim,
	unsigned  PaddingBefore,
	unsigned  PaddingBehind,
	unsigned  NumChannels,
	unsigned  SIMD,
	typename  In_t
>
void FMPadding_Batch_test(
	hls::stream<ap_uint<SIMD*In_t::width>> &in,
	hls::stream<ap_uint<SIMD*In_t::width>> &out,
	ap_uint<4> outputdim,
	ap_uint<3> paddingleft,
	ap_uint<3> paddingright,
	ap_uint<3> paddingtop,
	ap_uint<3> paddingbottom,
	unsigned const  numReps
) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		FMPadding_test<ImgDim, OutputDim, PaddingBefore, PaddingBehind, NumChannels, SIMD, In_t>(in, out, outputdim, paddingleft, paddingright, paddingtop, paddingbottom);
	}
}


/**
 * \brief Sliding Window unit that produces output vectors for feeding
 * a Matrix_Vector_Activate_Batch, implementing the im2col algorithm. To be used only if
 * ConvKernelDim%Stride = 0
 *
 * \tparam ConvKernelDim    Dimension of the convolutional kernel (assumed square)
 * \tparam IFMChannels      Number of Input Feature Maps
 * \tparam Input_precision  Number bits per pixel
 * \tparam IFMDim           Width and Heigth of the Input Feature Map (assumed square)
 * \tparam OFMDim           Width and Heigth of the Output Feature Map (assumed square)
 * \tparam SIMD             Number of input columns computed in parallel
 * \tparam Stride           Stride of the convolutional kernel
 * \tparam R          	  Datatype for the resource used for FPGA implementation of the SWG  - safely deducible from the paramaters
 *
 * \param in                Input stream
 * \param out               Output stream
 * \param numReps           Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r			  Resource type for the hardware implementation of the memory block
 */

template<unsigned int ConvKernelDim,
		 unsigned int IFMChannels,
		 unsigned int Input_precision,
		 unsigned int IFMDim,
		 unsigned int OFMDim,
		 unsigned int SIMD,
		 unsigned int Stride,
		 typename R>
void ConvolutionInputGenerator_test(
		hls::stream<ap_uint<SIMD*Input_precision> > & in,
		hls::stream<ap_uint<SIMD*Input_precision> > & out,
		ap_uint<4> convkerneldim,
		ap_uint<4> stride,
		ap_uint<4> ifmdim,
		ap_uint<4> ofmdim,
		const unsigned int numReps,
		R const &r)
{
	static_assert(IFMChannels % SIMD == 0, "");
	static_assert(ConvKernelDim % Stride == 0, "");
	const unsigned int multiplying_factor = IFMChannels/SIMD;
	const unsigned int number_blocks = ConvKernelDim/Stride + 1;
	const unsigned int Stride_BUF = 2;
	ap_uint<SIMD*Input_precision> inputBuf[number_blocks][Stride_BUF * IFMDim * multiplying_factor];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
	memory_resource(inputBuf, r);
	const unsigned int cycles_write_block = (OFMDim * ConvKernelDim * ConvKernelDim * multiplying_factor);
	const unsigned int cycles_read_block = Stride * IFMDim * multiplying_factor;
	const unsigned int max_cycles = std::max(cycles_write_block,cycles_read_block);
	const unsigned int baseIter = IFMDim * ConvKernelDim * multiplying_factor// Initial buffer
								+ OFMDim * std::max(cycles_write_block,cycles_read_block);
	unsigned int counter_internal_block = 0;
	unsigned int current_block_write = 0;
	unsigned int current_line = 0;
	unsigned int read_block = 0;
	unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, count_simd =0;

	// test
	const unsigned int number_blocks_test = convkerneldim/stride + 1 ;
	const unsigned int cycles_write_block_test = (ofmdim * convkerneldim * convkerneldim * multiplying_factor);
	const unsigned int cycles_read_block_test = stride * ifmdim * multiplying_factor;
	const unsigned int max_cycles_test = std::max(cycles_write_block_test,cycles_read_block_test);
	const unsigned int baseIter_test = ifmdim * convkerneldim * multiplying_factor// Initial buffer
									+ ofmdim * max_cycles_test;
	const unsigned int in_line_leftovers = (ifmdim % convkerneldim) % stride;
	const unsigned int in_data_leftovers = in_line_leftovers * ifmdim;

	for (unsigned int count_image = 0; count_image < numReps; count_image++) {
		for (unsigned int i = 0; i < baseIter; i++) {
#pragma HLS pipeline style=flp II=1
			if(i < baseIter_test)
			{
				// For one image input at a time, this "if (inp < IFMDim * ConvKernelDim*multiplying_factor)" will only run once, to initially read data to buffer.
				// Then after this initial buffer, the buffer will be written and read at the same time.
				// The amount of parallel write and read will depend on the cycles_write_block and cycles_read_block.
				// For instance, ConvKernelDim = 3, IFMDim = 8, OFMDim = 6, Stride = 1,
				// there will be 54 cycles to write to stream and 8 cycles to read new data to buffer
				// counter_internal_block will count the cycles
				//if (inp < IFMDim * ConvKernelDim*multiplying_factor) {// Initial buffer of ConvKernelDim lines
				if (inp < ifmdim * convkerneldim * multiplying_factor)
				{// Initial buffer of ConvKernelDim lines
					ap_uint<SIMD*Input_precision> inElem;
					inElem = in.read();
					inputBuf[current_block_write][current_line] = inElem;
					current_line++;
					inp++;
					if (current_line == stride * ifmdim * multiplying_factor ) {
						current_line = 0;
						current_block_write++;
						if (current_block_write == number_blocks_test) {
							current_block_write=0;
						}
						read_block++;
						counter_internal_block = 0;
					}
				} else
				{
					if (counter_internal_block < cycles_write_block_test-1)
					{ // We are writing output, MMV IFMChan per cycle
						unsigned int current_block_read = (current_block_write + 1 + k_y / stride);
						if (current_block_read >= number_blocks_test)
						{
							current_block_read-= number_blocks_test;
						}
						unsigned int current_line_in_block = ((k_y%stride) * ifmdim + ofm_x*stride + k_x)*multiplying_factor + count_simd;
						ap_uint<SIMD*Input_precision> outElem = inputBuf[current_block_read][(current_line_in_block)];
						out.write(outElem);
						count_simd++;
						if (count_simd == multiplying_factor)
						{
							count_simd=0;
							k_x++;
							if (k_x == convkerneldim)
							{
								k_x = 0;
								k_y++;
								if (k_y == convkerneldim)
								{
									k_y = 0;
									ofm_x ++;
									//if (ofm_x == OFMDim) {
									if (ofm_x == ofmdim)
									{
										ofm_x = 0;
										ofm_y++;
										//if (ofm_y == OFMDim) {
										if (ofm_y == ofmdim)
										{
											ofm_y = 0;
											inp = 0;
										} // if (ofm_y == OFMDim)
									} // if (ofm_x == OFMDim)
								} // if (k_y == ConvKernelDim)
							} // if (k_x == ConvKernelDim)
						} // if (count_simd == multiplying_factor)
					} // if (counter_internal_block < cycles_write_block-1)


					// read_block counts how many rows have been read to the buf (the "row" in a IFMDim * IFMDim matrix)
					//if ((counter_internal_block < cycles_read_block-1) && (read_block<IFMDim/Stride)) { // In parallel we write in the buffer, in the current block write if we still need to
					if ((counter_internal_block < cycles_read_block_test-1) && (read_block<ifmdim/stride))
					{ // In parallel we write in the buffer, in the current block write if we still need to
						ap_uint<SIMD*Input_precision> inElem;
						inElem = in.read();
						inputBuf[current_block_write][current_line] = inElem;
#pragma AP dependence variable=inputBuf intra false
#pragma AP dependence variable=inputBuf inter false
						current_line++;
						if (current_line == stride * ifmdim * multiplying_factor)
						{// We read the whole block, we change the next block in which we want to we
							// We filled up a block, let's not read until
							current_line = 0;
							read_block++;
							current_block_write++;
							if (current_block_write == number_blocks_test)
							{
								current_block_write=0;
							}
#pragma AP dependence variable=current_block_write intra false
						} // if (current_line == Stride * IFMDim * multiplying_factor)
					} // if ((counter_internal_block < cycles_read_block-1) && (read_block<IFMDim/Stride))
					counter_internal_block++; // = (counter_internal_block +1) % max_cycles;
					if (counter_internal_block == (max_cycles_test-1))
					{
						counter_internal_block = 0;
					}
				} // else
			} // baseIter_test
			else
			{
				break;
			}
		} // End base_iter
		read_block = 0;
	} // End count_image

	const int leftovers = 3;
	for (unsigned int i = 0; i < leftovers; i++)
	{
		if(in_line_leftovers && i < in_data_leftovers){
			ap_uint<SIMD*Input_precision> consumer;
			consumer = in.read();
		}
		else
			break;
	}
} // End generator



/**
 * \brief Matrix vector activate function
 *
 * The function performs the multiplication between a weigth matrix and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result.
 *
 *
 * \tparam MatrixW    Width of the input matrix
 * \tparam MatrixH    Heigth of the input matrix
 * \tparam SIMD       Number of input columns computed in parallel
 * \tparam PE         Number of output rows computed in parallel
 * \tparam MMV        Number of output pixels computed in parallel
 * \tparam TSrcI      DataType of the input activation (as used in the MAC)
 * \tparam TDstI      DataType of the output activation (as generated by the activation)
 * \tparam TWeightI   DataType of the weights and how to access them in the array
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 * \tparam TW         DataType of the weights matrix - safely deducible from the paramaters
 * \tparam TA         DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 * \tparam R          Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param weights     Weights matrix (currently supports BinaryWeights or FixedPointWeights)
 * \param activation  Activation class
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r           Resource type for the hardware implementation of the MAC block
 */

template<
  unsigned MatrixW, unsigned MatrixH, unsigned SIMD, unsigned PE, unsigned MMV, unsigned REPS,
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
void Matrix_Vector_Activate_Batch_test(hls::stream<TI> &in,
				  hls::stream<TO> &out,
				  TW  const &weights,
				  TA  const &activation,
				  ap_uint<8> const pe_compute,
				  ap_uint<8> const simd_compute,
				  int const  reps_compute,
				  unsigned const nf_compute,
				  unsigned const sf_compute,
				  R const &r) {

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;

  // input vector buffers
  TI  inputBuf[SF];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=0


  decltype(activation.init(0,0))  accu[MMV][PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;
  unsigned const TOTAL_FOLD_test = nf_compute * sf_compute;

  // FOR TEST USE
  int count = 0;
  int checkpoint = 0;

//  int loop_count = REPS * TOTAL_FOLD;
  unsigned const loop_count = 6*6*9;


  for(unsigned i = 0; i < loop_count; i++) {
#pragma HLS pipeline style=flp II=1
	  if(i < reps_compute*TOTAL_FOLD_test) {
    TI  inElem;
    // every TOTAL_FOLD loops nf will be 0 again, which means the input vector will be updated every TOTAL_FOLD loops
    // 36 reps to cover all vectors
    if(nf == 0) {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // Threshold Initialisation
    if(sf == 0) {
      for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        for(unsigned mmv = 0; mmv < MMV; mmv++) {
#pragma HLS UNROLL
        	//if(pe < pe_compute){
        		accu[mmv][pe] = activation.init(nf, pe); // zeros
        	//}
        }
      }
    }

    // compute matrix-vector product for each processing element
    auto const &w = weights.weights(tile);
    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      // wgt is std::array object to save "SIMD" 4-bit values
      // For instance, in weights, the first tile of first pe is ap_uint<8>("0x91", 16) with SIMD = 2,
      // TWeightI is ap_int<4> whose range is from -8 to 7, so wgt would be [1, -7]
	  if(pe < pe_compute){
		  auto const wgt = TWeightI()(w[pe]);
		  for (unsigned mmv = 0; mmv < MMV; mmv++){
			// act is Slice<ap_int<4>> object, act(i, mmv) will choose inElem((i+1)*3, i),
			// in which case act(0, mmv) is inElem(3 to 0) and act(1, mmv) is inElem(7 to 4)
			// act(i, mmv) is used in mac block
			auto const  act = TSrcI()(inElem, mmv);
			accu[mmv][pe] = mac<SIMD>(accu[mmv][pe], wgt, act, r, mmv);
		  } // MMV
      }
    } // PE

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == sf_compute) {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      //std::cout << "outElem = " << outElem(61,1,1) << std::endl;
      for (unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        for (unsigned mmv = 0; mmv < MMV; mmv++){
#pragma HLS UNROLL
        	if(pe < pe_compute){
			  // concat "PE" 4-bit comparison results to one 4*PE-bit outElem
			  outElem(pe,mmv,1) = activation.activate(nf, pe, accu[mmv][pe]);
        	}
        }
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
      // load new input vectors to inElem
      if(++nf == nf_compute) {
	    nf   = 0;
	    tile = 0;
      }
    } // if(++sf == SF)
  } // reps_compute
  else{
	  break;
  }
  } // for(unsigned  i = 0; i < REPS * TOTAL_FOLD; i++)
}



/**
 * \brief   Stream Data Width Converter - Converts the width of the input stream in the output stream
 *
 * Used to upscale or downscale a stream, without any loss of data in the procedure.
 * For downscaling (InWidth > OutWidth), InWidth has to be a multiple of OutWidth.
 * For upscaling (InWidth < OutWidth), OutWidth has to be a multiple of InWidth.
 *
 * \tparam     InWidth      Width, in number of bits, of the input stream
 * \tparam     OutWidth     Width, in number of bits, of the output stream
 * \tparam     NumInWords   Number of input words to process
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the function has to be called
 *
 */

template<unsigned int InWidth,
		unsigned int OutWidth,
		unsigned int NumInWords
>
void StreamingDataWidthConverter_Batch_test(
		hls::stream<ap_uint<InWidth> > & in,
		hls::stream<ap_uint<OutWidth> > & out,
		const unsigned int numinwords,
		const unsigned int nf_compute,
		const unsigned int numReps
		)
{
  static_assert((InWidth % OutWidth == 0) || (OutWidth % InWidth == 0), "");

  if (InWidth > OutWidth) {
    // emit multiple output words per input word read
    const unsigned int outPerIn = InWidth / OutWidth;
    const unsigned int totalIters = NumInWords * outPerIn * numReps;
    unsigned int o = 0;
    ap_uint<InWidth> ei = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS pipeline style=flp II=1
      // read new input word if current out count is zero
      if (o == 0) {
        ei = in.read();
	  }
      // pick output word from the rightmost position
      ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
      out.write(eo);
      // shift input to get new output word for next iteration
      ei = ei >> OutWidth;
      // increment written output count
      o++;
      // wraparound indices to recreate the nested loop structure
      if (o == outPerIn) {
        o = 0;
      }
    }
  } else if (InWidth == OutWidth) {
    // straight-through copy
    for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS pipeline style=flp II=1
      ap_uint<InWidth> e = in.read();
      out.write(e);
    }
  } else { // InWidth < OutWidth
    // read multiple input words per output word emitted
    const unsigned int inPerOut = OutWidth / InWidth;
    const unsigned int totalIters = NumInWords * numReps;
    const unsigned int totalIters_test = numinwords * numReps;
    unsigned int i = 0;
    ap_uint<OutWidth> eo = 0;
    if(nf_compute == 1){ // pad zeros at the front
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS pipeline style=flp II=1
        	if(t < totalIters_test){
				// read input and shift into output buffer
				ap_uint<InWidth> ei = in.read();
				eo = eo >> InWidth;
				eo(InWidth - 1, 0) = ei;
				out.write(eo);
        	}
        }
    }
    else if(nf_compute == 2){
    	for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS pipeline style=flp II=1
    		if(t < totalIters_test){
				// read input and shift into output buffer
				ap_uint<InWidth> ei = in.read();
				eo = eo >> InWidth;
				eo(OutWidth - 1, OutWidth - InWidth) = ei;
				// increment read input count
				i++;
				// wraparound logic to recreate nested loop functionality
				if (i == inPerOut) {
					i = 0;
					out.write(eo);
				}
    		}
		}
    }
  }
}


/**
 * \brief Pool_batch function
 *
 * The function performs a generic pool function (defined in pool.hpp) and works in conjuction
 * with a sliding window unit performing im2col on the input data, allowing
 * generic kernel and stride values
 *
 * \tparam Channels   Number of channels in the pool layer
 * \tparam PE         Number of channels in the pool layer computed in parallel
 * \tparam TotalK     Total kernel size of pooling (e.g. 3x3=9)
 * \tparam TSrcI      DataType of the input value (Slice)
 * \tparam TDstI      DataType of the output value (Slice)
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 * \tparam TA         DataType of the function class (e.g. Max, Avg, Sum) - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param function    Function class in the pool (Max, Avg, Sum)
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 */

template<
  unsigned Channels, unsigned PE, unsigned TotalK, unsigned REPS,
  typename TSrcI = Identity,typename TDstI = Identity,
  typename TI, typename TO, typename TA
>
void Pool_batch_test(hls::stream<TI> &in,
                  hls::stream<TO> &out,
                  TA  const &function,
                  int const  reps_compute) {

  //constexpr unsigned  NF = Channels / PE;
  const unsigned  NF = 1;
  constexpr unsigned  SF = TotalK;
  constexpr unsigned  TOTAL_FOLD = NF * SF ;

  decltype(function.init())  accu[PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  sf   = 0;
  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelining the way we want
  for(unsigned  i = 0; i < REPS * TOTAL_FOLD; i++) {
#pragma HLS pipeline style=flp II=1
	  if(i < reps_compute * TOTAL_FOLD){
		TI  pixel_slice; // ap_uint<SIMD*Input_precision>
		pixel_slice = in.read();

		// Threshold Initialisation
		if(sf == 0) {
		  for(unsigned  pe = 0; pe < PE; pe++) {
	#pragma HLS UNROLL
//			  if(pe < pe_compute){
				  accu[pe] = function.init();
//			  }
		  }
		}

		auto const  slice_channels = TSrcI()(pixel_slice,0);
		for(unsigned  pe = 0; pe < PE; pe++) {
	#pragma HLS UNROLL
//			if(pe < pe_compute){
				accu[pe] = function.pool(slice_channels(pe,0), accu[pe]); // comp::max<T, T, T>()(input,accu);
//			}
		}

		// keep track of which folded synapse/neuron we are processing
		if(++sf == SF) {
		  // produce output and clear accumulators
		  auto  outElem = TDstI().template operator()<TO>();
		  for(unsigned  pe = 0; pe < PE; pe++) {
	#pragma HLS UNROLL
//			  if(pe < pe_compute){
				  outElem(pe,0,1) = function.activate(accu[pe]); //
//			  }
		  }
		  out.write(outElem);
		  // next folded neuron or image
		  sf = 0;
		}
	  }
	  else{
		  break;
	  }
  }
}


/**
 * \brief Upsampling with the Nearest Neighbour algorithm. Works with square feature maps
 *
 * \tparam 	OFMDim 		Size of the output feature map
 * \tparam 	IFMDim 		Size of the input feature map
 * \tparam 	NumChannels 	Amount of channels of the input feature map
 * \tparam 	In_t		 	Input datatype
 *
 * \param 	in 				Input stream
 * \param 	out 			Output stream
 */

template<unsigned int OFMDim,
	unsigned int IFMDim,
	unsigned int NumChannels,
	typename In_t>
void UpsampleNearestNeighbour_test(
        hls::stream<ap_uint<NumChannels * In_t::width>> & in,
        hls::stream<ap_uint<NumChannels * In_t::width>> & out,
		ap_uint<4> scale_factor_argu,
		ap_uint<4> Padding_argu,
		ap_uint<4> ofmdim,
		ap_uint<4> ifmdim
) {
  static_assert(OFMDim > IFMDim, "");

  constexpr unsigned int scale_factor = OFMDim/IFMDim;
  constexpr unsigned int Padding = OFMDim % IFMDim;

//  ofmdim: 			3		ofmdim: 			6
//  ifmdim: 			1		ifmdim: 			3
//  scale_factor_test:	3		scale_factor_test:	2
//  Padding_argu:		0		Padding_argu:		0

//  const unsigned int scale_factor_test = ofmdim/ifmdim;
//  const unsigned int Padding_test = ofmdim % ifmdim;

  const unsigned int scale_factor_test = scale_factor_argu;
  const unsigned int Padding_test = 0; //Padding_argu

  // Padding might be asymmetrical
  const unsigned int PaddingDown = Padding_test/2;
  const unsigned int PaddingUp = Padding_test - PaddingDown;
  // Padding might be asymmetrical
  const unsigned int PaddingRight = Padding_test/2;
  const unsigned int PaddingLeft = Padding_test - PaddingRight;

  ap_uint<NumChannels * In_t::width> outData, inData;
  ap_uint<NumChannels * In_t::width> RowBuf[IFMDim];
  int count_row = -PaddingUp; // Counter used to understand whether reading (and buffering) a row or not - Made in order to avoid modulo operations
  for (unsigned int y = 0; y < OFMDim; y++) {
	  for (unsigned int x = 0; x < OFMDim; x++) {
#pragma HLS pipeline style=flp II=1
		if(x < ofmdim && y < ofmdim){
			bool read_row = (y == 0) || count_row == scale_factor_test;
			if ((x < ifmdim) && read_row)
			{
				inData = in.read();
				RowBuf[x] = inData;
			}
			// Padding Cols
			if(x < PaddingLeft){
				outData = RowBuf[0];
			}
			else if (x >= (ofmdim - PaddingRight)){
				outData = RowBuf[ifmdim-1];

			}
			// Padding Rows
			else if(y < PaddingUp || y >= (ofmdim - PaddingDown)){
				outData = RowBuf[(x-PaddingLeft)/scale_factor_test];
			}
			// No Padding
			else{
				outData = RowBuf[(x-PaddingLeft)/scale_factor_test];
			}
			out.write(outData);
	    }
	  }// end for y
	  count_row++;
	  if (count_row > scale_factor_test)
		  count_row =1;
  } // end for x

}


/**
 * \brief Upsampling with the Nearest Neighbour algorithm. Works with square feature maps on multiple images
 *
 * \tparam 	OFMDim 		Size of the output feature map
 * \tparam 	IFMDim 		Size of the input feature map
 * \tparam 	NumChannels 	Amount of channels of the input feature map
 * \tparam 	In_t		 	Input datatype
 *
 * \param 	in 			Input stream
 * \param 	out 			Output stream
 * \param     numReps      Number of time the function has to be repeatedly executed (e.g. number of images)
 */

template<unsigned int OFMDim,
	unsigned int IFMDim,
	unsigned int NumChannels,
	typename In_t>
void UpsampleNearestNeighbour_Batch_test(
        hls::stream<ap_uint<NumChannels * In_t::width>> & in,
        hls::stream<ap_uint<NumChannels * In_t::width>> & out,
		ap_uint<4> scale_factor_argu,
		ap_uint<4> Padding_argu,
		ap_uint<4> ofmdim,
		ap_uint<4> ifmdim,
		unsigned int numReps) {
  for (unsigned int rep = 0; rep < numReps; rep++) {
	  UpsampleNearestNeighbour_test<OFMDim, IFMDim, NumChannels, In_t>(in, out, scale_factor_argu, Padding_argu, ofmdim, ifmdim);
  }
}


/**
 * \brief Stream to buf_concat
 */
template<unsigned int MAX_BUFSIZE,
	unsigned int NumChannels,
	typename In_t>
void write_buf_concat(
		int buf_index,
		hls::stream<ap_uint<NumChannels * In_t::width>> & in,
		ap_uint<NumChannels * In_t::width> (&buf)[2][MAX_BUFSIZE],
		hls::stream<ap_uint<NumChannels * In_t::width>> & out,
		int const data_size)
{
	for (int i = 0; i < MAX_BUFSIZE; i++) {
#pragma HLS pipeline style=flp II=1
		if(i < data_size){
			ap_uint<NumChannels * In_t::width> temp = in.read();
			buf[buf_index][i] = temp;
			out.write(buf[buf_index][i]);
		}
	}
}


/**
 * \brief buf_concat to stream
 */
template<unsigned int MAX_BUFSIZE,
	unsigned int NumChannels,
	typename In_t>
void read_buf_concat(
		const int buf_index,
		ap_uint<NumChannels * In_t::width> (&buf)[2][MAX_BUFSIZE],
		hls::stream<ap_uint<NumChannels * In_t::width>> & out,
		int const data_size)
{
	for (int i = 0; i < MAX_BUFSIZE; i++) {
#pragma HLS pipeline style=flp II=1
		if(i < data_size){
			ap_uint<NumChannels * In_t::width> temp = buf[buf_index][i];
			out.write(temp);
		}
	}
}


/**
 * \brief Stream to buf_reload
 */
template<unsigned int MAX_BUFSIZE,
	unsigned int NumChannels,
	typename In_t>
void write_buf_reload(
		hls::stream<ap_uint<NumChannels * In_t::width>> & in,
		ap_uint<NumChannels * In_t::width> (&buf)[MAX_BUFSIZE],
		int const data_size)
{
	for (int i = 0; i < MAX_BUFSIZE; i++) {
#pragma HLS pipeline style=flp II=1
		if(i < data_size){
			ap_uint<NumChannels * In_t::width> temp = in.read();
			buf[i] = temp;
		}
	}
}


/**
 * \brief buf_reload to stream
 */
template<unsigned int MAX_BUFSIZE,
	unsigned int NumChannels,
	typename In_t>
void read_buf_reload(
		ap_uint<NumChannels * In_t::width> (&buf)[MAX_BUFSIZE],
		hls::stream<ap_uint<NumChannels * In_t::width>> & out,
		int const data_size)
{
	for (int i = 0; i < MAX_BUFSIZE; i++) {
#pragma HLS pipeline style=flp II=1
		if(i < data_size){
			ap_uint<NumChannels * In_t::width> temp = buf[i];
			out.write(temp);
		}
	}
}


/*!
 * \brief Thresholding function for multiple images
 *
 * The function performs thresholds comparison with input activation vector,
 * and generating output based on the comparison results
 *
 * \tparam ImgDim         Total spatial size of input feature map
 * \tparam NumChannels    Number of channels in input feature map
 * \tparam PE             Number of output rows computed in parallel
 * \tparam TSrcI          DataType of the input activation (as used in the MAC)
 * \tparam TDstI          DataType of the output activation (as generated by the activation)
 * \tparam TI             DataType of the input stream - safely deducible from the paramaters
 * \tparam TO             DataType of the output stream - safely deducible from the paramaters
 * \tparam TA             DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param activation      Activation class
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */

template <
    unsigned ImgDim, unsigned NumChannels, unsigned PE,
    typename TSrcI = Identity, typename TDstI = Identity,
    typename TI, typename TO, typename TA>
void Thresholding_Batch_test(
			hls::stream<TI> &in,
			hls::stream<TO> &out,
			TA const &activation,
			ap_uint<8> const pe_compute,
			int const reps,
			unsigned const imgdim_compute,
			unsigned const nf_compute
		)
{

	// how many different rows each neuron will compute
	// alternatively: number of vertical matrix chunks
	constexpr unsigned  NF = NumChannels / PE;

	// everything merged into a common iteration space (one "big" loop instead
	// of smaller nested loops) to get the pipelinening the way we want
	unsigned nf = 0;

	if(imgdim_compute == 9){ // imgdim = 9, pe = 32
		for (unsigned i = 0; i < reps * 9 * NF; i++) {
#pragma HLS pipeline style=flp II=1
			TI const  inElem = in.read();
			auto outElem = TDstI().template operator()<TO>();
			for (unsigned pe = 0; pe < 32; pe++)
			{
#pragma HLS UNROLL
				auto const act = TSrcI()(inElem);
				outElem(pe,0,1) = activation.activate(nf, pe, act(pe,0));
			}
			out.write(outElem);
			if (++nf == NF)
			{
			  nf = 0;
			}
		}
	}
	else if(imgdim_compute == 36){ // imgdim = 36, pe = 16
		for (unsigned i = 0; i < reps * 36 * NF; i++) {
#pragma HLS pipeline style=flp II=1
			TI const  inElem = in.read();
			auto outElem = TDstI().template operator()<TO>();
			for (unsigned pe = 0; pe < 16; pe++)
			{
#pragma HLS UNROLL
				auto const act = TSrcI()(inElem);
				outElem(pe,0,1) = activation.activate(nf, pe, act(pe,0));
			}
			out.write(outElem);
			if (++nf == NF)
			{
			  nf = 0;
			}
		}
	}
}


template <
    unsigned ImgDim, unsigned NumChannels, unsigned PE,
    typename TSrcI = Identity, typename TDstI = Identity,
    typename TI, typename TO, typename TA>
void Thresholding_Batch_test02(hls::stream<TI> &in,
                        hls::stream<TO> &out,
                        TA const &activation,
                        int const reps)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  constexpr unsigned  NF = 1;

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned nf = 0;
  for (unsigned i = 0; i < reps * ImgDim * NF; i++) {
#pragma HLS pipeline style=flp II=1

    TI const  inElem = in.read();
    auto outElem = TDstI().template operator()<TO>();
    for (unsigned pe = 0; pe < PE; pe++)
    {
#pragma HLS UNROLL
      auto const act = TSrcI()(inElem);
      outElem(pe,0,1) = activation.activate(nf, pe, act(pe,0));
    }
    out.write(outElem);
    if (++nf == NF)
    {
      nf = 0;
    }
  }
}


/**
 * \brief   Element-Wise Addition - Reads in data elements from two streams and writes the sum of these elements to an output
 *
 * \tparam     NumChannels  Amount of channels of the streams
 * \tparam     In1_t        First operand datatype
 * \tparam     In2_t        Second operand datatype
 * \tparam     Out_t        Datatype of the accumulation output
 * \tparam     NumTotal     Total number of words in the input streams
 * \tparam     offset       Offset value for the accumulation
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 *
 */

template <unsigned int NumChannels,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal,
          int offset = 0>
void AddStreams_test(
		hls::stream<ap_uint<NumChannels * In1_t::width>> &in1,
		hls::stream<ap_uint<NumChannels * In2_t::width>> &in2,
		hls::stream<ap_uint<NumChannels * Out_t::width>> &out,
		ap_uint<32> const numTotal,
		ap_uint<8> const numChannels
	)
{

	if(numTotal == 9){ // numTotal = 288, numChannels = 32
	  for (unsigned int i = 0; i < 9; i++) {
#pragma HLS pipeline style=flp II=1
		ap_uint<NumChannels * In1_t::width> e1 = in1.read();
		ap_uint<NumChannels * In2_t::width> e2 = in2.read();
		ap_uint<NumChannels * Out_t::width> e;
		for (unsigned int j = 0; j < 32; j++) {
#pragma HLS UNROLL
		  In1_t op1 = e1((j + 1) * In1_t::width - 1, j * In1_t::width);
		  In2_t op2 = e2((j + 1) * In2_t::width - 1, j * In2_t::width);
		  Out_t sum = op1 + op2 + offset;
		  e((j + 1) * Out_t::width - 1, j * Out_t::width) = sum;
		}
		out.write(e);
	  }
	}
	else if(numTotal == 36){ // numTotal = 576, numChannels = 16
		for (unsigned int i = 0; i < 36; i++) {
#pragma HLS pipeline style=flp II=1
			ap_uint<NumChannels * In1_t::width> e1 = in1.read();
			ap_uint<NumChannels * In2_t::width> e2 = in2.read();
			ap_uint<NumChannels * Out_t::width> e;
			for (unsigned int j = 0; j < 16; j++) {
#pragma HLS UNROLL
			  In1_t op1 = e1((j + 1) * In1_t::width - 1, j * In1_t::width);
			  In2_t op2 = e2((j + 1) * In2_t::width - 1, j * In2_t::width);
			  Out_t sum = op1 + op2 + offset;
			  e((j + 1) * Out_t::width - 1, j * Out_t::width) = sum;
			}
			out.write(e);
		}
	}
}


/**
 * \brief
 *
 * Used to implement point-wise addition in Resnet-50 for multiple images
 *
 * \tparam     NumChannels  Amount of channels of the streams
 * \tparam     In1_t        First operand datatype
 * \tparam     In2_t        Second operand datatype
 * \tparam     Out_t        Datatype of the accumulation output
 * \tparam     NumTotal     Total number of words in the input streams
 * \tparam     offset       Offset value for the accumulation
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 * \param      numReps      Number of frames / images
 *
 */

template <unsigned int NumChannels,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal,
          int offset = 0>
void AddStreams_Batch_test(
			hls::stream<ap_uint<NumChannels * In1_t::width>> &in1,
			hls::stream<ap_uint<NumChannels * In2_t::width>> &in2,
			hls::stream<ap_uint<NumChannels * Out_t::width>> &out,
			ap_uint<32> const numTotal,
			ap_uint<8> const numChannels,
			const unsigned int numReps
		)
{
	for (unsigned int image = 0; image < numReps; image++) {
		AddStreams_test<NumChannels, In1_t, In2_t, Out_t, NumTotal, offset>(in1, in2, out, numTotal, numChannels);
	}
}


/**
 * \brief   LabelSelect_Batch - returns labels of top-NumTop in stream
 *
 * \tparam NumClasses   Number of classes of the dataset
 * \tparam PECount      Number of inputs to be processed in parallel
 * \tparam NumTop       Number of top classes to be selected in output
 * \tparam In_T         Datatype of the input
 * \tparam Out_T        Datatype of the output
 *
 * \param in            Input stream
 * \param out           Output stream
 * \param numReps       Number of times the function has to be repeatedly executed (e.g. number of images)
 *
 */

template<
	// tensor size parameters
	unsigned int NumClasses,
	unsigned int PECount,
	unsigned int NumTop,
	typename In_T,
	typename Out_T
>
void LabelSelect_Batch_test(hls::stream<ap_uint<PECount * In_T::width> > & in,
		hls::stream<Out_T> & out, const unsigned int numReps)
{

	// Check that classes, aka. labels / indeces, can be encoded as non-negative outputs
	static_assert(clog2(NumClasses) <= Out_T::width - Out_T::sign_flag, "");
	static In_T const  In_T_MIN_VAL = (In_T(-1)<0)? 1<<(In_T::width-1) : 0;

	// Array of encountered top values
	//  - maintains topval[i] <= topval[i+1]
	//  - keeps in alignment with toplabels
	In_T topval[NumTop];
#pragma HLS ARRAY_PARTITION variable=topval complete dim=1
	Out_T toplabels[NumTop];
#pragma HLS ARRAY_PARTITION variable=toplabels complete dim=1

	for(unsigned int reps=0; reps<numReps; reps++){
		unsigned int idx = 0;
		for(unsigned int topx=0; topx<NumTop; topx++){
#pragma HLS UNROLL
			topval   [topx] = In_T_MIN_VAL;
			toplabels[topx] = 0;
		}
		for(unsigned int block=0; block<(NumClasses/PECount); block++){
#pragma HLS pipeline style=flp II=1
			ap_uint<PECount * In_T::width> const  inval = in.read();
			for(unsigned int elem=0; elem<PECount; elem++){
#pragma HLS UNROLL
				// Extract individual input
				unsigned const  lowBit = elem * In_T::width;
				unsigned const  highBit = (elem+1) * In_T::width - 1;
				In_T const  val = inval(highBit,lowBit);

				// Compare input against all current tops
				bool  cmp[NumTop+1];
				for(unsigned  i = 0; i < NumTop; i++) {
#pragma HLS UNROLL
					cmp[i] = val > topval[i];
				}
				cmp[NumTop] = false;

				// Shift input into top array at the highest index where it is greater
				for(unsigned  i = 0; i < NumTop; i++) {
#pragma HLS UNROLL
					if(cmp[i]) {
						if(cmp[i+1]) {
							// Shift
							topval   [i] = topval   [i+1];
							toplabels[i] = toplabels[i+1];
						}
						else {
							// Insert
							topval   [i] = val;
							toplabels[i] = idx;
						}
					}
				}
				idx++;
			}
		}

		// Output - index of highest value first
		for(unsigned int topx = 0; topx < NumTop; topx++){
			out.write(toplabels[NumTop - topx - 1]);
		}
	}
}



#endif
