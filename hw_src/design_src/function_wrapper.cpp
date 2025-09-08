#include "function_wrapper.hpp"

void configure_top_parameters(
	ap_uint<8> block,
    ap_uint<4> &IFMDim_arg,
    ap_uint<4> &OFMDim_arg,
    ap_uint<8> &IFMChannel_arg,
    ap_uint<8> &MVAU_OFMChannel_arg,
	ap_uint<8> &MVAU_tmem_arg,
    ap_uint<8> &weight_in_pe_arg,
    ap_uint<8> &MVAU_Tiles_arg,
    ap_uint<8> &UpS_Tiles_arg,
    ap_uint<8> &OUPChannel_arg,
    ap_uint<2> &nf_compute,
    ap_uint<8> &scale_factor_arg,
    ap_uint<8> &Padding_arg,
    ap_uint<1> &MaxPooling_en,
    ap_uint<1> &Upsampling_en,
    ap_uint<2> &buf_index
) {
    // Arrays to hold the values for each parameter
    const ap_uint<4> IFMDim_values[10] = {6, 6, 3, 3, 1, 1, 3, 3, 6, 6};
    const ap_uint<4> OFMDim_values[10] = {6, 3, 3, 1, 1, 3, 3, 6, 6, 6};
    const ap_uint<8> IFMChannel_values[10] = {2, 16, 16, 32, 32, 64, 32, 16, 16, 8};
    const ap_uint<8> MVAU_OFMChannel_values[10] = {16, 16, 32, 32, 64, 64, 16, 16, 8, 8};
    const ap_uint<8> MVAU_tmem_values[10] = {1, 1, 1, 1, 2, 2, 1, 1, 1, 1};
    const ap_uint<8> weight_in_pe_values[10] = {16, 16, 32, 32, 32, 32, 16, 16, 8, 8};
    const ap_uint<8> MVAU_Tiles_values[10] = {9, 9, 9, 9, 18, 18, 9, 9, 9, 9};
    const ap_uint<8> UpS_Tiles_values[10] = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0};
    const ap_uint<8> OUPChannel_values[10] = {0, 0, 0, 0, 0, 32, 0, 16, 0, 0};
    const ap_uint<2> nf_compute_values[10] = {1, 1, 1, 1, 2, 2, 1, 1, 1, 1};
    const ap_uint<8> scale_factor_values[10] = {0, 0, 0, 0, 0, 3, 0, 2, 0, 0};
    const ap_uint<8> Padding_values[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const ap_uint<1> MaxPooling_en_values[10] = {0, 1, 0, 1, 0, 0, 0, 0, 0, 0};
    const ap_uint<1> Upsampling_en_values[10] = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0};
    const ap_uint<2> buf_index_values[10] = {0, 0, 0, 1, 0, 1, 0, 0, 0, 0};

    // Set the values based on the block number
    IFMDim_arg = IFMDim_values[block - 1];
    OFMDim_arg = OFMDim_values[block - 1];
    IFMChannel_arg = IFMChannel_values[block - 1];
    MVAU_OFMChannel_arg = MVAU_OFMChannel_values[block - 1];
    MVAU_tmem_arg = MVAU_tmem_values[block - 1];
    weight_in_pe_arg = weight_in_pe_values[block - 1];
    MVAU_Tiles_arg = MVAU_Tiles_values[block - 1];
    UpS_Tiles_arg = UpS_Tiles_values[block - 1];
    OUPChannel_arg = OUPChannel_values[block - 1];
    nf_compute = nf_compute_values[block - 1];
    scale_factor_arg = scale_factor_values[block - 1];
    Padding_arg = Padding_values[block - 1];
    MaxPooling_en = MaxPooling_en_values[block - 1];
    Upsampling_en = Upsampling_en_values[block - 1];
    buf_index = buf_index_values[block - 1];
}


void FMPadding_Batch_edge_6to8(hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
		hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
		ap_uint<4> outputdim,
		ap_uint<3> paddingleft,
		ap_uint<3> paddingright,
		ap_uint<3> paddingtop,
		ap_uint<3> paddingbottom)
{
	FMPadding_Batch_test<ImgDim, OutputDim, PaddingBefore, PaddingBehind, Input_NumChannels, SIMD, ap_int<Input_precision>> (in0_V, out_V, outputdim, paddingleft, paddingright, paddingtop, paddingbottom, Padding_numReps);
}


void ConvolutionInputGenerator_3by3(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                hls::stream<ap_uint<SIMD*Input_precision>> &out_V,
				ap_uint<4> convkerneldim,
				ap_uint<4> stride,
				ap_uint<4> ifmdim,
				ap_uint<4> ofmdim)
{
	ConvolutionInputGenerator_test<ConvKernelDim, IFMChannels, Input_precision, IFMDim, OFMDim, SIMD, Stride> (in0_V, out_V, convkerneldim, stride, ifmdim, ofmdim, Conv_numReps, ap_resource_lutram());
}


void ConvolutionInputGenerator_for_maxpool(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                hls::stream<ap_uint<SIMD*Input_precision>> &out_V,
				ap_uint<4> convkerneldim,
				ap_uint<4> stride,
				ap_uint<4> ifmdim,
				ap_uint<4> ofmdim)
{
	ConvolutionInputGenerator_test<ConvKernelDim_mp, IFMChannels, Input_precision, IFMDim, OFMDim, SIMD, Stride_mp> (in0_V, out_V, convkerneldim, stride, ifmdim, ofmdim, Conv_numReps, ap_resource_lutram());
}




void MatrixVectorActivation_9by64(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                    hls::stream<ap_uint<PE*Input_precision>> &out_V,
					FixedPointWeights<SIMD,ap_int<Input_precision>,PE,TILES> &weights,
					ThresholdsActivation<TMEM,PE,NUM_THRES,ap_int<BW_THRESHOLDS>,ap_uint<Input_precision>,0,comp::less_equal<ap_int<BW_THRESHOLDS>, ap_int<BW_THRESHOLDS>>> &threshs,
					ap_uint<8> const simd_compute,
				    ap_uint<8> const pe_compute,
				    int const  reps_compute,
					unsigned const nf_compute,
					unsigned const sf_compute
				)
{
	Matrix_Vector_Activate_Batch_test<MW, MH, SIMD, PE, 1, MVA_NUMREPS, Slice<ap_uint<Input_precision>>, Slice<ap_uint<Input_precision>>, Identity>
					(in0_V, out_V, weights, threshs, pe_compute, simd_compute, reps_compute, nf_compute, sf_compute, ap_resource_lut());
}



void MatrixVectorActivation_out(hls::stream<ap_uint<SIMD*Input_precision>> &in0_V,
                    hls::stream<ap_uint<4*32>> &out_V,
					const FixedPointWeights<8,ap_int<Input_precision>,4,1> &weights,
					ap_uint<8> const simd_compute,
				    ap_uint<8> const pe_compute,
				    int const  reps_compute,
					unsigned const nf_compute,
					unsigned const sf_compute)
{
	Matrix_Vector_Activate_Batch_test<MW_preUpS, 4, 64, 4, 1, 36, Slice<ap_uint<Input_precision>>, Slice<ap_int<32>>, Identity>
					(in0_V, out_V, weights, PassThroughActivation<ap_int<32>>(), pe_compute, simd_compute, reps_compute, nf_compute, sf_compute, ap_resource_dflt());
}


void DataWidthConverter_PEtoSIMD(
		hls::stream<ap_uint<PE*Input_precision> > &in0_V,
		hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
		unsigned int numinwords,
		unsigned int nf_compute
		)
{
	StreamingDataWidthConverter_Batch_test<InWidth, OutWidth, NumInWords>(in0_V, out_V, numinwords, nf_compute, numReps_dwc);
}


void Pool_Batch_1in4(hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			int const  reps_compute)
{
	MaxPoolFunction<ap_uint<Input_precision>,4> pool_fxn;
	Pool_batch_test<Input_NumChannels, PE, 4, POOL_REPS, Slice<ap_uint<Input_precision>>, Slice<ap_uint<Input_precision>>>
        	(in0_V, out_V, pool_fxn,reps_compute);
}


void UpsampleNearestNeighbour_Batch_3to6(hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			ap_uint<4> scale_factor_argu,
			ap_uint<4> Padding_argu,
			ap_uint<4> ifmdim,
			ap_uint<4> ofmdim)
{
	UpsampleNearestNeighbour_Batch_test<OFMDim_UpS_3to6, IFMDim_UpS_3to6, SIMD,
                	ap_uint<Input_precision> > (in0_V, out_V, scale_factor_argu, Padding_argu, ofmdim, ifmdim, numReps);
}


void Write_StreamToBuf_concat(
			int buf_index,
			hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
			ap_uint<SIMD*Input_precision> (&buf)[2][MAX_BUFSIZE],
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			int const data_size
		)
{
	write_buf_concat<MAX_BUFSIZE, Input_NumChannels, ap_uint<Input_precision>>(buf_index, in0_V, buf, out_V, data_size);
}


void Read_Buf_concatToStream(
			int buf_index,
			ap_uint<SIMD*Input_precision> (&buf)[2][MAX_BUFSIZE],
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			int const data_size
		)
{
	read_buf_concat<MAX_BUFSIZE, Input_NumChannels, ap_uint<Input_precision>>(buf_index, buf, out_V, data_size);
}


void Write_StreamToBuf_reload(
			hls::stream<ap_uint<SIMD*Input_precision> > &in0_V,
			ap_uint<SIMD*Input_precision> (&buf)[MAX_BUFSIZE],
			int const data_size
		)
{
	write_buf_reload<MAX_BUFSIZE, Input_NumChannels, ap_uint<Input_precision>>(in0_V, buf, data_size);
}


void Read_Buf_reloadToStream(
			ap_uint<SIMD*Input_precision> (&buf)[MAX_BUFSIZE],
			hls::stream<ap_uint<SIMD*Input_precision> > &out_V,
			int const data_size
		)
{
	read_buf_reload<MAX_BUFSIZE, Input_NumChannels, ap_uint<Input_precision>>(buf, out_V, data_size);
}


void AddStreams_Batch_01(
			hls::stream<ap_uint<Thresholding_PE*Input_precision>> &in0_V,
			hls::stream<ap_uint<Thresholding_PE*Input_precision>> &in1_V,
			hls::stream<ap_uint<SIMD*(Input_precision+1)>> &out_V,
			ap_uint<32> const data_compute, // numTotal
			ap_uint<8> const simd_compute
		)
{
	AddStreams_Batch_test<SIMD, ap_int<Input_precision>, ap_int<Input_precision>, ap_int<Input_precision+1>, NumTotal_Add> (in0_V, in1_V, out_V, data_compute, simd_compute, 1);
}


void Thresholding_Batch_initial(
				hls::stream<ap_uint<2*Input_precision>> &in0_V,
				hls::stream<ap_uint<Thresholding_PE*Input_precision>> &out_V,
				ThresholdsActivation<1,2,15,ap_int<8>,ap_int<Input_precision>,-8,comp::less_equal<ap_int<8>, ap_int<8>>> const &threshs
			)
{
	Thresholding_Batch_test02<36, 2, 2, Slice<ap_int<Input_precision>>, Slice<ap_int<Input_precision>>>
	                (in0_V, out_V, threshs, 1);
}


void LabelSelect(
		hls::stream<ap_uint<4*32>> &in0_V,
		hls::stream<ap_uint<32>> &out_V
	)
{
	LabelSelect_Batch_test<TopK_class, TopK_PE, K, ap_int<32>>(in0_V, out_V, TopK_reps);
}

