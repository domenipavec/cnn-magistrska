#include "cnn.h"

void cnn_general(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, int size, int in_layers, int out_layers, bool stream_weights, int max_type) {
#pragma HLS DATAFLOW
	// general asserts
	assert(size > 0);
	assert(size <= 416);
	assert(in_layers > 0);
	assert(in_layers <= 1024);
	assert(out_layers > 0);
	assert(out_layers <= 1024);
	assert(max_type >= 0);
	assert(max_type <= 2);

	// test asserts
//	assert(size <= 13);
//	assert(in_layers <= 1024);
//	assert(out_layers <= 1024);
//	assert(stream_weights == 1);
//	assert(max_type == 0);

	hls::stream<decimal_t> conv_out("conv_out");
#pragma HLS STREAM variable=conv_out depth=1 dim=1
	hls::stream<decimal_t> batch_out("batch_out");
#pragma HLS STREAM variable=batch_out depth=1 dim=1
	hls::stream<decimal_t> leaky_out("leaky_out");
#pragma HLS STREAM variable=leaky_out depth=1 dim=1

	hls::stream<decimal_t> weights_data("weights_data");
#pragma HLS STREAM variable=weights_data depth=1 dim=1
	hls::stream<decimal_t> scale_add("scale_add");
#pragma HLS STREAM variable=scale_add depth=1 dim=1
	hls::stream<decimal_t> weights("weights");
#pragma HLS STREAM variable=weights depth=1 dim=1
	hls::stream<decimal_t> data("data");
#pragma HLS STREAM variable=data depth=1 dim=1

	int size2 = size*size;
	int out_size = size2*out_layers;
	int in_size = size2*in_layers;
	int weights_size = 3*3*in_layers*out_layers;

	split<decimal_t>(in, scale_add, weights_data, 2*out_layers, weights_size + in_size);

	if (stream_weights) {
		split<decimal_t>(weights_data, data, weights, in_size, weights_size);
		conv2d_stream_weights(data, conv_out, weights, in_layers, out_layers);
		batch_norm_per_layer<decimal_t, 1024>(conv_out, batch_out, scale_add, size, out_layers);
	} else {
		split<decimal_t>(weights_data, weights, data, weights_size, in_size);
		conv2d_use_class(data, conv_out, weights, size, in_layers, out_layers);
		batch_norm<decimal_t, 1024>(conv_out, batch_out, scale_add, size, out_layers);
	}

	leaky_relu<decimal_t>(batch_out, leaky_out, out_size);

	if (max_type == 0) {
		direct<decimal_t>(leaky_out, out, out_size);
	} else if (max_type == 1) {
		// TODO: change this max size
		max_pool_1<decimal_t, 416>(leaky_out, out, size, out_layers);
	} else {
		max_pool<decimal_t, 3328>(leaky_out, out, size, out_layers);
	}
}

void conv2d_use_class(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int size, int in_layers, int out_layers) {
	ConvClass<decimal_t, 128, 64, 1110> c_impl;
	c_impl.set_size(size);
	c_impl.set_in_layers(in_layers);
	c_impl.set_out_layers(out_layers);
	c_impl.load_weights(weights);
	c_impl.convolute(in, out);
}

void conv2d_stream_weights(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int in_layers, int out_layers) {
	StreamWeights<decimal_t, 11, 1024> c_impl;
	c_impl.set_in_layers(in_layers);
	c_impl.set_out_layers(out_layers);
	c_impl.load_input(in);
	c_impl.run(weights, out);
}
