#include "cnn.h"

void cnn_general(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int width, int height, int layers) {
#pragma HLS DATAFLOW
	assert(width <= 416);
	assert(height <= 416);
	assert(layers <= 3);
	hls::stream<decimal_t> conv_out("full_layer_conv_out");
	hls::stream<decimal_t> batch_out("full_layer_batch_out");
	hls::stream<decimal_t> leaky_out("full_layer_leaky_out");

	hls::stream<decimal_t> weights_only("full_layer_weights_only");
	decimal_t scale;
	decimal_t add;
	extract_scale_add<decimal_t>(weights, weights_only, scale, add, 3*3*layers);

	conv2d_use_class(in, conv_out, weights_only, width, height, layers);

	batch_norm<decimal_t>(conv_out, batch_out, scale, add, width, height);
	leaky_relu<decimal_t>(batch_out, leaky_out, width, height);
	max_pool<decimal_t, 416>(leaky_out, out, width, height);
}

void conv2d_use_class(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int width, int height, int layers) {
	ConvClass<decimal_t, 128, 1110> c_impl;
	c_impl.set_width(width);
	c_impl.set_height(height);
	c_impl.set_layers(layers);
	c_impl.load_weights(weights);
	c_impl.convolute(in, out);
}

void stream_weights(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int in_layers, int out_layers) {
	StreamWeights<decimal_t, 11, 1024> c_impl;
	c_impl.set_in_layers(in_layers);
	c_impl.set_out_layers(out_layers);
	c_impl.load_input(in);
	c_impl.run(weights, out);
}
