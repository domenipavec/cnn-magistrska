#include "cnn_sim.h"

void cnn_sim(hls::stream<decimal_t> &data, hls::stream<decimal_t> &weights_data, hls::stream<decimal_t> &out, int size, int in_layers, int out_layers, int in_size, int out_size, int weights_size, int scale_add_size, ap_uint<8> control) {

	hls::stream<decimal_t> conv_out("conv_out");
	hls::stream<decimal_t> batch_out("batch_out");
	hls::stream<decimal_t> leaky_out("leaky_out");

	hls::stream<decimal_t> scale_add("scale_add");
	hls::stream<decimal_t> weights("weights");

	split<decimal_t, 0>(weights_data, scale_add, weights, scale_add_size, weights_size);

	conv2d(data, conv_out, weights, size, in_layers, out_layers, control.get_bit(CTRL_STREAM_WEIGHTS));

	if (control.get_bit(CTRL_STREAM_WEIGHTS)) {
		batch_norm_per_layer<decimal_t, 1024>(conv_out, batch_out, scale_add, size, out_layers);
	} else {
		batch_norm<decimal_t, 1024>(conv_out, batch_out, scale_add, size, out_layers);
	}

	if (control.get_bit(CTRL_LEAKY)) {
		leaky_relu<decimal_t>(batch_out, leaky_out, out_size);
	} else {
		direct<decimal_t, 1>(batch_out, leaky_out, out_size);
	}

	if (control.get_bit(CTRL_MAXPOOL)) {
		if (control.get_bit(CTRL_MAXPOOL1)) {
			max_pool_1<decimal_t, 416>(leaky_out, out, size, out_layers);
		} else {
			max_pool<decimal_t, 3328>(leaky_out, out, size, out_layers);
		}
	} else {
		direct<decimal_t, 0>(leaky_out, out, out_size);
	}
}

void conv2d(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int size, int in_layers, int out_layers, bool stream_weights) {
	ConvClass<decimal_t, 128, 64, 3328> c_impl;
	c_impl.set_size(size);
	c_impl.set_in_layers(in_layers);
	c_impl.set_out_layers(out_layers);
	if (stream_weights) {
		c_impl.load_input(in);
		c_impl.run_weights(weights, out);
	} else {
		c_impl.load_weights(weights);
		c_impl.convolute(in, out);
	}
}

int shift_from_layers(int layers) {
	if (layers <= 4) {
		return 2;
	} else if (layers <= 8) {
		return 3;
	} else if (layers <= 16) {
		return 4;
	} else if (layers <= 32) {
		return 5;
	} else if (layers <= 64) {
		return 6;
	} else if (layers <= 128) {
		return 7;
	} else if (layers <= 256) {
		return 8;
	} else if (layers <= 512) {
		return 9;
	} else if (layers <= 1024) {
		return 10;
	}
	return 11;
}
