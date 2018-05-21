#include "cnn.h"

void cnn(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights) {
#pragma HLS DATAFLOW

	hls::stream<decimal_t> layer1_out("layer1_out");
	hls::stream<decimal_t> layer2_out("layer2_out");
	hls::stream<decimal_t> layer3_out("layer3_out");
	hls::stream<decimal_t> layer4_out("layer4_out");
	hls::stream<decimal_t> layer5_out("layer5_out");
	hls::stream<decimal_t> layer6_out("layer6_out");
	hls::stream<decimal_t> layer7_out("layer7_out");
	hls::stream<decimal_t> layer8_out("layer8_out");
	hls::stream<decimal_t> layer_weights[6];
	split<decimal_t, 6, 128>(weights, layer_weights);

	// layer 1
	full_layer_stack<decimal_t, 3, 16, 416, 416>(in, layer1_out, layer_weights[0]);

	// layer 2
	full_layer_stack<decimal_t, 16, 32, 208, 208>(layer1_out, layer2_out, layer_weights[1]);

	// layer 3
	full_layer_stack<decimal_t, 32, 64, 104, 104>(layer2_out, layer3_out, layer_weights[2]);

	// layer 4
	full_layer_stack<decimal_t, 64, 128, 52, 52>(layer3_out, layer4_out, layer_weights[3]);

	// layer 5
	full_layer_stack<decimal_t, 128, 256, 26, 26>(layer4_out, layer5_out, layer_weights[4]);

	// layer 6
	full_layer_stack<decimal_t, 256, 512, 13, 13>(layer5_out, layer6_out, layer_weights[5]);

//	// layer 7
//	fully_connected<decimal_t, 3, 16, 416, 416>(layer6_out, layer7_out, weights);
//
//	// layer 8
//	fully_connected<decimal_t, 3, 16, 416, 416>(layer7_out, layer8_out, weights);
//
//	// layer 9
//	conv2d<T, LAYERS, WIDTH, HEIGHT>(in, conv_out, weights_only);
//	batch_norm<T, WIDTH, HEIGHT>(conv_out, batch_out, 1.0, add); // add bias only
}

void cnn_full_layer_stack(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights) {
	full_layer_stack<decimal_t, 3, 16, 416, 416>(in, out, weights);
}

void cnn_general(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights) {
	general_conv2d<decimal_t, 128, 3328>(in, out, weights, 3, 416, 416);
}
