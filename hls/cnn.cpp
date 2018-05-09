#include "cnn.h"

void cnn(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights) {
	//full_layer_stack<decimal_t, 3, 16, 416, 416>(in, out, weights);
}

void cnn_full_layer_stack(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights) {
	full_layer_stack<decimal_t, 3, 16, 416, 416>(in, out, weights);
}
