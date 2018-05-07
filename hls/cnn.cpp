#include "cnn.h"

void cnn(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights) {
	conv2d<decimal_t, 3, 416, 416>(in, out, weights);
}
