#include "../cnn.h"

void cnn_sim(hls::stream<decimal_t> &data, hls::stream<decimal_t> &weights_data, hls::stream<decimal_t> &out, int size, int in_layers, int out_layers, int in_size, int out_size, int weights_size, int scale_add_size, ap_uint<8> control);
