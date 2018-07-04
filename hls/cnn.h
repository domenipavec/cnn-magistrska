#ifndef CNN_H_
#define CNN_H_

#include <hls_stream.h>
#include <hls_video.h>
#include <hls_half.h>

//typedef float decimal_t;
typedef ap_fixed<24, 7, AP_TRN, AP_SAT> decimal_t;

void cnn(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights);
void cnn_full_layer_stack(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights);
void conv2d_use_class(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int size, int in_layers, int out_layers);
void conv2d_stream_weights(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int in_layers, int out_layers);
void cnn_general(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, int size, int in_layers, int out_layers, bool stream_weights, int max_type);

#include "cnn_impl.h"
#include "cnn_class.h"
#include "cnn_streamw.h"

#endif
