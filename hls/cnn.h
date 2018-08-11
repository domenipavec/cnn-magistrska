#ifndef CNN_H_
#define CNN_H_

#include <hls_stream.h>
#include <hls_video.h>
#include <hls_half.h>

#define DECIMAL_BITS 24
#define DECIMAL_ABOVE 7
//typedef float decimal_t;
typedef ap_fixed<DECIMAL_BITS, DECIMAL_ABOVE, AP_TRN, AP_SAT> decimal_t;

typedef ap_axiu<24,1,1,1> stream_t;

#define CTRL_STREAM_WEIGHTS 0
#define CTRL_LEAKY 1
#define CTRL_MAXPOOL 2
#define CTRL_MAXPOOL1 3
#define CTRL_8BITIN 4

int shift_from_layers(int layers);
void parse_input(hls::stream<stream_t> &in, hls::stream<decimal_t> &out, int size1, int size2, bool in8bit);
void format_output(hls::stream<decimal_t> &in, hls::stream<stream_t> &out, int size);

void cnn(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights);
void cnn_full_layer_stack(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights);
void conv2d(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int size, int in_layers, int out_layers, bool stream_weights);
void cnn_general(hls::stream<stream_t> &in, hls::stream<stream_t> &out, int size, int in_layers, int out_layers, ap_uint<8> control, int &progress, int prsize);

#include "cnn_impl.h"
#include "cnn_class.h"
#include "cnn_streamw.h"

#endif
