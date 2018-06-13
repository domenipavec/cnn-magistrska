#ifndef CNN_H_
#define CNN_H_

#include <hls_stream.h>
#include <hls_video.h>
#include <hls_half.h>

//typedef float decimal_t;
typedef ap_fixed<24, 7, AP_TRN, AP_SAT> decimal_t;

void cnn(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights);
void cnn_full_layer_stack(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights);
void conv2d_use_class(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int width, int height, int layers);
void cnn_general(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights, int width, int height, int layers);

template <typename T, int LAYERS, int WIDTH, int HEIGHT>
void conv2d(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &weights);
template <typename T, int WIDTH, int HEIGHT>
void max_pool(hls::stream<T> &in, hls::stream<T> &out);
template <typename T, int WIDTH, int HEIGHT>
void batch_norm(hls::stream<T> &in, hls::stream<T> &out, T scale, T add);
template <typename T, int WIDTH, int HEIGHT>
void leaky_relu(hls::stream<T> &in, hls::stream<T> &out);
template <typename T, int LAYERS, int WIDTH, int HEIGHT>
void full_layer(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &weights);
template <typename T, int SIZE>
void extract_scale_add(hls::stream<T> &weights, hls::stream<T> &weights_only, T &scale, T &add);
template <typename T, int IN_LAYERS, int OUT_LAYERS, int WIDTH, int HEIGHT>
void full_layer_stack(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &weights);
template <typename T, int LAYERS, int SIZE>
void broadcast(hls::stream<T> &in, hls::stream<T> outs[LAYERS]);
template <typename T, int LAYERS, int SIZE>
void split(hls::stream<T> &in, hls::stream<T> outs[LAYERS]);
template <typename T, int LAYERS, int SIZE>
void join(hls::stream<T> &out, hls::stream<T> ins[LAYERS]);

#include "cnn_impl.h"

#endif
