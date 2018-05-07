#ifndef CNN_H_
#define CNN_H_

#include <hls_stream.h>
#include <hls_video.h>
#include <hls_half.h>

typedef ap_fixed<18, 5, AP_TRN, AP_SAT> decimal_t;

void cnn(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &weights);

#include "cnn_impl.h"

#endif
