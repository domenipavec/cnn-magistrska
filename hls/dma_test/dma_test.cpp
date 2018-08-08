#include "../cnn.h"

#include "ap_axi_sdata.h"

typedef ap_axiu<24,1,1,1> axi24_t;
typedef ap_axiu<8,1,1,1> axi8_t;

void dma_test(hls::stream<axi24_t> &in, hls::stream<axi8_t> &out, int n) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=n bundle=CTRL_BUS
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in

	axi8_t valOut;

	valOut.keep = 0b111;
	valOut.strb = 0b111;

	valOut.dest = 0;
	valOut.id = 0;
	valOut.user = 0;

	valOut.last = 0;

	for (int i = 0; i < n; i++) {
		axi24_t valIn = in.read();

		valOut.data = valIn.data&0xff;
		out.write(valOut);

		valOut.data = (valIn.data >> 8)&0xff;
		out.write(valOut);

		if (i == n-1) {
			valOut.last = 1;
		}
		valOut.data = (valIn.data >> 16)&0xff;
		out.write(valOut);
	}
}
