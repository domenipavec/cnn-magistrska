#include "cnn.h"

template <int ID>
void sink(hls::stream<decimal_t> &in, int size) {
	for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
		in.read();
	}
}

void source(hls::stream<decimal_t> &out, int size) {
	for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
		if (size < (1 << (DECIMAL_ABOVE - 1))) {
			out.write(decimal_t(i));
		} else {
			out.write(decimal_t(1.1));
		}
	}
}

void measure(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, int size, int &real) {
	for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
		real = i;
		out.write(in.read());
	}
}

void split_data_weights(hls::stream<decimal_t> &in, hls::stream<decimal_t> &weights, hls::stream<decimal_t> &data, int weights_size, int data_size, bool stream_weights) {
	if (stream_weights) {
		split<decimal_t, 1>(in, data, weights, data_size, weights_size);
	} else {
		split<decimal_t, 2>(in, weights, data, weights_size, data_size);
	}
}

void max_pool_full(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, ap_uint<8> control, int size, int out_layers, int out_size) {
	if (control.get_bit(CTRL_MAXPOOL)) {
		if (control.get_bit(CTRL_MAXPOOL1)) {
			max_pool_1<decimal_t, 416>(in, out, size, out_layers);
		} else {
			max_pool<decimal_t, 3328>(in, out, size, out_layers);
		}
	} else {
		direct<decimal_t, 0>(in, out, out_size);
	}
}

void leaky_relu_full(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, int size, bool leaky) {
	if (leaky) {
		leaky_relu<decimal_t>(in, out, size);
	} else {
		direct<decimal_t, 1>(in, out, size);
	}
}

void batch_norm_full(hls::stream<decimal_t> &in, hls::stream<decimal_t> &out, hls::stream<decimal_t> &scale_add, int size, int out_layers, bool stream_weights) {
	if (stream_weights) {
		batch_norm_per_layer<decimal_t, 1024>(in, out, scale_add, size, out_layers);
	} else {
		batch_norm<decimal_t, 1024>(in, out, scale_add, size, out_layers);
	}
}

void cnn_general(hls::stream<stream_t> &in, hls::stream<stream_t> &out, int size, int in_layers, int out_layers, int in_size, int out_size, int weights_size, int scale_add_size, ap_uint<8> control, int &progress, int prsize) {
#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in

#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=size bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=in_layers bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=out_layers bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=in_size bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=out_size bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=weights_size bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=scale_add_size bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=control bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=progress bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=prsize bundle=CTRL_BUS

#pragma HLS DATAFLOW

	// general asserts
	assert(size > 0);
	assert(size <= 416);
	assert(in_size > 0);
	assert(in_size <= 416*416*1024);
	assert(out_size > 0);
	assert(out_size <= 416*416*1024);
	assert(weights_size > 0);
	assert(weights_size <= 3*3*1024*1024);
	assert(scale_add_size > 0);
	assert(scale_add_size <= 2*1024);
	assert(in_layers > 0);
	assert(in_layers <= 1024);
	assert(out_layers > 0);
	assert(out_layers <= 1024);

	// test asserts
//	assert(size <= 13);
//	assert(in_layers <= 1024);
//	assert(out_layers <= 1024);
//	assert(stream_weights == 1);
//	assert(max_type == 0);

	hls::stream<decimal_t> non_axi_in("non_axi_in");
#pragma HLS STREAM variable=non_axi_in depth=1 dim=1
	hls::stream<decimal_t> non_axi_in1("non_axi_in1");
#pragma HLS STREAM variable=non_axi_in1 depth=1 dim=1
	hls::stream<decimal_t> non_axi_out("non_axi_out");
#pragma HLS STREAM variable=non_axi_out depth=1 dim=1

	hls::stream<decimal_t> conv_out("conv_out");
#pragma HLS STREAM variable=conv_out depth=1 dim=1
	hls::stream<decimal_t> batch_out("batch_out");
#pragma HLS STREAM variable=batch_out depth=1 dim=1
	hls::stream<decimal_t> leaky_out("leaky_out");
#pragma HLS STREAM variable=leaky_out depth=1 dim=1

	hls::stream<decimal_t> weights_data("weights_data");
#pragma HLS STREAM variable=weights_data depth=1 dim=1
	hls::stream<decimal_t> scale_add("scale_add");
#pragma HLS STREAM variable=scale_add depth=1 dim=1
	hls::stream<decimal_t> weights("weights");
#pragma HLS STREAM variable=weights depth=1 dim=1
	hls::stream<decimal_t> data("data");
#pragma HLS STREAM variable=data depth=1 dim=1

	parse_input(in, non_axi_in, weights_size+scale_add_size, in_size, control.get_bit(CTRL_8BITIN));

	measure(non_axi_in, non_axi_in1, weights_size+scale_add_size+in_size, progress);

	split_data_weights(non_axi_in1, weights_data, data, weights_size+scale_add_size, in_size, control.get_bit(CTRL_STREAM_WEIGHTS));

	split<decimal_t, 0>(weights_data, scale_add, weights, scale_add_size, weights_size);

	conv2d(data, conv_out, weights, size, in_layers, out_layers, control.get_bit(CTRL_STREAM_WEIGHTS));

	batch_norm_full(conv_out, batch_out, scale_add, size, out_layers, control.get_bit(CTRL_STREAM_WEIGHTS));

	leaky_relu_full(batch_out, leaky_out, out_size, control.get_bit(CTRL_LEAKY));

	max_pool_full(leaky_out, non_axi_out, control, size, out_layers, out_size);

	int final_size = out_size;
	if (control.get_bit(CTRL_MAXPOOL) && !control.get_bit(CTRL_MAXPOOL1)) {
		final_size /= 4;
	}

	format_output(non_axi_out, out, final_size);
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

void parse_input(hls::stream<stream_t> &in, hls::stream<decimal_t> &out, int size1, int size2, bool in8bit) {
	stream_t tmp;
	decimal_t value;

	for (int i = 0; i < size1; i++) {
#pragma HLS PIPELINE
		tmp = in.read();
		value.range() = tmp.data;
		out.write(value);
	}
	if (in8bit) {
		for (int i = 0; i < size2/3; i++) {
#pragma HLS PIPELINE
			tmp = in.read();

			value.range() = (tmp.data & 0xff)<< (DECIMAL_BITS - DECIMAL_ABOVE - 8);
			out.write(value);
			value.range() = ((tmp.data >> 8) & 0xff)<< (DECIMAL_BITS - DECIMAL_ABOVE - 8);
			out.write(value);
			value.range() = ((tmp.data >> 16) & 0xff)<< (DECIMAL_BITS - DECIMAL_ABOVE - 8);
			out.write(value);
		}
	} else {
		for (int i = 0; i < size2; i++) {
#pragma HLS PIPELINE
			tmp = in.read();
			value.range() = tmp.data;
			out.write(value);
		}
	}
}

void format_output(hls::stream<decimal_t> &in, hls::stream<stream_t> &out, int size) {
	stream_t valOut;

	valOut.keep = 0b111;
	valOut.strb = 0b111;

	valOut.dest = 0;
	valOut.id = 0;
	valOut.user = 0;

	valOut.last = 0;

	for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
		if (i == size-1) {
			valOut.last = 1;
		}

		valOut.data = in.read().range();
		out.write(valOut);
	}
}
