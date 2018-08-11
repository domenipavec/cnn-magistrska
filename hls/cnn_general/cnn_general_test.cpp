#include "../cnn.h"

#include <iostream>
#include <fstream>

int main() {
	// for float
//	const float err = 1.5e-4;
//	const float rel_err = 0;
	// for decimal
	const float err = 1.5e-2;
	const float rel_err = 1.5e-2;

	hls::stream<stream_t> in("in");
	hls::stream<stream_t> out("out");

	stream_t inV;
	inV.keep = 0b111;
	inV.strb = 0b111;

	inV.dest = 0;
	inV.id = 0;
	inV.user = 0;

	inV.last = 0;

	stream_t outV;

	std::ifstream imgin;
	imgin.open("/home/domen/github/magistrska/tensorflow/img.dat");
	int img_size;
	int img_layers;
	imgin >> img_size;
	imgin >> img_layers;
	bool in8bit = true;
	decimal_t *in_buffer;
	if (in8bit) {
		in_buffer = new decimal_t[img_size*img_size*img_layers/3];
	} else {
		in_buffer = new decimal_t[img_size*img_size*img_layers];
	}
	decimal_t *out_buffer;

	ap_uint<24> imgbyte;
	ap_uint<24> imgtotal;
	float imgvalue;
	for (int i = 0; i < img_size*img_size*img_layers; i++) {
		if (in8bit) {
			if (i % 3 == 0) {
				imgtotal = 0;
			}
			imgin >> imgbyte;
			imgtotal |= (imgbyte << (i%3)*8);
			if (i%3 == 2) {
				in_buffer[i/3].range() = imgtotal;
			}
		} else {
			imgin >> imgvalue;
			in_buffer[i] = imgvalue;
		}
	}

	std::ifstream weightsin;
	weightsin.open("/home/domen/github/magistrska/tensorflow/weights.dat");
	int n;
	weightsin >> n;

	bool prev_stream_weights = false;
	int size;
	int in_layers;
	int out_layers;
	int real_out_layers;
	int processed_layers;
	ap_uint<8> control;
	int out_size;

	for (int x = 0; x < n; x++) {
		weightsin >> size;
		weightsin >> in_layers;
		weightsin >> real_out_layers;
		weightsin >> control;

		out_size = size;
		if (control.get_bit(CTRL_MAXPOOL) && !control.get_bit(CTRL_MAXPOOL1)) {
			out_size = size/2;
		}
		out_buffer = new decimal_t[out_size*out_size*real_out_layers];
		for (int i = 0; i < out_size*out_size*real_out_layers; i++) {
			out_buffer[i] = 0;
		}

		out_layers = real_out_layers;
		if (!control.get_bit(CTRL_STREAM_WEIGHTS) && real_out_layers > 64) {
			out_layers = 64;
		}

		processed_layers = 0;
		bool still_working = true;
		while (still_working) {

			if (control.get_bit(CTRL_STREAM_WEIGHTS)) {
				if (!prev_stream_weights) { // shuffle output to different format
					std::cerr << "Reshuffling input" << std::endl;
					decimal_t *tmp = new decimal_t[size*size*in_layers];
					for (int i = 0; i < size*size; i++) {
						for (int j = 0; j < in_layers; j++) {
							tmp[i+j*size*size] = in_buffer[j + i*in_layers];
						}
					}
					decimal_t *tmp1 = in_buffer;
					in_buffer = tmp;
					delete[] tmp1;
				}
				for (int i = 0; i < size*size*in_layers; i++) {
					inV.data = in_buffer[i].range();
					in.write(inV);
				}
			}

			std::cerr << size << " " << in_layers << " " << out_layers << " " << control << std::endl;
			float v; // for multiple runs we format weights correctly in python
			float prev_v = -314.1234;
			for (int i = 0; i < 2*out_layers + 3*3*in_layers*out_layers; i++) {
				weightsin >> v;
				if (v == 0.0 && !(in_layers == 1024 && out_layers == 125)) {
					std::cerr << v << " " << i << std::endl;
				}
				if (v == prev_v && v != 0.0) {
					std::cerr << v << " " << i << " " << prev_v << std::endl;
				}
				inV.data = decimal_t(v).range();
				in.write(inV);
				prev_v = v;
			}

			if (!control.get_bit(CTRL_STREAM_WEIGHTS)) {
				int in_size = size*size*in_layers;
				if (control.get_bit(CTRL_8BITIN)) {
					in_size /= 3;
				}
				for (int i = 0; i < in_size; i++) {
					inV.data = in_buffer[i].range();
					in.write(inV);
				}
			}

			int asdf;
			cnn_general(in, out, size, in_layers, out_layers, control, asdf, 0);

			for (int i = 0; i < out_size*out_size; i++) {
				for (int o = 0; o < out_layers; o++) {
					outV = out.read();
					out_buffer[i*real_out_layers + processed_layers + o].range() = outV.data;
				}
			}

			if (!out.empty()) {
				std::cerr << "out not empty\n";
			}

			// check if multiple runs are required
			processed_layers += out_layers;
			if (processed_layers < real_out_layers) {
				out_layers = real_out_layers - processed_layers;
				if (out_layers > 64) {
					out_layers = 64;
				}
			} else {
				still_working = false;
			}
		}

		delete[] in_buffer;
		in_buffer = out_buffer;
		out_buffer = 0;

		prev_stream_weights = control.get_bit(CTRL_STREAM_WEIGHTS);
	}

	int result = 0;

	float v;
	std::cerr << "Comparing output for " << out_size << ", " << out_layers << std::endl;
	if (control.get_bit(CTRL_STREAM_WEIGHTS)) {
		for (int i = 0; i < out_size*out_size; i++) {
			for (int o = 0; o < out_layers; o++) {
				imgin >> v;
				float ibv = in_buffer[o*out_size*out_size + i];
				float adv = fabs(ibv - v);
				float radv = adv/fmax(fabs(ibv), fabs(v));
				if (adv > err && radv > rel_err) {
					std::cerr << "Wrong output for " << i << ", " << o << ": " << ibv << ", should be: " << v << " diff: " << adv << " rel: " << radv << std::endl;
					result = 1;
//					return result;
				}
			}
		}
	} else {
		for (int i = 0; i < out_size*out_size*out_layers; i++) {
			imgin >> v;
			float ibv = in_buffer[i];
			float adv = fabs(ibv - v);
			float radv = adv/fmax(fabs(ibv), fabs(v));
			if (adv > err && radv > rel_err) {
				std::cerr << "Wrong output for " << i << ": " << ibv << ", should be: " << v << " diff: " << adv << " rel: " << radv << std::endl;
				result = 1;
//				return result;
			}
		}
	}

	return result;
}
