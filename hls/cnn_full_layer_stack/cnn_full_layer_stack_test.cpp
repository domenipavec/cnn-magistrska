#include "../cnn.h"

#include <iostream>

int main() {
	hls::stream<decimal_t> in("main_input");
	hls::stream<decimal_t> weights("main_weights");
	hls::stream<decimal_t> out("main_out");

	float v;
	for (int i = 0; i < (2+3*3*3)*16; i++) {
		std::cin >> v;
		weights.write(v);
	}

	for (int i = 0; i < 416*416*3; i++) {
		std::cin >> v;
		in.write(v);
	}

	full_layer_stack<decimal_t, 3, 16, 416, 416>(in, out, weights);

	for (int i = 0; i < 416/2*416/2*16; i++) {
		v = out.read();
		std::cout << v << std::endl;
	}

	return 0;
}
