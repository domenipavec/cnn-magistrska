#include "../cnn.h"

#include <iostream>

int main() {
	hls::stream<decimal_t> in;
	hls::stream<decimal_t> weights;
	hls::stream<decimal_t> out;

	float v;
	decimal_t scale;
	decimal_t add;
	std::cin >> v;
	scale = v;
	std::cin >> v;
	add = v;
	for (int i = 0; i < 3*3*3; i++) {
		std::cin >> v;
		weights.write(v);
	}

	for (int i = 0; i < 416*416*3; i++) {
		std::cin >> v;
		in.write(v);
	}

	full_layer<decimal_t, 3, 416, 416>(in, out, weights, scale, add);

	for (int i = 0; i < 416*416; i++) {
		std::cout << out.read() << std::endl;
	}

	return 0;
}
