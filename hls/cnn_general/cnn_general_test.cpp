#include "../cnn.h"

#include <iostream>

int main() {
	hls::stream<decimal_t> in("in");
	hls::stream<decimal_t> weights("weights");
	hls::stream<decimal_t> out("out");

	float v;
	for (int i = 0; i < 3*3*3; i++) {
		std::cin >> v;
		weights.write(v);
	}

	for (int i = 0; i < 416*416*3; i++) {
		std::cin >> v;
		in.write(v);
	}

	cnn_general(in, out, weights);

	for (int i = 0; i < 416*416; i++) {
		std::cout << out.read() << std::endl;
	}

	return 0;
}
