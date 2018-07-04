#include "../cnn.h"

#include <iostream>

int main() {
	hls::stream<decimal_t> in("in");
	hls::stream<decimal_t> weights("weights");
	hls::stream<decimal_t> out("out");

	float v;

	for (int i = 0; i < 11*11*512; i++) {
		std::cin >> v;
		in.write(v);
	}

	for (int i = 0; i < 3*3*512*1024; i++) {
		std::cin >> v;
		weights.write(v);
	}

	stream_weights(in, out, weights, 512, 1024);

	for (int i = 0; i < 11*11*1024; i++) {
		std::cout << out.read() << std::endl;
	}

	return 0;
}
