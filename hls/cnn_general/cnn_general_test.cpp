#include "../cnn.h"

#include <iostream>

int main() {
	hls::stream<decimal_t> in("in");
	hls::stream<decimal_t> out("out");

	int size;
	int in_layers;
	int out_layers;
	bool stream_weights;
	int max_type;

	std::cin >> size;
	std::cin >> in_layers;
	std::cin >> out_layers;
	std::cin >> stream_weights;
	std::cin >> max_type;

	std::cerr << size << " " << in_layers << " " << out_layers << " " << stream_weights << " " << max_type << std::endl;
	float v;
	for (int i = 0; i < 2*out_layers + 3*3*in_layers*out_layers + size*size*in_layers; i++) {
		std::cin >> v;
		if (v == 0.0) {
			std::cerr << v << " " << i << std::endl;
		}
		in.write(v);
	}

	cnn_general(in, out, size, in_layers, out_layers, stream_weights, max_type);

	int out_size = size;
	if (max_type == 2) {
		out_size = size/2;
	}
	for (int i = 0; i < out_size*out_size*out_layers; i++) {
		std::cout << out.read() << std::endl;
	}

	return 0;
}
