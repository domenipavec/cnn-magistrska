template <typename T, int MAX_SIZE>
void max_pool(hls::stream<T> &in, hls::stream<T> &out, int size, int out_layers) {
	assert(size <= 416);
	assert(out_layers <= 256);

	T buffer[MAX_SIZE];
	T tmp;

	int shift = shift_from_layers(out_layers);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int l = 0; l < out_layers; l++) {
#pragma HLS PIPELINE
				int idx = ((j>>1) << shift) | l;
				tmp = in.read();
				if ((i & 1) == 0) {
					if ((j & 1) == 0) {
						buffer[idx] = tmp;
					} else {
						if (tmp > buffer[idx]) {
							buffer[idx] = tmp;
						}
					}
				} else {
					if ((j & 1) == 0) {
						if (tmp > buffer[idx]) {
							buffer[idx] = tmp;
						}
					} else {
						if (tmp > buffer[idx]) {
							buffer[idx] = tmp;
						}
						out.write(buffer[idx]);
					}
				}
			}
		}
	}
}

template <typename T, int MAX_SIZE>
void max_pool_1(hls::stream<T> &in, hls::stream<T> &out, int size, int out_layers) {
	assert(size <= 416);
	assert(out_layers <= 1024);

	T buffer[MAX_SIZE];

	T v1;
	T v2;
	T max_row;
	T max;

	for (int l = 0; l < out_layers; l++) {
		for (int i = 0; i < size; i++) {
			v1 = 0;
			for (int j = 0; j <= size; j++) {
#pragma HLS PIPELINE
				if (j < size) { // on last element simply keep v2
					v2 = in.read();
				}

				if (j > 0) {
					max_row = v2;
					if (v1 > max_row) {
						max_row = v1;
					}

					if (i > 0) {
						max = max_row;
						if (buffer[j-1] > max) {
							max = buffer[j-1];
						}
						out.write(max);
					}

					buffer[j-1] = max_row;
				}

				v1 = v2;
			}
		}
		for (int j = 0; j < size; j++) {
			out.write(buffer[j]);
		}
	}
}

template <typename T, int MAX_LAYERS>
void batch_norm_per_layer(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &scale_add, int size, int out_layers) {
	T data[MAX_LAYERS][2];
	for (int l = 0; l < out_layers; l++) {
		data[l][0] = scale_add.read();
		data[l][1] = scale_add.read();
	}

	for (int l = 0; l < out_layers; l++) {
		T scale = data[l][0];
		T add = data[l][1];

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
#pragma HLS PIPELINE
				T tmp = in.read();
				tmp *= scale;
				tmp += add;
				out.write(tmp);
			}
		}
	}
}

template <typename T, int MAX_LAYERS>
void batch_norm(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &scale_add, int size, int out_layers) {
	T data[MAX_LAYERS][2];
	for (int l = 0; l < out_layers; l++) {
		data[l][0] = scale_add.read();
		data[l][1] = scale_add.read();
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int l = 0; l < out_layers; l++) {
				T scale = data[l][0];
				T add = data[l][1];
#pragma HLS PIPELINE
				T tmp = in.read();
				tmp *= scale;
				tmp += add;
				out.write(tmp);
			}
		}
	}
}

template <typename T>
void leaky_relu(hls::stream<T> &in, hls::stream<T> &out, int size) {
	T tmp;

	for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
		tmp = in.read();
		if (tmp < 0) {
			tmp *= T(0.1);
		}
		out.write(tmp);
	}
}

template <typename T, int ID>
void direct(hls::stream<T> &in, hls::stream<T> &out, int size) {
	for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
		out.write(in.read());
	}
}

template <typename T, int ID>
void split(hls::stream<T> &in, hls::stream<T> &out1, hls::stream<T> &out2, int size1, int size2) {
	for (int i = 0; i < size1; i++) {
#pragma HLS PIPELINE
		out1.write(in.read());
	}
	for (int i = 0; i < size2; i++) {
#pragma HLS PIPELINE
		out2.write(in.read());
	}
}
