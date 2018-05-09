template <typename T, int LAYERS, int WIDTH, int HEIGHT>
void conv2d(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &weights) {
	// load weights
	T weights_buffer[3][3*LAYERS];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3*LAYERS; j++) {
			weights_buffer[i][j] = weights.read();
		}
	}

	hls::LineBuffer<2, WIDTH*LAYERS, T> line_buffer;
	hls::Window<3, 3*LAYERS, T> window_buffer;

	// init top padding
	for (int j = 0; j < WIDTH*LAYERS; j++) {
		line_buffer.insert_bottom_row(0, j);
	}

	// init left padding
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < LAYERS; j++) {
			window_buffer.insert_pixel(0, i, 2*LAYERS+j);
		}
	}

	for (int i = 0; i <= HEIGHT; i++) {
		for (int j = 0; j <= WIDTH; j++) {
			// fill buffer
			for (int l = 0; l < LAYERS; l++) {
				window_buffer.shift_pixels_left();
				if (j < WIDTH) {
					T new_val;
					if (i < HEIGHT) {
						new_val = in.read();
					} else {
						new_val = 0;
					}

					window_buffer.insert_pixel(line_buffer.getval(0, j*LAYERS+l), 0, 3*LAYERS-1);
					window_buffer.insert_pixel(line_buffer.getval(1, j*LAYERS+l), 1, 3*LAYERS-1);
					window_buffer.insert_pixel(new_val, 2, 3*LAYERS-1);

					line_buffer.shift_pixels_up(j*LAYERS+l);
					line_buffer.insert_bottom_row(new_val, j*LAYERS+l);
				} else {
					for (int k =0; k < 3; k++) {
						window_buffer.insert_pixel(0, k, 3*LAYERS-1);
					}
				}
			}

			// convolution
			if (i > 0 && j > 0) {
				T sum = 0;
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3*LAYERS; l++) {
						sum += window_buffer.getval(k, l) * weights_buffer[k][l];
					}
				}
				out.write(sum);
			}
		}
	}
}

template <typename T, int WIDTH, int HEIGHT>
void max_pool(hls::stream<T> &in, hls::stream<T> &out) {
	T buffer[WIDTH/2];
	T tmp;
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			tmp = in.read();
			if ((i & 1) == 0) {
				if ((j & 1) == 0) {
					buffer[j/2] = tmp;
				} else {
					if (tmp > buffer[j/2]) {
						buffer[j/2] = tmp;
					}
				}
			} else {
				if ((j & 1) == 0) {
					if (tmp > buffer[j/2]) {
						buffer[j/2] = tmp;
					}
				} else {
					if (tmp > buffer[j/2]) {
						buffer[j/2] = tmp;
					}
					out.write(buffer[j/2]);
				}
			}
		}
	}
}

template <typename T, int WIDTH, int HEIGHT>
void batch_norm(hls::stream<T> &in, hls::stream<T> &out, T scale, T add) {
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			T tmp = in.read();
			tmp *= scale;
			tmp += add;
			out.write(tmp);
		}
	}
}

template <typename T, int WIDTH, int HEIGHT>
void leaky_relu(hls::stream<T> &in, hls::stream<T> &out) {
	T tmp;
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			tmp = in.read();
			if (tmp < 0) {
				tmp *= T(0.1);
			}
			out.write(tmp);
		}
	}
}

template <typename T, int LAYERS, int WIDTH, int HEIGHT>
void full_layer(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &weights) {
#pragma HLS DATAFLOW
	hls::stream<T> conv_out("full_layer_conv_out");
	hls::stream<T> batch_out("full_layer_batch_out");
	hls::stream<T> leaky_out("full_layer_leaky_out");

	hls::stream<T> weights_only("full_layer_weights_only");
	T scale;
	T add;
	extract_scale_add<T, 3*3*LAYERS>(weights, weights_only, scale, add);

	conv2d<T, LAYERS, WIDTH, HEIGHT>(in, conv_out, weights_only);
	batch_norm<T, WIDTH, HEIGHT>(conv_out, batch_out, scale, add);
	leaky_relu<T, WIDTH, HEIGHT>(batch_out, leaky_out);
	max_pool<T, WIDTH, HEIGHT>(leaky_out, out);
}

template <typename T, int SIZE>
void extract_scale_add(hls::stream<T> &weights, hls::stream<T> &weights_only, T &scale, T &add) {
	scale = weights.read();
	add = weights.read();

	for (int i = 0; i < SIZE; i++) {
		weights_only.write(weights.read());
	}
}

template <typename T, int IN_LAYERS, int OUT_LAYERS, int WIDTH, int HEIGHT>
void full_layer_stack(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &weights) {
#pragma HLS DATAFLOW
	hls::stream<T> layer_in[OUT_LAYERS];
	hls::stream<T> layer_out[OUT_LAYERS];
	hls::stream<T> layer_weights[OUT_LAYERS];

	broadcast<T, OUT_LAYERS, WIDTH*HEIGHT*IN_LAYERS>(in, layer_in);
	split<T, OUT_LAYERS, 2+3*3*IN_LAYERS>(weights, layer_weights);

	for (int l = 0; l < OUT_LAYERS; l++) {
//#pragma HLS UNROLL
		full_layer<T, IN_LAYERS, WIDTH, HEIGHT>(layer_in[l], layer_out[l], layer_weights[l]);
	}

	join<T, OUT_LAYERS, WIDTH/2*HEIGHT/2>(out, layer_out);
}

template <typename T, int LAYERS, int SIZE>
void broadcast(hls::stream<T> &in, hls::stream<T> outs[LAYERS]) {
	T tmp;
	for (int i = 0; i < SIZE; i++) {
		tmp = in.read();
		for (int l = 0; l < LAYERS; l++) {
#pragma HLS UNROLL
			outs[l].write(tmp);
		}
	}
}

template <typename T, int LAYERS, int SIZE>
void split(hls::stream<T> &in, hls::stream<T> outs[LAYERS]) {
	for (int i = 0; i < SIZE; i++) {
		for (int l = 0; l < LAYERS; l++) {
			outs[l].write(in.read());
		}
	}
}

template <typename T, int LAYERS, int SIZE>
void join(hls::stream<T> &out, hls::stream<T> ins[LAYERS]) {
	for (int i = 0; i < SIZE; i++) {
		for (int l = 0; l < LAYERS; l++) {
			out.write(ins[l].read());
		}
	}
}
