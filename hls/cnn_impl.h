template <typename T, int LAYERS, int WIDTH, int HEIGHT>
void conv2d(hls::stream<T> &in, hls::stream<T> &out, hls::stream<T> &weights) {
	// load weights
	T weights_buffer[3][3*LAYERS];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3*LAYERS; j++) {
			weights_buffer[i][j] = weights.read();
		}
	}

	hls::LineBuffer<3, WIDTH*LAYERS, T> line_buffer;

	// init top padding
	for (int j = 0; j < WIDTH*LAYERS; j++) {
		line_buffer.insert_bottom_row(0, j);
	}

	for (int i = 0; i <= HEIGHT; i++) {
		for (int j = 0; j <= WIDTH; j++) {
			// fill buffer
			if (j < WIDTH) {
				for (int l = 0; l < LAYERS; l++) {
					T new_val;
					if (i < HEIGHT) {
						new_val = in.read();
					} else {
						new_val = 0;
					}

					line_buffer.shift_pixels_up(j*LAYERS+l);
					line_buffer.insert_bottom_row(new_val, j*LAYERS+l);
				}
			}

			// convolution
			if (i > 0 && j > 0) {
				T sum = 0;
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3*LAYERS; l++) {
						if (j == 1 && l < LAYERS) {
							continue;
						}
						if (j == WIDTH && l >= 2*LAYERS) {
							continue;
						}
						sum += line_buffer.getval(k, (j-2)*LAYERS+l) * weights_buffer[k][l];
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
//#pragma HLS UNROLL
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

template <typename T, int MAX_LAYERS, int MAX_LINE>
class ConvClass {
private:
// dimensions: y, bank, l + x/3*layers (x is cyclic with bank)
	T line_buffer[3][3][MAX_LINE];
	T weights[MAX_LAYERS][3][3];
	int layers;
	int lshift;
	int width;
	int height;

protected:
	void push(int l, int b, int j, T v) {
// push is always used for whole line, so no close dependence
#pragma HLS DEPENDENCE variable=line_buffer inter false
#pragma HLS PIPELINE
		assert(l >= 0);
		assert(l < MAX_LAYERS);
		assert(l < layers);
		assert(b >= 0);
		assert(b < 3);
		assert(j >= 0);
		assert(j < MAX_LINE);
		assert(j < width);

		int idx = (j<<lshift) | l;
		assert(idx < MAX_LINE);
		PUSH: for (int k = 1; k < 3; k++) {
			line_buffer[k-1][b][idx] = line_buffer[k][b][idx];
		}
		line_buffer[2][b][idx] = v;
	}

	T push_and_conv(int l, int b, int j, T v) {
// inside one pipeline run (for layers) no dependence on line_buffer
#pragma HLS DEPENDENCE variable=line_buffer inter false
#pragma HLS PIPELINE
		assert(l >= 0);
		assert(l < MAX_LAYERS);
		assert(l < layers);
		assert(b >= 0);
		assert(b < 3);
		assert(j >= 0);
		assert(j < MAX_LINE);
		assert(j < width);

		int idx = (j << lshift) | l;
		int previdx = ((j-1) << lshift) | l;
		assert(idx < MAX_LINE);

		T sum(0);
		PC_I: for (int i = 0; i < 3; i++) {
			T tmp[3];
			switch (b) {
			case 0:
				if (previdx < 0) {
					tmp[0] = T(0);
					tmp[1] = T(0);
				} else {
					tmp[0] = line_buffer[i][1][previdx];
					tmp[1] = line_buffer[i][2][previdx];
				}
				if (i == 2 || 3*j+b >= width) {
					tmp[2] = v;
				} else {
					tmp[2] = line_buffer[i+1][0][idx];
				}
				if (3*j+b < width) {
					line_buffer[i][0][idx] = tmp[2];
				}
				break;
			case 1:
				if (previdx < 0) {
					tmp[0] = T(0);
				} else {
					tmp[0] = line_buffer[i][2][previdx];
				}
				tmp[1] = line_buffer[i][0][idx];
				if (i == 2 || 3*j+b >= width) {
					tmp[2] = v;
				} else {
					tmp[2] = line_buffer[i+1][1][idx];
				}
				if (3*j+b < width) {
					line_buffer[i][1][idx] = tmp[2];
				}
				break;
			case 2:
				tmp[0] = line_buffer[i][0][idx];
				tmp[1] = line_buffer[i][1][idx];
				if (i == 2 || 3*j+b >= width) {
					tmp[2] = v;
				} else {
					tmp[2] = line_buffer[i+1][2][idx];
				}
				if (3*j+b < width) {
					line_buffer[i][2][idx] = tmp[2];
				}
				break;
			}


			// partial sum for optimize summing
			T psum(0);
			PC_K: for (int k = 0; k < 3; k++) {
				psum += tmp[k]*weights[l][i][k];
			}
			sum += psum;
		}
		return sum;
	}

public:
	ConvClass() {
		// pragmas have to be in function, so we put them in constructor
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights complete dim=3

		layers = 0;
		width = 0;
		height = 0;
		lshift = 0;
	}

	void set_layers(int l) {
		assert(l <= MAX_LAYERS);
		assert(l <= 1024);

		layers = l;
		if (layers <= 4) {
			lshift = 2;
		} else if (layers <= 8) {
			lshift = 3;
		} else if (layers <= 16) {
			lshift = 4;
		} else if (layers <= 32) {
			lshift = 5;
		} else if (layers <= 64) {
			lshift = 6;
		} else if (layers <= 128) {
			lshift = 7;
		} else if (layers <= 256) {
			lshift = 8;
		} else if (layers <= 512) {
			lshift = 9;
		} else if (layers <= 1024) {
			lshift = 10;
		}
	}

	void set_width(int w) {
		assert(w < MAX_LINE);

		width = w;
	}

	void set_height(int h) {
		height = h;
	}

	void load_weights(hls::stream<T> &win) {
		LOAD_WEIGHTS_I: for (int i = 0; i < 3; i++) {
			LOAD_WEIGHTS_J: for (int j = 0; j < 3; j++) {
				LOAD_WEIGHTS_L: for (int l = 0; l < layers; l++) {
					weights[l][i][j] = win.read();
				}
			}
		}
	}

	void convolute(hls::stream<T> &in, hls::stream<T> &out) {
		INIT_2LINES: for (int i = 0; i < 2; i++) {
			INIT_WIDTH: for (int j = 0; 3*j < width; j++) {
				INIT_BANK: for (int b = 0; b < 3 && 3*j+b < width; b++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
					INIT_LAYERS: for (int l = 0; l < layers; l++) {
#pragma HLS PIPELINE
						T v(0);
						if (i == 1) {
							v = in.read();
						}
						push(l, b, j, v);
					}
				}
			}
		}

		CONV_HEIGHT: for (int i = 0; i < height; i++) {
			CONV_WIDTH: for (int j = 0; 3*j <= width; j++) {
				CONV_BANK: for (int b = 0; b < 3 && 3*j+b <= width; b++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
					T sum(0);
					CONV_LAYERS: for (int l = 0; l < layers; l++) {
#pragma HLS PIPELINE
						T v(0);
						if (3*j + b < width && i < height - 1) {
							v = in.read();
						}
						sum += push_and_conv(l, b, j, v);
					}
					if (!(j == 0 && b == 0)) {
						out.write(sum);
					}
				}
			}
		}
	}
};
