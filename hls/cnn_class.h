#define SIZE 11

template <typename T, int MAX_IN_LAYERS, int MAX_OUT_LAYERS, int MAX_LINE>
class ConvClass {
private:
// dimensions: y, l + x*layers
	T line_buffer[2][MAX_LINE];
	T buffer[2048][66];

	int in_layers;
	int out_layers;
	int lshift;
	int size;

protected:
	void run_1output(hls::stream<T> &win, hls::stream<T> &out) {
		T output_buffer[SIZE][SIZE];
#pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=2

		INIT_I: for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE
			INIT_J: for (int j = 0; j < SIZE; j++) {
#pragma HLS UNROLL
				output_buffer[i][j] = 0;
			}
		}

		CONV_L: for (int l = -1; l <= 1; l++) {
			CONV_M: for (int m = -1; m <= 1; m++) {
				CONV_IN_LAYERS: for (int i = 0; i < in_layers; i++) {
					T weight = win.read();

					APPLY_WEIGHT_J: for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
						APPLY_WEIGHT_K: for (int k = 0; k < SIZE; k++) {
#pragma HLS UNROLL
							T value;
							if (j+l >= 0 && j+l < SIZE && k+m >= 0 && k+m < SIZE) {
								value = buffer[(((j+l)&1)<<10)|i][((j+l)>>1)*11 + k+m];
							} else {
								value = 0;
							}
							output_buffer[j][k] += value*weight;
						}
					}
				}
			}

		}

		OUT_I: for (int i = 0; i < SIZE; i++) {
			OUT_J: for (int j= 0 ; j < SIZE; j++) {
#pragma HLS PIPELINE
				out.write(output_buffer[i][j]);
			}
		}
	}

public:
	ConvClass() {
		// pragmas have to be in function, so we put them in constructor
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=2

		in_layers = 0;
		out_layers = 0;
		size = 0;
		lshift = 0;
	}

	void set_in_layers(int l) {
		assert(l <= 1024);

		in_layers = l;
		lshift = shift_from_layers(l);
	}

	void set_out_layers(int l) {
		assert(l <= 1024);
		out_layers = l;
	}

	void set_size(int s) {
		assert(s < MAX_LINE);

		size = s;
	}

	void load_weights(hls::stream<T> &win) {
		LOAD_WEIGHTS_O: for (int o = 0; o < out_layers; o++) {
			LOAD_WEIGHTS_I: for (int i = 0; i < 3; i++) {
				LOAD_WEIGHTS_J: for (int j = 0; j < 3; j++) {
					LOAD_WEIGHTS_L: for (int l = 0; l < in_layers; l++) {
#pragma HLS PIPELINE
						buffer[((o & 0xf) << 7) | l][((o >> 4) << 4) | (i << 2) | j] = win.read();
					}
				}
			}
		}
	}

	void load_input(hls::stream<T> &in) {
		for (int i = 0; i < in_layers; i++) {
			for (int j = 0; j < SIZE; j++) {
				for (int k = 0; k < SIZE; k++) {
#pragma HLS PIPELINE
						buffer[((j&1)<<10)|i][(j>>1)*11 + k] = in.read();
				}
			}
		}
	}

	void run_weights(hls::stream<T> &win, hls::stream<T> &out) {
		for (int l = 0; l < out_layers; l++) {
			run_1output(win, out);
		}
	}

	void convolute(hls::stream<T> &in, hls::stream<T> &out) {
		T window[3][MAX_IN_LAYERS];
#pragma HLS ARRAY_PARTITION variable=window complete dim=1

		INIT_2LINES: for (int i = 0; i < 2; i++) {
			INIT_WIDTH: for (int j = 0; j < size; j++) {
				INIT_LAYERS: for (int l = 0; l < in_layers; l++) {
#pragma HLS PIPELINE
					T v(0);
					if (i == 1) {
						v = in.read();
					}

					assert(l >= 0);
					assert(l < MAX_IN_LAYERS);
					assert(l < in_layers);
					assert(j >= 0);
					assert(j < MAX_LINE);
					assert(j < size);

					int idx = (j<<lshift) | l;
					assert(idx < MAX_LINE);
					PUSH: for (int k = 1; k < 2; k++) {
						line_buffer[k-1][idx] = line_buffer[k][idx];
					}
					line_buffer[1][idx] = v;
				}
			}
		}

		for (int l = 0; l < in_layers; l++) {
			window[2][l] = 0;
		}

		CONV_HEIGHT: for (int i = 0; i < size; i++) {
			CONV_WIDTH: for (int j = 0; j <= size; j++) {
				T sum[MAX_OUT_LAYERS];
				for (int o = 0; o < out_layers; o++) {
#pragma HLS UNROLL
					sum[o] = T(0);
				}
				CONV_LAYERS: for (int l = 0; l < in_layers; l++) {
					T v(0);
					if (j < size && i < size - 1) {
						v = in.read();
					}

					assert(l >= 0);
					assert(l < MAX_IN_LAYERS);
					assert(l < in_layers);
					assert(j >= 0);
					assert(j < MAX_LINE);
					assert(j <= size);

					int idx = (j << lshift) | l;
					int pidx = ((j-1) << lshift) | l;
					int ppidx = ((j-2) << lshift) | l;
					assert(idx < MAX_LINE);

					// shift window
					T next_window(0);
					if (j < size) {
						next_window = line_buffer[0][idx];
					}
					for (int k = 0; k < 2; k++) {
#pragma HLS UNROLL
						window[k][l] = window[k+1][l];
					}
					window[2][l] = next_window;

					// sum window (first) row

					for (int o = 0; o < out_layers; o++) {
#pragma HLS PIPELINE
						T psum = 0;
						for (int k = 0; k < 3; k++) {
#pragma HLS UNROLL
							psum += window[k][l]*buffer[((o & 0xf) << 7) | l][((o >> 4) << 4) | (0 << 2) | k];
						}
						sum[o] += psum;
					}

					// sum other two rows
					PC_I: for (int m = 0; m < 2; m++) {
						T tmp[3];
						if (ppidx < 0) {
							tmp[0] = T(0);
						} else {
							tmp[0] = line_buffer[m][ppidx];
						}
						if (pidx < 0) {
							tmp[1] = T(0);
						} else {
							tmp[1] = line_buffer[m][pidx];
						}
						if (m == 1) {
							tmp[2] = v;
						} else {
							tmp[2] = line_buffer[m+1][idx];
						}

						line_buffer[m][idx] = tmp[2];


						// partial sum for optimize summing
//							T psum(0);


						for (int o = 0; o < out_layers; o++) {
#pragma HLS PIPELINE
							T psum = 0;
							PC_K: for (int k = 0; k < 3; k++) {
#pragma HLS UNROLL
								psum += tmp[k]*buffer[((o & 0xf) << 7) | l][((o >> 4) << 4) | ((m+1) << 2) | k];
							}
							sum[o] += psum;
						}
//							suml += psum;
					}
//						sum += suml;
				}
				if (j != 0) {
					for (int o = 0; o < out_layers; o++) {
#pragma HLS PIPELINE
						out.write(sum[o]);
					}
				}
			}
		}
	}
};
