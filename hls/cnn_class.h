template <typename T, int MAX_IN_LAYERS, int MAX_OUT_LAYERS, int MAX_LINE>
class ConvClass {
private:
// dimensions: y, l + x*layers
	T line_buffer[2][MAX_LINE];
	T window[3*MAX_IN_LAYERS];
	T weights[MAX_IN_LAYERS][3][3][MAX_OUT_LAYERS];
	int in_layers;
	int out_layers;
	int lshift;
	int size;

protected:

public:
	ConvClass() {
		// pragmas have to be in function, so we put them in constructor
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights complete dim=3

		in_layers = 0;
		out_layers = 0;
		size = 0;
		lshift = 0;
	}

	void set_in_layers(int l) {
		assert(l <= MAX_IN_LAYERS);
		assert(l <= 1024);

		in_layers = l;
		lshift = shift_from_layers(l);
	}

	void set_out_layers(int l) {
		assert(l <= MAX_OUT_LAYERS);
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
						weights[l][i][j][o] = win.read();
					}
				}
			}
		}
	}

	void convolute(hls::stream<T> &in, hls::stream<T> &out) {

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
			window[3*l+2] = 0;
		}

		CONV_HEIGHT: for (int i = 0; i < size; i++) {
			CONV_WIDTH: for (int j = 0; j <= size; j++) {
				T sum[MAX_OUT_LAYERS];
				for (int o = 0; o < out_layers; o++) {
					sum[o] = T(0);
				}
				CONV_LAYERS: for (int l = 0; l < in_layers; l++) {
#pragma HLS PIPELINE
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
						window[3*l+k] = window[3*l+k+1];
					}
					window[3*l+2] = next_window;

					// sum window (first) row
					for (int k = 0; k < 3; k++) {
						for (int o = 0; o < out_layers; o++) {
							sum[o] += window[3*l+k]*weights[l][0][k][o];
						}
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

						PC_K: for (int k = 0; k < 3; k++) {
							for (int o = 0; o < out_layers; o++) {
								sum[o] += tmp[k]*weights[l][m+1][k][o];
							}
						}
//							suml += psum;
					}
//						sum += suml;
				}
				if (j != 0) {
					for (int o = 0; o < out_layers; o++) {
						out.write(sum[o]);
					}
				}
			}
		}
	}
};
