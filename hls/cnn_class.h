template <typename T, int MAX_IN_LAYERS, int MAX_OUT_LAYERS, int MAX_LINE>
class ConvClass {
private:
// dimensions: y, bank, l + x/3*layers (x is cyclic with bank)
	T line_buffer[3][3][MAX_LINE];
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
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
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
		if (in_layers <= 4) {
			lshift = 2;
		} else if (in_layers <= 8) {
			lshift = 3;
		} else if (in_layers <= 16) {
			lshift = 4;
		} else if (in_layers <= 32) {
			lshift = 5;
		} else if (in_layers <= 64) {
			lshift = 6;
		} else if (in_layers <= 128) {
			lshift = 7;
		} else if (in_layers <= 256) {
			lshift = 8;
		} else if (in_layers <= 512) {
			lshift = 9;
		} else if (in_layers <= 1024) {
			lshift = 10;
		}
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
			INIT_WIDTH: for (int j = 0; 3*j < size; j++) {
				INIT_BANK: for (int b = 0; b < 3 && 3*j+b < size; b++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
					INIT_LAYERS: for (int l = 0; l < in_layers; l++) {
#pragma HLS PIPELINE
						T v(0);
						if (i == 1) {
							v = in.read();
						}

						assert(l >= 0);
						assert(l < MAX_IN_LAYERS);
						assert(l < in_layers);
						assert(b >= 0);
						assert(b < 3);
						assert(j >= 0);
						assert(j < MAX_LINE);
						assert(j < size);

						int idx = (j<<lshift) | l;
						assert(idx < MAX_LINE);
						PUSH: for (int k = 1; k < 3; k++) {
							line_buffer[k-1][b][idx] = line_buffer[k][b][idx];
						}
						line_buffer[2][b][idx] = v;
					}
				}
			}
		}

		CONV_HEIGHT: for (int i = 0; i < size; i++) {
			CONV_WIDTH: for (int j = 0; 3*j <= size; j++) {
				CONV_BANK: for (int b = 0; b < 3 && 3*j+b <= size; b++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3 avg=3
					T sum[MAX_OUT_LAYERS];
					for (int o = 0; o < out_layers; o++) {
						sum[o] = T(0);
					}
					CONV_LAYERS: for (int l = 0; l < in_layers; l++) {
#pragma HLS PIPELINE
						T v(0);
						if (3*j + b < size && i < size - 1) {
							v = in.read();
						}

						assert(l >= 0);
						assert(l < MAX_IN_LAYERS);
						assert(l < in_layers);
						assert(b >= 0);
						assert(b < 3);
						assert(j >= 0);
						assert(j < MAX_LINE);
						assert(j < size);

						int idx = (j << lshift) | l;
						int previdx = ((j-1) << lshift) | l;
						assert(idx < MAX_LINE);

						T suml(0);
						PC_I: for (int m = 0; m < 3; m++) {
							T tmp[3];
							switch (b) {
							case 0:
								if (previdx < 0) {
									tmp[0] = T(0);
									tmp[1] = T(0);
								} else {
									tmp[0] = line_buffer[m][1][previdx];
									tmp[1] = line_buffer[m][2][previdx];
								}
								if (m == 2 || 3*j+b >= size) {
									tmp[2] = v;
								} else {
									tmp[2] = line_buffer[m+1][0][idx];
								}
								if (3*j+b < size) {
									line_buffer[m][0][idx] = tmp[2];
								}
								break;
							case 1:
								if (previdx < 0) {
									tmp[0] = T(0);
								} else {
									tmp[0] = line_buffer[m][2][previdx];
								}
								tmp[1] = line_buffer[m][0][idx];
								if (m == 2 || 3*j+b >= size) {
									tmp[2] = v;
								} else {
									tmp[2] = line_buffer[m+1][1][idx];
								}
								if (3*j+b < size) {
									line_buffer[m][1][idx] = tmp[2];
								}
								break;
							case 2:
								tmp[0] = line_buffer[m][0][idx];
								tmp[1] = line_buffer[m][1][idx];
								if (m == 2 || 3*j+b >= size) {
									tmp[2] = v;
								} else {
									tmp[2] = line_buffer[m+1][2][idx];
								}
								if (3*j+b < size) {
									line_buffer[m][2][idx] = tmp[2];
								}
								break;
							}


							// partial sum for optimize summing
//							T psum(0);

							PC_K: for (int k = 0; k < 3; k++) {
								for (int o = 0; o < out_layers; o++) {
									sum[o] += tmp[k]*weights[l][m][k][o];
								}
							}
//							suml += psum;
						}
//						sum += suml;
					}
					if (!(j == 0 && b == 0)) {
						for (int o = 0; o < out_layers; o++) {
							out.write(sum[o]);
						}
					}
				}
			}
		}
	}
};
