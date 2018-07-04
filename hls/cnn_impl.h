template <typename T, int MAX_SIZE, int MAX_OUT_LAYERS>
void max_pool(hls::stream<T> &in, hls::stream<T> &out, int size, int out_layers) {
	T buffer[MAX_SIZE/2][MAX_OUT_LAYERS];
	T tmp;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int l = 0; l < out_layers; l++) {
#pragma HLS PIPELINE
				tmp = in.read();
				if ((i & 1) == 0) {
					if ((j & 1) == 0) {
						buffer[j/2][l] = tmp;
					} else {
						if (tmp > buffer[j/2][l]) {
							buffer[j/2][l] = tmp;
						}
					}
				} else {
					if ((j & 1) == 0) {
						if (tmp > buffer[j/2][l]) {
							buffer[j/2][l] = tmp;
						}
					} else {
						if (tmp > buffer[j/2][l]) {
							buffer[j/2][l] = tmp;
						}
						out.write(buffer[j/2][l]);
					}
				}
			}
		}
	}
}

template <typename T, int MAX_SIZE>
void max_pool_1(hls::stream<T> &in, hls::stream<T> &out, int size, int out_layers) {
	T buffer[MAX_SIZE];

	T v1;
	T v2;
	T max_row;
	T max;

	for (int l = 0; l < out_layers; l++) {
		for (int i = 0; i <= size; i++) {
			v1 = 0;
			for (int j = 0; j <= size; j++) {
#pragma HLS PIPELINE
				if (j < size) {
					v2 = in.read();
				} else {
					v2 = 0;
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

template <typename T>
void direct(hls::stream<T> &in, hls::stream<T> &out, int size) {
	for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE
		out.write(in.read());
	}
}

template <typename T>
void split(hls::stream<T> &in, hls::stream<T> &out1, hls::stream<T> &out2, int size1, int size2) {
	for (int i = 0; i < size1; i++) {
		out1.write(in.read());
	}
	for (int i = 0; i < size2; i++) {
		out2.write(in.read());
	}
}

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

template <typename T, int SIZE, int MAX_LAYERS>
class StreamWeights {
private:
	T input_buffer[MAX_LAYERS][SIZE][SIZE];
	T output_buffer[SIZE][SIZE];

	int in_layers;
	int out_layers;

protected:
	void run_1output(hls::stream<T> &win, hls::stream<T> &out) {
		INIT_I: for (int i = 0; i < SIZE; i++) {
//#pragma HLS UNROLL
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
						APPLY_WEIGHT_K: for (int k = 0; k < SIZE; k++) {
#pragma HLS PIPELINE
							T value;
							if (j+l >= 0 && j+l < SIZE && k+m >= 0 && k+m < SIZE) {
								value = input_buffer[i][j+l][k+m];
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
				out.write(output_buffer[i][j]);
			}
		}
	}

public:
	StreamWeights() {
//#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=2
//#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=3
//#pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=1
//#pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=2

		in_layers = 1;
		out_layers = 1;
	}

	void set_in_layers(int l) {
		assert(l > 0);
		assert(l <= MAX_LAYERS);

		in_layers = l;
	}

	void set_out_layers(int l) {
		assert(l > 0);
		assert(l <= MAX_LAYERS);

		out_layers = l;
	}

	void load_input(hls::stream<T> &in) {
		for (int j = 0; j < SIZE; j++) {
			for (int k = 0; k < SIZE; k++) {
				for (int i = 0; i < in_layers; i++) {
						input_buffer[i][j][k] = in.read();
				}
			}
		}
	}

	void run(hls::stream<T> &win, hls::stream<T> &out) {
		for (int l = 0; l < out_layers; l++) {
			run_1output(win, out);
		}
	}
};
