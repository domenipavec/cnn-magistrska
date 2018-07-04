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
