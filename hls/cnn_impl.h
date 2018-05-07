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
