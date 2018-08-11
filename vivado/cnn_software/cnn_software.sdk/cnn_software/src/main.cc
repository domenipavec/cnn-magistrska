#include <iostream>
#include <math.h>

#include "image.h"
#include "weights.h"

#include "timer.h"

#define CTRL_STREAM_WEIGHTS 0
#define CTRL_LEAKY 1
#define CTRL_MAXPOOL 2
#define CTRL_MAXPOOL1 3
#define CTRL_8BITIN 4

//#define CHECKS
//#define CLEARS

float BUFFER1[3000000];
float BUFFER2[3000000];

void clear(float *buffer, int size) {
	for (int i = 0; i < size; i++) {
		buffer[i] = 0;
	}
}

int main()
{
	const float err = 1e-3;
	const float rel_err = 1e-3;

	// INITIALIZE
	Timer timer;

	float *txBuffer = BUFFER1;
	float *rxBuffer = BUFFER2;

	for (int x = 0; x < n; x++) {
		std::cout << "Starting layer " << x << std::endl;
		timer.start();

#ifdef CHECKS
		if (x < 0 || x >= int(sizeof(weights)/sizeof(weights[0]))) {
			std::cout << "Invalid access weights" << std::endl;
			return 1;
		}
#endif
		float *current_weights = weights[x];


		// initialize input
		if (x == 0) {
#ifdef CLEARS
			clear(txBuffer, layer_sizes[x]*layer_sizes[x]*input_layers[x]);
#endif
			for (int i = 0; i < layer_sizes[x]; i++) {
				for (int j = 0; j < layer_sizes[x]; j++) {
					for (int l = 0; l < input_layers[x]; l++) {
						int txBufi = i*layer_sizes[x]*input_layers[x] + j*input_layers[x]+l;
#ifdef CHECKS
						if (txBufi < 0 || txBufi >= int(sizeof(BUFFER1)/sizeof(BUFFER1[0]))) {
							std::cout << "Invalid access init tx" << std::endl;
							return 1;
						}
#endif
						int imgi = i*layer_sizes[x]*input_layers[x] + j*input_layers[x]+l;
#ifdef CHECKS
						if (imgi < 0 || imgi >= int(sizeof(img)/sizeof(img[0]))) {
							std::cout << "Invalid access init img" << std::endl;
							return 1;
						}
#endif
						txBuffer[txBufi] = img[imgi];
						txBuffer[txBufi] /= 256;
					}
				}
			}
		}

#ifdef CLEARS
		clear(rxBuffer, layer_sizes[x]*layer_sizes[x]*output_layers[x]);
#endif

		// convolution
		for (int i = 0; i < layer_sizes[x]; i++) {
			for (int j = 0; j < layer_sizes[x]; j++) {
				for (int lo = 0; lo < output_layers[x]; lo++) {
					float res = 0;

					// convolution
					for (int k = 0; k < 3; k++) {
						for (int l = 0; l < 3; l++) {
							for (int li = 0; li < input_layers[x]; li++) {
								if (i+k-1 < 0 || i+k-1 >= layer_sizes[x]) {
									continue;
								}
								if (j+l-1 < 0 || j+l-1 >= layer_sizes[x]) {
									continue;
								}
								int cwi = 2*output_layers[x] + 3*3*input_layers[x]*lo+3*input_layers[x]*k+l*input_layers[x]+li;
#ifdef CHECKS

								if (cwi < 0 || cwi >= weights_lens[x]) {
									std::cout << "Invalid access cv cw" << std::endl;
									return 1;
								}
#endif
								int txi = (i+k-1)*layer_sizes[x]*input_layers[x] + (j+l-1)*input_layers[x]+li;
#ifdef CHECKS
								if (txi < 0 || txi >= int(sizeof(BUFFER1)/sizeof(BUFFER1[0]))) {
									std::cout << "Invalid access cv tx" << std::endl;
									return 1;
								}
#endif
								float w = current_weights[cwi];
								float v = txBuffer[txi];
								res += w*v;
							}
						}
					}

					// batch norm
					res *= current_weights[2*lo];
					res += current_weights[2*lo+1];

					// leaky relu
					if (controls[x] & (1<<CTRL_LEAKY)) {
						if (res < 0) {
							res *= 0.1;
						}
					}

					int rxi = i*layer_sizes[x]*output_layers[x] + j*output_layers[x]+lo;
#ifdef CHECKS
					if (rxi < 0 || rxi >= int(sizeof(BUFFER1)/sizeof(BUFFER1[0]))) {
						std::cout << "Invalid access cv rx" << std::endl;
						return 1;
					}
#endif
					rxBuffer[rxi] = res;
				}
			}
		}

		// maxpool 1 or 2
		if (controls[x] & (1<<CTRL_MAXPOOL)) {
			// change output buffers for maxpool
			float *tmp = rxBuffer;
			rxBuffer = txBuffer;
			txBuffer = tmp;

#ifdef CLEARS
			clear(rxBuffer, layer_sizes[x]*layer_sizes[x]*output_layers[x]);
#endif

			if (controls[x] & (1<<CTRL_MAXPOOL1)) {
				std::cout << "Maxpool 1" << std::endl;
				for (int i = 0; i < layer_sizes[x]; i++) {
					for (int j = 0; j < layer_sizes[x]; j++) {
						for (int lo = 0; lo < output_layers[x]; lo++) {
							float vmax = -1e30;
							for (int k = 0; k < 2; k++) {
								for (int l = 0; l < 2; l++) {
									if (i+k >= layer_sizes[x] || j+l >= layer_sizes[x]) {
										continue;
									}

									int txi = (i+k)*layer_sizes[x]*output_layers[x] + (j+l)*output_layers[x]+lo;
#ifdef CHECKS
									if (txi < 0 || txi >= int(sizeof(BUFFER1)/sizeof(BUFFER1[0]))) {
										std::cout << "Invalid access mp1 tx" << std::endl;
										return 1;
									}
#endif
									float v = txBuffer[txi];
									if (v > vmax) {
										vmax = v;
									}
								}
							}
							int rxi = i*layer_sizes[x]*output_layers[x] + j*output_layers[x]+lo;
#ifdef CHECKS
							if (rxi < 0 || rxi >= int(sizeof(BUFFER1)/sizeof(BUFFER1[0]))) {
								std::cout << "Invalid access mp1 rx" << std::endl;
								return 1;
							}
#endif
							rxBuffer[rxi] = vmax;
						}
					}
				}
			} else {
				std::cout << "Maxpool 2" << std::endl;
				for (int i = 0; i < layer_sizes[x]/2; i++) {
					for (int j = 0; j < layer_sizes[x]/2; j++) {
						for (int lo = 0; lo < output_layers[x]; lo++) {
							float vmax = -1e30;
							for (int k = 0; k < 2; k++) {
								for (int l = 0; l < 2; l++) {
									int txi = (2*i+k)*layer_sizes[x]*output_layers[x] + (2*j+l)*output_layers[x]+lo;
#ifdef CHECKS
									if (txi < 0 || txi >= int(sizeof(BUFFER1)/sizeof(BUFFER1[0]))) {
										std::cout << "Invalid access mp2 tx" << std::endl;
										return 1;
									}
#endif
									float v = 0;
									if (2*i+k < layer_sizes[x] && 2*j+l < layer_sizes[x]) {
										v = txBuffer[txi];
									}
									if (v > vmax) {
										vmax = v;
									}
								}
							}
							int rxi = i*layer_sizes[x]/2*output_layers[x] + j*output_layers[x]+lo;
#ifdef CHECKS
							if (rxi < 0 || rxi >= int(sizeof(BUFFER1)/sizeof(BUFFER1[0]))) {
								std::cout << "Invalid access mp2 rx" << std::endl;
								return 1;
							}
#endif
							rxBuffer[rxi] = vmax;
						}
					}
				}
			}
		}

		// change output buffers
		float *tmp = rxBuffer;
		rxBuffer = txBuffer;
		txBuffer = tmp;

		timer.stop();
		std::cout << "Layer " << x << " took: " << timer << std::endl;
	}

	int outSize = layer_sizes[n-1]*layer_sizes[n-1]*output_layers[n-1];
	if ((controls[n-1] & (1 << CTRL_MAXPOOL)) && !(controls[n-1] & (1 << CTRL_MAXPOOL1))) {
		outSize /= 4;
	}
//	if (controls[n-1] & (1 << CTRL_STREAM_WEIGHTS)) {
//		for (int i = 0; i < layer_sizes[n-1]*layer_sizes[n-1]; i++) {
//			for (int o = 0; o < output_layers[n-1]; o++) {
//				uint32_t uintv = *((uint32_t *)&txBuffer[(o*layer_sizes[n-1]*layer_sizes[n-1] + i)*3]);
//				uintv <<= 8;
//				int32_t intv = *((int32_t *) &uintv);
//				intv >>= 8;
//				float v = intv;
//				v /= (1 << FIXED_POINT_SHIFT);
//				float vexp = expected[i*output_layers[n-1]+o];
//				float adv = fabs(v-vexp);
//				float radv = adv/fmax(fabs(v), fabs(vexp));
//				if (adv > err && radv > rel_err) {
//					std::cout << "Wrong output for " << i << ", " << o << ": " << v << ", should be: " << vexp << " diff: " << adv << " rel: " << radv << std::endl;
//				}
//			}
//		}
//	} else {
		for (int i = 0; i < outSize; i++) {
			float v = txBuffer[i];
			float vexp = expected[i];
			float adv = fabs(v - vexp);
			float radv = adv/fmax(fabs(v), fabs(vexp));
			if (adv > err && radv > rel_err) {
				std::cout << "Wrong output for " << i << ": " << v << ", should be: " << vexp << " diff: " << adv << " rel: " << radv << std::endl;
			}
		}


	std::cout << "Finished" << std::endl;

	return 0;
}
