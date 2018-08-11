#include <iostream>
#include <math.h>

#include "image.h"
#include "weights.h"

#include "timer.h"

#include "xaxidma.h"
#include "xcnn_general.h"

#define CTRL_STREAM_WEIGHTS 0
#define CTRL_LEAKY 1
#define CTRL_MAXPOOL 2
#define CTRL_MAXPOOL1 3
#define CTRL_8BITIN 4

#define FIXED_POINT_SHIFT (24-7)
#define MAX_OUT_LAYERS 64
#define RX_MAX_VSIZE 1936
#define WEIGHTS_MAX_VSIZE 4096

uint8_t BUFFER1[3000000];
uint8_t BUFFER2[3000000];
uint8_t RX_BD[100000];
uint8_t TX_BD[100000];

uint8_t * align(uint8_t *p, int alignment) {
	p += alignment-1;
	p -= ((int)p)%alignment;
	return p;
}

int main()
{
	const float err = 5e-2;
	const float rel_err = 5e-2;

	// INITIALIZE
	Timer timer;

	XCnn_general cnnGeneral;
	XCnn_general_Config *cnnGeneralConfig;
	XAxiDma axiDma;
	XAxiDma_Config *axiDmaConfig;

	cnnGeneralConfig = XCnn_general_LookupConfig(XPAR_CNN_GENERAL_0_DEVICE_ID);
	if (cnnGeneralConfig) {
		int status = XCnn_general_CfgInitialize(&cnnGeneral, cnnGeneralConfig);
		if (status != XST_SUCCESS) {
			std::cout << "Could not initialize dma test" << std::endl;
			return 1;
		}
	}
	XCnn_general_DisableAutoRestart(&cnnGeneral);

	axiDmaConfig = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
	if (axiDmaConfig) {
		int status = XAxiDma_CfgInitialize(&axiDma, axiDmaConfig);
		if (status != XST_SUCCESS) {
			std::cout << "Could not initialize axi dma" << std::endl;
			return 1;
		}
	}
	if (!XAxiDma_HasSg(&axiDma)) {
		std::cout << "No scather gather in dma" << std::endl;
		return 1;
	}

	// disable dma interrupts
	XAxiDma_IntrDisable(&axiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&axiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

	// tx ring init
	XAxiDma_BdRing *txRing = XAxiDma_GetTxRing(&axiDma);
	int status = XAxiDma_BdRingCreate(txRing, (UINTPTR)align(TX_BD, XAXIDMA_BD_MINIMUM_ALIGNMENT), (UINTPTR)align(TX_BD, XAXIDMA_BD_MINIMUM_ALIGNMENT), XAXIDMA_BD_MINIMUM_ALIGNMENT, 20*2);
	if (status != XST_SUCCESS) {
		std::cout << "Could not create bd ring" << std::endl;
		return 1;
	}

	// rx ring init
	XAxiDma_BdRing *rxRing = XAxiDma_GetRxRing(&axiDma);
	status = XAxiDma_BdRingCreate(rxRing, (UINTPTR)align(RX_BD, XAXIDMA_BD_MINIMUM_ALIGNMENT), (UINTPTR)align(RX_BD, XAXIDMA_BD_MINIMUM_ALIGNMENT), XAXIDMA_BD_MINIMUM_ALIGNMENT, layer_sizes[0]*layer_sizes[0]/4/RX_MAX_VSIZE);
	if (status != XST_SUCCESS) {
		std::cout << "Could not create bd ring" << std::endl;
		return 1;
	}

	uint8_t *txBuffer = img;
	uint8_t *rxBuffer = align(BUFFER1, 4);

	bool prev_stream_weights = false;

	for (int x = 0; x < n; x++) {
		if (controls[x] & (1 << CTRL_STREAM_WEIGHTS)) {
			if (!prev_stream_weights) { // shuffle output to different format
				std::cout << "Reshuffling input" << std::endl;
				timer.start();

				for (int i = 0; i < layer_sizes[x]*layer_sizes[x]; i++) {
					for (int j = 0; j < input_layers[x]; j++) {
						for (int k = 0; k < 3; k++) {
							rxBuffer[(i+j*layer_sizes[x]*layer_sizes[x])*3+k] = txBuffer[(j + i*input_layers[x])*3+k];
						}
					}
				}
				uint8_t *tmp = txBuffer;
				txBuffer = rxBuffer;
				rxBuffer = tmp;

				timer.stop();
				std::cout << "Reshuffling took: " << timer << std::endl;
			}
		}

		std::cout << "Starting layer " << x << std::endl;
		if (x == 8) {
			timer.stop();
		}
		timer.start();

		int out_layers = output_layers[x];
		if (!(controls[x] & (1 << CTRL_STREAM_WEIGHTS)) && out_layers > MAX_OUT_LAYERS) {
			out_layers = MAX_OUT_LAYERS;
		}

		// PREPARE config
		int txHSize = input_layers[x]*layer_sizes[x];
		int txVSize = 3*layer_sizes[x];
		if (controls[x] & (1 << CTRL_8BITIN)) {
			txVSize /= 3;
		}
		int txSize = txHSize*txVSize;

		int rxVSize = layer_sizes[x]*layer_sizes[x];
		if ((controls[x] & (1 << CTRL_MAXPOOL)) && !(controls[x] & (1 << CTRL_MAXPOOL1))) {
			rxVSize /= 4;
		}
		int rxHSize = 3*out_layers;
		int rxStride = 3*output_layers[x];
		int rxSize = rxStride*rxVSize;

		int weightsHSize = 3*out_layers; // hsize needs to be divisible by 4
		int weightsVSize = 3*3*input_layers[x] + 2;
		int weightsSize = weightsVSize*3*output_layers[x];

		Xil_DCacheFlushRange((INTPTR)rxBuffer, rxSize);
		Xil_DCacheFlushRange((INTPTR)txBuffer, txSize);
		Xil_DCacheFlushRange((INTPTR)weights[x], weightsSize);

		for (int p = 0; ((controls[x] & (1 << CTRL_STREAM_WEIGHTS)) && p == 0 ) || (!(controls[x] & (1 << CTRL_STREAM_WEIGHTS)) && p < (output_layers[x]+MAX_OUT_LAYERS-1)/MAX_OUT_LAYERS); p++) {
			if (!(controls[x] & (1 << CTRL_STREAM_WEIGHTS)) && output_layers[x]/MAX_OUT_LAYERS > 1) {
				std::cout << "Computing part " << p << "/" << output_layers[x]/MAX_OUT_LAYERS << std::endl;
			}

			// EXECUTE

			XCnn_general_Set_size(&cnnGeneral, layer_sizes[x]);
			XCnn_general_Set_in_layers(&cnnGeneral, input_layers[x]);
			XCnn_general_Set_out_layers(&cnnGeneral, out_layers);
			XCnn_general_Set_in_size(&cnnGeneral, layer_sizes[x]*layer_sizes[x]*input_layers[x]);
			XCnn_general_Set_out_size(&cnnGeneral, layer_sizes[x]*layer_sizes[x]*out_layers);
			XCnn_general_Set_weights_size(&cnnGeneral, 3*3*input_layers[x]*out_layers);
			XCnn_general_Set_scale_add_size(&cnnGeneral, 2*out_layers);
			XCnn_general_Set_control_V(&cnnGeneral, controls[x]);
			XCnn_general_Set_prsize(&cnnGeneral, 372176);
			XCnn_general_Start(&cnnGeneral);
			// TODO remove progress
			int progress = XCnn_general_Get_progress(&cnnGeneral);

			// TX descriptors
			int weights_bd_count = (weightsVSize + WEIGHTS_MAX_VSIZE - 1) / WEIGHTS_MAX_VSIZE;
			XAxiDma_Bd *txBd, *startTxBd;
			status = XAxiDma_BdRingAlloc(txRing, 1+weights_bd_count, &startTxBd);
			if (status != XST_SUCCESS) {
				std::cout << "Could not allocate bd ring" << std::endl;
				return 1;
			}

			txBd = startTxBd;
			XAxiDma_BdSetCtrl(txBd, XAXIDMA_BD_CTRL_TXSOF_MASK);
			for (int weights_bd_index = 0; weights_bd_index < weights_bd_count; weights_bd_index++) {
				txBd = (XAxiDma_Bd *)XAxiDma_BdRingNext(txRing, txBd);
			}
			XAxiDma_BdSetCtrl(txBd, XAXIDMA_BD_CTRL_TXEOF_MASK);

			txBd = startTxBd;
			if (controls[x] & (1 << CTRL_STREAM_WEIGHTS)) {
				// second descriptor for weights when streaming
				txBd = (XAxiDma_Bd *)XAxiDma_BdRingNext(txRing, txBd);
			}

			// weights descriptor (weights)
			for (int weights_bd_index = 0; weights_bd_index < weights_bd_count; weights_bd_index++) {
				status = XAxiDma_BdSetBufAddr(txBd, (UINTPTR)(weights[x]+p*weightsVSize*weightsHSize+weights_bd_index*WEIGHTS_MAX_VSIZE*weightsHSize));
				if (status != XST_SUCCESS) {
					std::cout << "Could not set buffer address" << std::endl;
					return 1;
				}
				status = XAxiDma_BdSetLength(txBd, weightsHSize, txRing->MaxTransferLen);
				if (status != XST_SUCCESS) {
					std::cout << "Could not set hsize" << std::endl;
					return 1;
				}
				XAxiDma_BdSetARCache(txBd, 0x3); // cache default is 0b011
				XAxiDma_BdSetStride(txBd, weightsHSize);
				if (weights_bd_index < weights_bd_count-1) {
					XAxiDma_BdSetVSize(txBd, WEIGHTS_MAX_VSIZE);
				} else {
					XAxiDma_BdSetVSize(txBd, weightsVSize - (weights_bd_count-1)*WEIGHTS_MAX_VSIZE);
				}

				txBd = (XAxiDma_Bd *)XAxiDma_BdRingNext(txRing, txBd);
			}

			if (controls[x] & (1 << CTRL_STREAM_WEIGHTS)) {
				txBd = startTxBd;
			}

			// second descriptor (data)
			status = XAxiDma_BdSetBufAddr(txBd, (UINTPTR)txBuffer);
			if (status != XST_SUCCESS) {
				std::cout << "Could not set buffer address" << std::endl;
				return 1;
			}
			status = XAxiDma_BdSetLength(txBd, txHSize, txRing->MaxTransferLen);
			if (status != XST_SUCCESS) {
				std::cout << "Could not set hsize" << std::endl;
				return 1;
			}
			XAxiDma_BdSetARCache(txBd, 0x3); // cache default is 0b011
			XAxiDma_BdSetStride(txBd, txHSize);
			XAxiDma_BdSetVSize(txBd, txVSize);

			// TODO: fix rx descriptor hsize should be 4x for last layer
			// RX descriptors
			int rx_bd_count = (rxVSize + RX_MAX_VSIZE - 1) / RX_MAX_VSIZE;
			XAxiDma_Bd *rxBd, *startRxBd;
			status = XAxiDma_BdRingAlloc(rxRing, rx_bd_count, &startRxBd);
			if (status != XST_SUCCESS) {
				std::cout << "Could not allocate bd ring" << std::endl;
				return 1;
			}
			rxBd = startRxBd;

			for (int rx_bd_index = 0; rx_bd_index < rx_bd_count; rx_bd_index++) {
				status = XAxiDma_BdSetBufAddr(rxBd, (UINTPTR)(rxBuffer + rx_bd_index*RX_MAX_VSIZE*rxStride + p*rxHSize));
				if (status != XST_SUCCESS) {
					std::cout << "Could not set buffer address" << std::endl;
					return 1;
				}
				status = XAxiDma_BdSetLength(rxBd, rxHSize, rxRing->MaxTransferLen);
				if (status != XST_SUCCESS) {
					std::cout << "Could not set hsize" << std::endl;
					return 1;
				}

				XAxiDma_BdSetARCache(rxBd, 0x3); // cache default is 0b011
				XAxiDma_BdSetStride(rxBd, rxStride);
				if (rx_bd_index < rx_bd_count-1) {
					XAxiDma_BdSetVSize(rxBd, RX_MAX_VSIZE);
				} else {
					XAxiDma_BdSetVSize(rxBd, rxVSize - (rx_bd_count-1)*RX_MAX_VSIZE);
				}
				rxBd = (XAxiDma_Bd *)XAxiDma_BdRingNext(rxRing, rxBd);
			}

			// start receive
			status = XAxiDma_BdRingToHw(rxRing, rx_bd_count, startRxBd);
			if (status != XST_SUCCESS) {
				std::cout << "Could not start receive" << std::endl;
				return 1;
			}

			status = XAxiDma_BdRingStart(rxRing);
			if (status != XST_SUCCESS) {
				std::cout << "Could not start bd ring" << std::endl;
				return 1;
			}

			// start transmit
			status = XAxiDma_BdRingStart(txRing);
			if (status != XST_SUCCESS) {
				std::cout << "Could not start bd ring" << std::endl;
				return 1;
			}

			status = XAxiDma_BdRingToHw(txRing, 1+weights_bd_count, startTxBd);
			if (status != XST_SUCCESS) {
				std::cout << "Could not start transmit" << std::endl;
				return 1;
			}

//			XAxiDma_SimpleTransfer(&axiDma, (UINTPTR)rxBuffer, rxSize, XAXIDMA_DEVICE_TO_DMA);


//			XAxiDma_SimpleTransfer(&axiDma, (UINTPTR)weights[x], weightsSize, XAXIDMA_DMA_TO_DEVICE);
//			progress = XCnn_general_Get_progress(&cnnGeneral);
//			while (XAxiDma_Busy(&axiDma, XAXIDMA_DMA_TO_DEVICE)) {
//				progress = XCnn_general_Get_progress(&cnnGeneral);
//			}
//			progress = XCnn_general_Get_progress(&cnnGeneral);
//
//			XAxiDma_SimpleTransfer(&axiDma, (UINTPTR)txBuffer, txSize, XAXIDMA_DMA_TO_DEVICE);
			progress = XCnn_general_Get_progress(&cnnGeneral);
			while (XAxiDma_Busy(&axiDma, XAXIDMA_DMA_TO_DEVICE)) {
				progress = XCnn_general_Get_progress(&cnnGeneral);
			}
			progress = XCnn_general_Get_progress(&cnnGeneral);

			while (XAxiDma_Busy(&axiDma, XAXIDMA_DEVICE_TO_DMA));
			while (!XCnn_general_IsDone(&cnnGeneral)) {
				progress = XCnn_general_Get_progress(&cnnGeneral);
			}
			while (!XCnn_general_IsReady(&cnnGeneral)) {
				progress = XCnn_general_Get_progress(&cnnGeneral);
			}
			while (!XCnn_general_IsIdle(&cnnGeneral)) {
				progress = XCnn_general_Get_progress(&cnnGeneral);
			}

			// free tx bd
			int bdCount = XAxiDma_BdRingFromHw(txRing, XAXIDMA_ALL_BDS, &startTxBd);
			if (bdCount != 1+weights_bd_count) {
				std::cout << "Invalid number of bd" << std::endl;
				return 1;
			}
			status = XAxiDma_BdRingFree(txRing, bdCount, startTxBd);
			if (status != XST_SUCCESS) {
				std::cout << "Could not free bd ring" << std::endl;
				return 1;
			}

			// free rx bd
			bdCount = XAxiDma_BdRingFromHw(rxRing, XAXIDMA_ALL_BDS, &startRxBd);
			if (bdCount != rx_bd_count) {
				std::cout << "Invalid number of bd" << std::endl;
				return 1;
			}
			status = XAxiDma_BdRingFree(rxRing, bdCount, startRxBd);
			if (status != XST_SUCCESS) {
				std::cout << "Could not free bd ring" << std::endl;
				return 1;
			}

			XAxiDma_Reset(&axiDma);
		}

		Xil_DCacheInvalidateRange((INTPTR)rxBuffer, rxSize);

		// change output buffers
		uint8_t *tmp = rxBuffer;
		if (x == 0) {
			rxBuffer = align(BUFFER2, 4);
		} else {
			rxBuffer = txBuffer;
		}
		txBuffer = tmp;
		prev_stream_weights = !!(controls[x] & (1 << CTRL_STREAM_WEIGHTS));

		timer.stop();
		std::cout << "Layer " << x << " took: " << timer << std::endl;
	}

	int outSize = layer_sizes[n-1]*layer_sizes[n-1]*output_layers[n-1];
	if ((controls[n-1] & (1 << CTRL_MAXPOOL)) && !(controls[n-1] & (1 << CTRL_MAXPOOL1))) {
		outSize /= 4;
	}
	if (controls[n-1] & (1 << CTRL_STREAM_WEIGHTS)) {
		for (int i = 0; i < layer_sizes[n-1]*layer_sizes[n-1]; i++) {
			for (int o = 0; o < output_layers[n-1]; o++) {
				uint32_t uintv = *((uint32_t *)&txBuffer[(o*layer_sizes[n-1]*layer_sizes[n-1] + i)*3]);
				uintv <<= 8;
				int32_t intv = *((int32_t *) &uintv);
				intv >>= 8;
				float v = intv;
				v /= (1 << FIXED_POINT_SHIFT);
				float vexp = expected[i*output_layers[n-1]+o];
				float adv = fabs(v-vexp);
				float radv = adv/fmax(fabs(v), fabs(vexp));
				if (adv > err && radv > rel_err) {
					std::cout << "Wrong output for " << i << ", " << o << ": " << v << ", should be: " << vexp << " diff: " << adv << " rel: " << radv << std::endl;
				}
			}
		}
	} else {
		for (int i = 0; i < outSize; i++) {
			uint32_t uintv = *((uint32_t *)&txBuffer[3*i]);
			uintv <<= 8;
			int32_t intv = *((int32_t *) &uintv);
			intv >>= 8;
			float v = intv;
			v /= (1 << FIXED_POINT_SHIFT);
			float vexp = expected[i];
			float adv = fabs(v - vexp);
			float radv = adv/fmax(fabs(v), fabs(vexp));
			if (adv > err && radv > rel_err) {
				std::cout << "Wrong output for " << i << ": " << v << ", should be: " << vexp << " diff: " << adv << " rel: " << radv << std::endl;
			}
		}
	}

	std::cout << "Finished" << std::endl;

	return 0;
}
