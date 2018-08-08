/*
 * Empty C++ Application
 */

#include <iostream>

#include "xtmrctr.h"
#include "xaxidma.h"
#include "xdma_test.h"

class Timer {
public:
	Timer() {
		XTmrCtr_Initialize(&axiTimer, XPAR_TMRCTR_0_DEVICE_ID);
	}

	void start() {
		XTmrCtr_Reset(&axiTimer, 0);
		startTime = XTmrCtr_GetValue(&axiTimer, 0);
		XTmrCtr_Start(&axiTimer, 0);
	}

	void stop() {
		XTmrCtr_Stop(&axiTimer, 0);
		endTime = XTmrCtr_GetValue(&axiTimer, 0);
	}

	uint32_t getValue() const {
		return endTime-startTime;
	}

	uint32_t getPeriod() const {
		return XPAR_AXI_TIMER_0_CLOCK_FREQ_HZ;
	}

private:
	XTmrCtr axiTimer;
	uint32_t startTime;
	uint32_t endTime;
};

std::ostream &operator<<(std::ostream &os, Timer const &m) {
	return os << m.getValue() << "/" << m.getPeriod();
}

// DMA addresses
#define MEM_BASE_ADDR 0x01000000
#define RX_BD     (MEM_BASE_ADDR + 0x00000000)
#define TX_BD     (MEM_BASE_ADDR + 0x00010000)
#define TX_BUFFER (MEM_BASE_ADDR + 0x00100000);
#define RX_BUFFER (MEM_BASE_ADDR + 0x00300000);

#define SIZE 24

int main()
{
	Timer timer;

	// INITIALIZE

	XDma_test dmaTest;
	XDma_test_Config *dmaTestConfig;
	XAxiDma axiDma;
	XAxiDma_Config *axiDmaConfig;

	dmaTestConfig = XDma_test_LookupConfig(XPAR_DMA_TEST_0_DEVICE_ID);
	if (dmaTestConfig) {
		int status = XDma_test_CfgInitialize(&dmaTest, dmaTestConfig);
		if (status != XST_SUCCESS) {
			std::cout << "Could not initialize dma test" << std::endl;
		}
	}

	axiDmaConfig = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
	if (axiDmaConfig) {
		int status = XAxiDma_CfgInitialize(&axiDma, axiDmaConfig);
		if (status != XST_SUCCESS) {
			std::cout << "Could not initialize axi dma" << std::endl;
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
	int status = XAxiDma_BdRingCreate(txRing, TX_BD, TX_BD, XAXIDMA_BD_MINIMUM_ALIGNMENT, 10);
	if (status != XST_SUCCESS) {
		std::cout << "Could not create bd ring" << std::endl;
		return 1;
	}

	status = XAxiDma_BdRingStart(txRing);
	if (status != XST_SUCCESS) {
		std::cout << "Could not start bd ring" << std::endl;
		return 1;
	}

	// rx ring init
	XAxiDma_BdRing *rxRing = XAxiDma_GetRxRing(&axiDma);
	status = XAxiDma_BdRingCreate(rxRing, RX_BD, RX_BD, XAXIDMA_BD_MINIMUM_ALIGNMENT, 10);
	if (status != XST_SUCCESS) {
		std::cout << "Could not create bd ring" << std::endl;
		return 1;
	}

	status = XAxiDma_BdRingStart(rxRing);
	if (status != XST_SUCCESS) {
		std::cout << "Could not start bd ring" << std::endl;
		return 1;
	}

	uint8_t *txBuffer = (uint8_t *)TX_BUFFER;
	uint8_t *rxBuffer = (uint8_t *)RX_BUFFER;

	// PREPARE DATA
	for (int i = 0; i < SIZE; i++) {
		txBuffer[i] = i%64;
		rxBuffer[i] = 0;
	}
	Xil_DCacheFlushRange((INTPTR)rxBuffer, SIZE);
	Xil_DCacheFlushRange((INTPTR)txBuffer, SIZE);

	std::cout << "Starting" << std::endl;

	// EXECUTE
	timer.start();

	XDma_test_Set_n(&dmaTest, SIZE/3);
	XDma_test_Start(&dmaTest);

	// TX descriptors
	XAxiDma_Bd *txBd, *startTxBd;
	status = XAxiDma_BdRingAlloc(txRing, 1, &startTxBd);
	if (status != XST_SUCCESS) {
		std::cout << "Could not allocate bd ring" << std::endl;
		return 1;
	}
	txBd = startTxBd;

	// first descriptor (weights)
	// TODO: fix weights for parts
	status = XAxiDma_BdSetBufAddr(txBd, (UINTPTR)txBuffer);
	if (status != XST_SUCCESS) {
		std::cout << "Could not set buffer address" << std::endl;
		return 1;
	}
	status = XAxiDma_BdSetLength(txBd, 4, txRing->MaxTransferLen);
	if (status != XST_SUCCESS) {
		std::cout << "Could not set hsize" << std::endl;
		return 1;
	}
	XAxiDma_BdSetCtrl(txBd, XAXIDMA_BD_CTRL_TXSOF_MASK|XAXIDMA_BD_CTRL_TXEOF_MASK); // first bd
	XAxiDma_BdSetARCache(txBd, 0x3); // cache default is 0b011
	XAxiDma_BdSetStride(txBd, 4);
	XAxiDma_BdSetVSize(txBd, 6);
	std::cout << "Length set to " << XAxiDma_BdGetLength(txBd, 0xff) << " mask is " << txRing->MaxTransferLen << std::endl;
	for (int i = 0; i < XAXIDMA_BD_NUM_WORDS; i++) {
		std::cout << i << ": " << (*txBd)[i] << std::endl;
	}

	// RX descriptors
	XAxiDma_Bd *rxBd, *startRxBd;
	status = XAxiDma_BdRingAlloc(rxRing, 1, &startRxBd);
	if (status != XST_SUCCESS) {
		std::cout << "Could not allocate bd ring" << std::endl;
		return 1;
	}
	rxBd = startRxBd;

	status = XAxiDma_BdSetBufAddr(rxBd, (UINTPTR)rxBuffer);
	if (status != XST_SUCCESS) {
		std::cout << "Could not set buffer address" << std::endl;
		return 1;
	}
	status = XAxiDma_BdSetLength(rxBd, SIZE, rxRing->MaxTransferLen);
	if (status != XST_SUCCESS) {
		std::cout << "Could not set hsize" << std::endl;
		return 1;
	}

	XAxiDma_BdSetARCache(rxBd, 0x3); // cache default is 0b011
	XAxiDma_BdSetStride(rxBd, SIZE);
	XAxiDma_BdSetVSize(rxBd, 1);


	// start receive
	status = XAxiDma_BdRingToHw(rxRing, 1, startRxBd);
	if (status != XST_SUCCESS) {
		std::cout << "Could not start receive" << std::endl;
		return 1;
	}

	// start transmit
	status = XAxiDma_BdRingToHw(txRing, 1, startTxBd);
	if (status != XST_SUCCESS) {
		std::cout << "Could not start transmit" << std::endl;
		return 1;
	}


//	XAxiDma_SimpleTransfer(&axiDma, (UINTPTR)txBuffer, SIZE, XAXIDMA_DMA_TO_DEVICE);
//	XAxiDma_SimpleTransfer(&axiDma, (UINTPTR)rxBuffer, SIZE, XAXIDMA_DEVICE_TO_DMA);

	while (XAxiDma_Busy(&axiDma, XAXIDMA_DEVICE_TO_DMA));
	while (!XDma_test_IsDone(&dmaTest));

	timer.stop();

	std::cout << "Timer took: " << timer << std::endl;

	// CHECK DATA
	Xil_DCacheInvalidateRange((INTPTR)rxBuffer, SIZE);
	for (int i = 0; i < SIZE; i++) {
		std::cout << i << ": " << (int)rxBuffer[i] << std::endl;
//		int expected = 2*(i%64);
//		if (i % 3 == 0) {
//			expected++;
//		}
//		if (rxBuffer[i] != expected) {
//			std::cout << "Invalid point at " << i << std::endl;
//		}
	}

	std::cout << "All data correct." << std::endl;

	return 0;
}
