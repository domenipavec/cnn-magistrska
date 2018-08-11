/*
 * timer.h
 *
 *  Created on: Jul 29, 2018
 *      Author: domen
 */

#ifndef SRC_TIMER_H_
#define SRC_TIMER_H_

#include "xtmrctr.h"

class Timer {
public:
	Timer() {
		XTmrCtr_Initialize(&axiTimer, XPAR_TMRCTR_0_DEVICE_ID);
		XTmrCtr_SetOptions(&axiTimer, 0, XTC_CASCADE_MODE_OPTION);
	}

	void start() {
		XTmrCtr_Reset(&axiTimer, 0);
		XTmrCtr_Reset(&axiTimer, 1);
		startTime = getTime();
		XTmrCtr_Start(&axiTimer, 0);
	}

	void stop() {
		XTmrCtr_Stop(&axiTimer, 0);
		endTime = getTime();
	}

	uint64_t getValue() const {
		return endTime-startTime;
	}

	uint32_t getPeriod() const {
		return XPAR_AXI_TIMER_0_CLOCK_FREQ_HZ;
	}

private:
	XTmrCtr axiTimer;
	uint64_t startTime;
	uint64_t endTime;

	uint64_t getTime() {
		uint64_t t = 0;
		t = XTmrCtr_GetValue(&axiTimer, 1);
		t <<= 32;
		t |= XTmrCtr_GetValue(&axiTimer, 0);
		return t;
	}
};

std::ostream &operator<<(std::ostream &os, Timer const &m) {
	return os << double(m.getValue()) / m.getPeriod();
}



#endif /* SRC_TIMER_H_ */
