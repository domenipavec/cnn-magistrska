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



#endif /* SRC_TIMER_H_ */
