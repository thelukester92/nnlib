#ifndef TEST_PROGRESS_H
#define TEST_PROGRESS_H

#include "nnlib/util/progress.h"
#include <sstream>
using namespace nnlib;

void TestProgress()
{
	std::stringstream ss;
	Progress::display(10, 20, '\n', 50, ss);
	NNAssertNotEquals(ss.str(), "", "Progress::display did not output anything!");
}

#endif
