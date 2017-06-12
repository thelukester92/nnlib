#ifndef TEST_ERROR_H
#define TEST_ERROR_H

#include "nnlib/error.h"
using namespace nnlib;

void TestError()
{
	Error e("file", "func", 123);
}

#endif
