#ifndef TEST_ERROR_H
#define TEST_ERROR_H

#include "nnlib/error.h"
#include "nnlib/tensor.h"
using namespace nnlib;

void TestError()
{
	Tensor<> tensor(3, 6, 9);
	
	Error e("file", "func", 123, tensor(0, 1, 2), tensor, std::string("failure"));
	try { throw e; }
	catch(const Error &) {}
	
	Error f("this is an error message");
	try { throw f; }
	catch(const Error &) {}
}

#endif
