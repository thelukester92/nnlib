#include "../test_error.hpp"
#include "nnlib/core/error.hpp"
using namespace nnlib;

struct testError_struct {};

void testError_constructor()
{
    Error e("reason");
    NNAssertEquals(e.what(), std::string("reason"), "Error::Error(string) failed!");

    Error f("file", "func", 123, "test");
    NNAssertEquals(f.what(), std::string("file:123 (func): test"), "Error::Error(string, string, int, string) failed!");
}

void testError_stringify()
{
    NNAssertEquals(Error::stringify("hi", "there"), "hithere", "Error::stringify(string, string) failed!");
    NNAssertEquals(Error::stringify(12), "12", "Error::stringify(int) failed!");
    NNAssertEquals(Error::stringify(testError_struct()), "object", "Error::stringify(Tensor) failed!");
    NNAssertEquals(Error::stringify(std::string("hi")), "hi", "Error::stringify(string) failed!");
    NNAssertEquals(Error::stringify("hi"), "hi", "Error::stringify(const char *) failed!");
    NNAssertEquals(Error::stringify(nullptr), "null", "Error::stringify(nullptr_t) failed!");
}

void TestError()
{
    testError_constructor();
    testError_stringify();
}
