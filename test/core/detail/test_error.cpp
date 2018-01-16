#include "../test_error.hpp"
#include "nnlib/core/error.hpp"
using namespace nnlib;

struct test_struct {};

NNTestClassImpl(Error)
{
    NNTestMethod(Error)
    {
        NNTestParams(const std::string &)
        {
            Error e("reason");
            NNTestEquals(std::string(e.what()), "reason");
        }

        NNTestParams(const std::string &, const std::string &, int)
        {
            Error e("file", "func", 0);
            NNTestEquals(std::string(e.what()), "file:0 (func): ");
        }
    }

    NNTestMethod(stringify)
    {
        NNTestParams(int)
        {
            NNTestEquals(Error::stringify(1), "1");
        }

        NNTestParams(test_struct)
        {
            NNTestEquals(Error::stringify(test_struct()), "object");
        }

        NNTestParams(const std::string &)
        {
            NNTestEquals(Error::stringify(std::string("test")), "test");
        }

        NNTestParams(const char *)
        {
            NNTestEquals(Error::stringify("test"), "test");
        }

        NNTestParams(nullptr_t)
        {
            NNTestEquals(Error::stringify(nullptr), "null");
        }

        NNTestParams()
        {
            NNTestEquals(Error::stringify(), "");
        }
    }
}
