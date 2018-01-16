#ifndef TEST_HPP
#define TEST_HPP

#include "nnlib/core/error.hpp"
#include <string>

namespace nnlib
{

class Test
{
public:
    Test(const std::string &nnClass) :
        nnClass(nnClass)
    {}

    virtual void run() = 0;

protected:
    std::string nnClass;
    std::string nnMethod;
    std::string nnParams;
};

}

#define NNTestClass(Class)               \
    namespace nnlib                      \
    {                                    \
        class Test##Class : public Test  \
        {                                \
        public:                          \
            Test##Class();               \
            virtual void run() override; \
        };                               \
    }                                    \
    nnlib::Test##Class::Test##Class() :  \
        nnlib::Test(#Class)              \
    {}                                   \
    void nnlib::Test##Class::run()

#define NNTestMethod(Method) \
    nnMethod = #Method;

#define NNTestParams(...) \
    nnParams = #__VA_ARGS__;

#define NNTest(...) \
    NNAssert(__VA_ARGS__, nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestEquals(...) \
    NNAssertEquals(__VA_ARGS__, nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestAlmostEquals(...) \
    NNAssertAlmostEquals(__VA_ARGS__, nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestNotEquals(...) \
    NNAssertNotEquals(__VA_ARGS__, nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestLessThan(...) \
    NNAssertLessThan(__VA_ARGS__, nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestLessThanOrEquals(...) \
    NNAssertLessThanOrEquals(__VA_ARGS__, nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestGreaterThan(...) \
    NNAssertGreaterThan(__VA_ARGS__, nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestGreaterThanOrEquals(...) \
    NNAssertGreaterThanOrEquals(__VA_ARGS__, nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#endif
