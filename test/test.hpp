#ifndef TEST_HPP
#define TEST_HPP

#include "nnlib/core/error.hpp"
#include <string>
#include <vector>

namespace nnlib
{

namespace test
{

class Test
{
public:
    static int &verbosity()
    {
        static int i;
        return i;
    }

    Test(const std::string &nnClass) :
        nnClass(nnClass)
    {}

    virtual void run() = 0;

protected:
    std::string nnClass;
    std::string nnMethod;
    std::string nnParams;
    std::vector<Error> nnErrors;
};

}

}

#define NNTestClassDecl(Class)               \
    namespace nnlib                          \
    {                                        \
        namespace test                       \
        {                                    \
            class Test##Class : public Test  \
            {                                \
            public:                          \
                Test##Class();               \
                virtual void run() override; \
            };                               \
        }                                    \
    }

#define NNTestClassImpl(Class)                \
    nnlib::test::Test##Class::Test##Class() : \
        nnlib::test::Test(#Class)             \
    {}                                        \
    void nnlib::test::Test##Class::run()

#define NNTestClass(Class) \
    NNTestClassDecl(Class) \
    NNTestClassImpl(Class)

#define NNTestMethod(Method) \
    nnMethod = #Method;      \
    if(verbosity() == 1) \
        std::cout << "\n\t" << nnMethod << "..." << std::flush;

#define NNTestParams(...)    \
    nnParams = #__VA_ARGS__; \
    if(verbosity() == 2) \
        std::cout << "\n\t" << nnMethod << "(" << nnParams << ")..." << std::flush;

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

#define NNRunTest(Class)                                          \
    try                                                           \
    {                                                             \
        std::cout << "Testing " << #Class << "..." << std::flush; \
        nnlib::test::Test##Class().run();                         \
        std::cout << " Passed!" << std::endl;                     \
    }                                                             \
    catch(const nnlib::Error &e)                                  \
    {                                                             \
        std::cerr << std::endl << "\t" << e.what() << std::endl;  \
        exit(1);                                                  \
    }

#endif
