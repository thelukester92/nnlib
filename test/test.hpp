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
        nnPrefix(""),
        nnClass(nnClass)
    {}

    virtual void run() = 0;

protected:
    std::string nnPrefix;
    std::string nnClass;
    std::string nnMethod;
    std::string nnParams;
};

class AbstractTest
{
public:
    static int &verbosity()
    {
        return Test::verbosity();
    }

    AbstractTest(const std::string &nnPrefix) :
        nnPrefix(nnPrefix)
    {}

protected:
    std::string nnPrefix;
    std::string nnMethod;
    std::string nnParams;
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

#define NNTestAbstractClassDecl(Class, Impl)                        \
    namespace nnlib                                                 \
    {                                                               \
        namespace test                                              \
        {                                                           \
            class Test##Class : public AbstractTest                 \
            {                                                       \
            public:                                                 \
                Test##Class();                                      \
                void run(const std::string &nnClass, Impl &nnImpl); \
            };                                                      \
        }                                                           \
    }

#define NNTestAbstractClassImpl(Class, Impl)                  \
    nnlib::test::Test##Class::Test##Class() :                 \
        nnlib::test::AbstractTest(#Class + std::string("::")) \
    {}                                                        \
    void nnlib::test::Test##Class::run(const std::string &nnClass, Impl &nnImpl)

#define NNTestAbstractClass(Class) \
    NNTestAbstractClassDecl(Class) \
    NNTestAbstractClassImpl(Class)

#define NNTestMethod(Method) \
    nnMethod = #Method;      \
    if(verbosity() == 1)     \
        std::cout << "\n\t" << nnPrefix << nnMethod << "..." << std::flush;

#define NNTestParams(...)    \
    nnParams = #__VA_ARGS__; \
    if(verbosity() == 2)     \
        std::cout << "\n\t" << nnPrefix << nnMethod << "(" << nnParams << ")..." << std::flush;

#define NNTest(...) \
    NNAssert(__VA_ARGS__, nnPrefix + nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestEquals(...) \
    NNAssertEquals(__VA_ARGS__, nnPrefix + nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestAlmostEquals(...) \
    NNAssertAlmostEquals(__VA_ARGS__, nnPrefix + nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestNotEquals(...) \
    NNAssertNotEquals(__VA_ARGS__, nnPrefix + nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestLessThan(...) \
    NNAssertLessThan(__VA_ARGS__, nnPrefix + nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestLessThanOrEquals(...) \
    NNAssertLessThanOrEquals(__VA_ARGS__, nnPrefix + nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestGreaterThan(...) \
    NNAssertGreaterThan(__VA_ARGS__, nnPrefix + nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

#define NNTestGreaterThanOrEquals(...) \
    NNAssertGreaterThanOrEquals(__VA_ARGS__, nnPrefix + nnClass + "::" + nnMethod + "(" + nnParams + ") failed!");

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

#define NNRunAbstractTest(Base, ClassName, Impl)          \
    {                                                     \
        auto impl = Impl;                                 \
        nnlib::test::Test##Base().run(#ClassName, *impl); \
        delete impl;                                      \
    }

#endif
