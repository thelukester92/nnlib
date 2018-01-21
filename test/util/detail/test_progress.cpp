#include "../test_progress.hpp"
#include "nnlib/util/progress.hpp"
#include <sstream>
using namespace nnlib;

NNTestClassImpl(Progress)
{
    NNTestMethod(Progress)
    {
        NNTestParams(size_t, std::ostream &, size_t)
        {
            std::stringstream ss;
            Progress p(100, ss, 50);
            p.display(0);
            NNTestEquals(ss.str().size(), 62);

            std::stringstream ss2;
            Progress p2(100, ss2, 58);
            p2.display(0);
            NNTestEquals(ss2.str().size(), 70);
        }
    }

    NNTestMethod(display)
    {
        NNTestParams(size_t)
        {
            std::stringstream ss;
            Progress p(100, ss);
            p.display(0);
            NNTestEquals(ss.str().find('\n'), std::string::npos);
            p.display(50);
            NNTestEquals(ss.str().find('\n'), std::string::npos);
            p.display(100);
            NNTestNotEquals(ss.str().find('\n'), std::string::npos);

            std::stringstream ss2;
            Progress p2(0, ss2);
            p2.display(0);
            NNTestEquals(ss2.str().size(), 0);
        }
    }
}
