#include "../test_args.hpp"
#include <sstream>

#define exit(x) ;
#include "nnlib/util/args.hpp"
#undef exit

using namespace nnlib;

NNTestClassImpl(Args)
{
    NNTestMethod(Args)
    {
        NNTestParams(int, const char **)
        {
            const char *argv[] = { "hello", "there" };
            Args args(2, argv);
            NNTestEquals(args.popString(), "hello");
            NNTestEquals(args.popString(), "there");
        }
    }

    NNTestMethod(unpop)
    {
        NNTestParams()
        {
            const char *argv[] = { "hello", "there" };
            Args args(2, argv);
            NNTestEquals(args.popString(), "hello");
            NNTestEquals(&args.unpop(), &args);
            NNTestEquals(args.popString(), "hello");
        }
    }

    NNTestMethod(unpop)
    {
        NNTestParams()
        {
            const char *argv[] = { "hello", "there" };
            Args args(2, argv);
            NNTestEquals(args.hasNext(), true);
            NNTestEquals(args.popString(), "hello");
            NNTestEquals(args.hasNext(), true);
            NNTestEquals(args.popString(), "there");
            NNTestEquals(args.hasNext(), false);
        }
    }

    NNTestMethod(ifPop)
    {
        NNTestParams(const std::string &)
        {
            const char *argv[] = { "hello", "there" };
            Args args(2, argv);
            NNTestEquals(args.ifPop("hello"), true);
            NNTestEquals(args.ifPop("hello"), false);
            NNTestEquals(args.ifPop("there"), true);
        }
    }

    NNTestMethod(nextIsNumber)
    {
        NNTestParams()
        {
            const char *argv[] = { "hello", "42", "-1", "3.14" };
            Args args(4, argv);
            NNTestEquals(args.nextIsNumber(), false);
            args.popString();
            NNTestEquals(args.nextIsNumber(), true);
            args.popString();
            NNTestEquals(args.nextIsNumber(), true);
            args.popString();
            NNTestEquals(args.nextIsNumber(), true);
        }
    }

    NNTestMethod(popString)
    {
        NNTestParams()
        {
            const char *argv[] = { "hello" };
            Args args(1, argv);
            NNTestEquals(args.popString(), "hello");
        }
    }

    NNTestMethod(popDouble)
    {
        NNTestParams()
        {
            const char *argv[] = { "3.14" };
            Args args(1, argv);
            NNTestAlmostEquals(args.popDouble(), 3.14, 1e-12);
        }
    }

    NNTestMethod(popInt)
    {
        NNTestParams()
        {
            const char *argv[] = { "42" };
            Args args(1, argv);
            NNTestEquals(args.popInt(), 42);
        }
    }
}

NNTestClassImpl(ArgsParser)
{
    NNTestMethod(ArgsParser)
    {
        NNTestParams(bool)
        {
            ArgsParser args(false);
            NNTestEquals(args.hasOpt('h'), false);
            NNTestEquals(args.hasOpt("help"), false);

            ArgsParser args2(true);
            NNTestEquals(args2.hasOpt('h'), true);
            NNTestEquals(args2.hasOpt("help"), true);
        }

        NNTestParams(char, std::string)
        {
            ArgsParser args('?', "helpOpt");
            NNTestEquals(args.hasOpt('?'), true);
            NNTestEquals(args.hasOpt("helpOpt"), true);
        }
    }

    NNTestMethod(addFlag)
    {
        NNTestParams(char, std::string)
        {
            ArgsParser args;
            args.addFlag('f', "flag");
            NNTestEquals(args.hasOpt('f'), true);
            NNTestEquals(args.hasOpt("flag"), true);
        }

        NNTestParams(std::string)
        {
            ArgsParser args;
            args.addFlag("flag");
            NNTestEquals(args.hasOpt("flag"), true);
        }
    }

    NNTestMethod(addInt)
    {
        NNTestParams(char, std::string)
        {
            ArgsParser args;
            args.addInt('i', "int");
            NNTestEquals(args.optName('i'), "int");
        }

        NNTestParams(char, std::string, int)
        {
            ArgsParser args;
            args.addInt('i', "int", 5);
            NNTestEquals(args.getInt('i'), 5);
            NNTestEquals(args.getInt("int"), 5);
        }

        NNTestParams(std::string)
        {
            const char *argv[] = { "cmd", "--int", "5" };
            ArgsParser args;
            args.addInt("int");
            args.parse(3, argv);
            NNTestEquals(args.getInt("int"), 5);
        }

        NNTestParams(char, int)
        {
            ArgsParser args;
            args.addInt("int", 5);
            NNTestEquals(args.getInt("int"), 5);
        }
    }

    NNTestMethod(addDouble)
    {
        NNTestParams(char, std::string)
        {
            ArgsParser args;
            args.addDouble('d', "double");
            NNTestEquals(args.optName('d'), "double");
        }

        NNTestParams(char, std::string, double)
        {
            ArgsParser args;
            args.addDouble('d', "double", 3.14);
            NNTestAlmostEquals(args.getDouble('d'), 3.14, 1e-12);
            NNTestAlmostEquals(args.getDouble("double"), 3.14, 1e-12);
        }

        NNTestParams(std::string)
        {
            const char *argv[] = { "cmd", "--double", "3.14" };
            ArgsParser args;
            args.addDouble("double");
            args.parse(3, argv);
            NNTestAlmostEquals(args.getDouble("double"), 3.14, 1e-12);
        }

        NNTestParams(char, double)
        {
            ArgsParser args;
            args.addDouble("double", 3.14);
            NNTestAlmostEquals(args.getDouble("double"), 3.14, 1e-12);
        }
    }

    NNTestMethod(addString)
    {
        NNTestParams(char, std::string)
        {
            ArgsParser args;
            args.addString('s', "string");
            NNTestEquals(args.optName('s'), "string");
        }

        NNTestParams(char, std::string, int)
        {
            ArgsParser args;
            args.addString('s', "string", "hello");
            NNTestEquals(args.getString('s'), "hello");
            NNTestEquals(args.getString("string"), "hello");
        }

        NNTestParams(std::string)
        {
            const char *argv[] = { "cmd", "--string", "hello" };
            ArgsParser args;
            args.addString("string");
            args.parse(3, argv);
            NNTestEquals(args.getString("string"), "hello");
        }

        NNTestParams(char, int)
        {
            ArgsParser args;
            args.addString("string", "hello");
            NNTestEquals(args.getString("string"), "hello");
        }
    }

    NNTestMethod(parse)
    {
        NNTestParams(int, const char **, bool, std::ostream &)
        {
            const char *argv[] = { "cmd", "-a", "opt", "-bc", "-5", "--dee", "3.14" };

            ArgsParser args;
            args.addString('a', "ayy");
            args.addFlag('b', "bee");
            args.addInt('c', "cee");
            args.addDouble('d', "dee");
            args.parse(7, argv);

            NNTestEquals(args.getString('a'), "opt");
            NNTestEquals(args.getFlag('b'), true);
            NNTestEquals(args.getInt('c'), -5);
            NNTestAlmostEquals(args.getDouble('d'), 3.14, 1e-12);
        }
    }
}
