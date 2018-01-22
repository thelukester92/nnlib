#include "../test_args.hpp"
#include "nnlib/util/args.hpp"
#include <sstream>
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
            args.popString();
            NNTestEquals(args.nextIsNumber(), false);

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

        NNTestParams()
        {
            const char *argv[] = { "cmd", "42" };
            ArgsParser args;
            args.addInt();
            args.parse(2, argv);
            NNTestEquals(args.getInt(0), 42);
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

        NNTestParams()
        {
            const char *argv[] = { "cmd", "3.14" };
            ArgsParser args;
            args.addDouble();
            args.parse(2, argv);
            NNTestAlmostEquals(args.getDouble(0), 3.14, 1e-12);
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

        NNTestParams()
        {
            const char *argv[] = { "cmd", "string" };
            ArgsParser args;
            args.addString();
            args.parse(2, argv);
            NNTestEquals(args.getString(0), "string");
        }
    }

    NNTestMethod(parse)
    {
        NNTestParams(int, const char **, bool, std::ostream &)
        {
            const char *argv[] = { "cmd", "1", "2.1", "three", "-a", "opt", "-bc", "-5", "--dee", "3.14", "-e" };
            ArgsParser args;
            args.addInt();
            args.addDouble();
            args.addString();
            args.addString('a', "ayy");
            args.addFlag('b', "bee");
            args.addInt('c', "cee");
            args.addDouble('d', "dee");
            args.addFlag('e', "eee");
            args.addFlag('f');
            args.parse(11, argv);
            NNTestEquals(args.getInt(0), 1);
            NNTestAlmostEquals(args.getDouble(1), 2.1, 1e-12);
            NNTestEquals(args.getString(2), "three");
            NNTestEquals(args.getString('a'), "opt");
            NNTestEquals(args.getFlag('b'), true);
            NNTestEquals(args.getInt('c'), -5);
            NNTestAlmostEquals(args.getDouble('d'), 3.14, 1e-12);
            NNTestEquals(args.getFlag("eee"), true);
            NNTestEquals(args.getFlag('f'), false);

            std::stringstream ss;
            const char *argv2[] = { "cmd", "-h", "1", "2.1", "three" };
            ArgsParser args2;
            args2.addInt('i', "int", 32);
            args2.addDouble('d', "double", 3.14);
            args2.addString('s', "string", "string");
            args2.addFlag('f');
            args2.addFlag("flag");
            args2.addInt();
            args2.addDouble();
            args2.addString();
            args2.parse(5, argv2, true, ss);
            NNTestGreaterThan(ss.str().size(), 0);
            NNTestEquals(args2.getInt(0), 1);
            NNTestAlmostEquals(args2.getDouble(1), 2.1, 1e-12);
            NNTestEquals(args2.getString(2), "three");

            const char *argv3[] = { "cmd", "-3", "-4" };
            ArgsParser args3;
            args3.addFlag('3');
            args3.addInt();
            args3.parse(3, argv3);
            NNTestEquals(args3.getFlag('3'), true);
            NNTestEquals(args3.getInt(0), -4);

            const char *argv4[] = { "cmd", "--", "-3" };
            ArgsParser args4;
            args4.addFlag('3');
            args4.addInt();
            args4.parse(3, argv4);
            NNTestEquals(args4.getFlag('3'), false);
            NNTestEquals(args4.getInt(0), -3);

            const char *argv5[] = { "cmd", "-3" };
            ArgsParser args5;
            args5.addFlag('f');
            args5.addInt();
            args5.parse(2, argv5);
            NNTestEquals(args5.getInt(0), -3);

            const char *argv6[] = { "cmd", "--string" };
            ArgsParser args6;
            args6.addFlag('f', "flag");
            args6.addString();
            args6.parse(2, argv6);
            NNTestEquals(args6.getString(0), "--string");
        }
    }

    NNTestMethod(printHelp)
    {
        NNTestParams(std::ostream &)
        {
            const char *argv[] = { "cmd", "1", "2.1", "three" };
            std::stringstream ss;
            ArgsParser args;
            args.addInt();
            args.addDouble();
            args.addString();
            args.addInt('i', "int", 32);
            args.addDouble('d', "double", 3.14);
            args.addString('s', "string", "string");
            args.parse(4, argv);
            args.printHelp(ss);
            NNTestGreaterThan(ss.str().size(), 0);

            ArgsParser args2;
            args2.addFlag('f');
            args2.addInt();
            args2.addDouble();
            args2.addString();
            args2.parse(4, argv);
            args2.printHelp(ss);
        }
    }

    NNTestMethod(printOpts)
    {
        NNTestParams(std::ostream &)
        {
            const char *argv[] = { "cmd", "1", "2.1", "three" };
            std::stringstream ss;
            ArgsParser args;
            args.addInt();
            args.addDouble();
            args.addString();
            args.addInt('i', "int", 32);
            args.addDouble('d', "double", 3.14);
            args.addString('s', "string", "string");
            args.parse(4, argv);
            args.printOpts(ss);
            NNTestGreaterThan(ss.str().size(), 0);

            ArgsParser args2;
            args2.addFlag('f');
            args2.addInt();
            args2.addDouble();
            args2.addString();
            args2.parse(4, argv);
            args2.printOpts(ss);
        }
    }

    NNTestMethod(optName)
    {
        NNTestParams(char)
        {
            ArgsParser args;
            args.addFlag('f', "flag");
            args.addFlag('g');
            NNTestEquals(args.optName('f'), "flag");
            NNTestEquals(args.optName('g'), "g");
        }
    }
}
