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
    // todo!
}

void TestArgs()
{
    const int argc = 7;
    const char *argv[argc];
    argv[0] = "ignore";
    argv[1] = "-qi";
    argv[2] = "5";
    argv[3] = "--pi";
    argv[4] = "3.14";
    argv[5] = "-f";
    argv[6] = "file.txt";

    ArgsParser argsParser;
    argsParser.addFlag('w');
    argsParser.addFlag('q');
    argsParser.addInt('i');
    argsParser.addInt('s', "six", 6);
    argsParser.addInt("unnamedInt");
    argsParser.addInt("unnamedInt2", 2);
    argsParser.addDouble('p', "pi");
    argsParser.addDouble('x', "chi", 2.12);
    argsParser.addDouble("unnamedDouble");
    argsParser.addDouble("unnamedDouble2", 3.14);
    argsParser.addString('f', "infile");
    argsParser.addString('o', "outfile", "nothing.txt");
    argsParser.addString("unnamedString");
    argsParser.addString("unnamedString2", "string!");
    argsParser.parse(argc, argv);

    NNAssert(!argsParser.hasOpt("unnamedInt"), "ArgsParser::hasOpt(string) failed for unset option!");
    NNAssert(argsParser.hasOpt('s'), "ArgsParser::hasOpt(char) failed for set option!");
    NNAssert(!argsParser.getFlag('w'), "ArgsParser::getFlag failed for unset flag!");
    NNAssert(argsParser.getFlag('q'), "ArgsParser::getFlag failed for set flag!");
    NNAssertEquals(argsParser.getInt('i'), 5, "ArgsParser::getInt(char) failed!");
    NNAssertEquals(argsParser.getInt("unnamedInt2"), 2, "ArgsParser::getInt(string) failed!");
    NNAssertEquals(argsParser.getDouble('x'), 2.12, "ArgsParser::getDouble(char) failed!");
    NNAssertEquals(argsParser.getDouble("pi"), 3.14, "ArgsParser::getDouble(string) failed!");
    NNAssertEquals(argsParser.getString('f'), "file.txt", "ArgsParser::getString(char) failed!");
    NNAssertEquals(argsParser.getString("outfile"), "nothing.txt", "ArgsParser::getString(string) failed!");

    ArgsParser two;
    two.addFlag('q');
    two.addFlag('i');
    argv[2] = "-h";

    {
        std::stringstream ss;
        two.parse(2, argv + 1, false, ss);
        NNAssertNotEquals(ss.str(), "", "ArgsParser::parse with -h set failed!");
    }

    {
        std::stringstream ss;
        argsParser.printHelp(ss);
        NNAssertNotEquals(ss.str(), "", "ArgsParser::printHelp failed!");
    }

    {
        std::stringstream ss;
        argsParser.printOpts(ss);
        NNAssertNotEquals(ss.str(), "", "ArgsParser::printOpts failed!");
    }
}

#undef exit
