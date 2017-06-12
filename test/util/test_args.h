#ifndef TEST_ARGS_H
#define TEST_ARGS_H

#define exit(x) ;

#include "nnlib/util/args.h"
#include <sstream>
using namespace nnlib;

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
	argsParser.addFlag('q');
	argsParser.addInt('i');
	argsParser.addInt('s', "six", 6);
	argsParser.addDouble('p', "pi");
	argsParser.addDouble('x', "chi", 2.12);
	argsParser.addString('f', "infile");
	argsParser.addString('o', "outfile", "nothing.txt");
	argsParser.parse(argc, argv);
	
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

#endif
