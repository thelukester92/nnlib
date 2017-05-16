#include "critics/test_criticsequencer.h"
#include "critics/test_mse.h"
#include "critics/test_nll.h"

#include <iostream>
#include <string>
#include <initializer_list>
#include <tuple>
#include <functional>
using namespace std;

#define TEST(Class) { std::string("Testing ") + #Class, Test##Class }

int main()
{
	int ret = 0;
	
	initializer_list<pair<std::string, std::function<void()>>> tests = {
		TEST(CriticSequencer),
		TEST(MSE),
		TEST(NLL)
	};
	
	try
	{
		for(auto test : tests)
		{
			std::cout << test.first << "..." << std::flush;
			test.second();
			std::cout << " Passed!" << std::endl;
		}
		
		std::cout << "All tests passed!" << std::endl;
	}
	catch(const std::runtime_error &e)
	{
		std::cerr << e.what();
		ret = 1;
	}
	return ret;
}
