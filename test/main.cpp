#include "critics/test_mse.h"

int main()
{
	int ret = 0;
	try
	{
		TestMSE();
		std::cout << "All tests passed!" << std::endl;
	}
	catch(const std::runtime_error &e)
	{
		std::cerr << e.what();
		ret = 1;
	}
	return ret;
}
