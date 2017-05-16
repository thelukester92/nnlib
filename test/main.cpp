#include "critics/test_mse.h"
#include "critics/test_nll.h"
#include <iostream>
using namespace std;

int main()
{
	int ret = 0;
	try
	{
		cout << "Testing MSE..." << endl;
		TestMSE();
		
		cout << "Testing NLL..." << endl;
		TestNLL();
		
		std::cout << "All tests passed!" << std::endl;
	}
	catch(const std::runtime_error &e)
	{
		std::cerr << e.what();
		ret = 1;
	}
	return ret;
}
