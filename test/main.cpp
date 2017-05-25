#include "test_tensor.h"
#include "critics/test_criticsequencer.h"
#include "critics/test_mse.h"
#include "critics/test_nll.h"
#include "math/test_math_base.h"
#include "math/test_math_blas.h"
#include "nn/test_batchnorm.h"
#include "nn/test_concat.h"
#include "nn/test_identity.h"
#include "nn/test_linear.h"
#include "nn/test_logistic.h"
#include "nn/test_relu.h"
#include "nn/test_tanh.h"

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
		// critics
		TEST(CriticSequencer),
		TEST(MSE),
		TEST(NLL),
		
		// math
		TEST(MathBase),
		TEST(MathBLAS),
		
		// nn
		TEST(BatchNorm),
		TEST(Concat),
		TEST(Identity),
		TEST(Linear),
		TEST(Logistic),
		TEST(ReLU),
		TEST(TanH)
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
