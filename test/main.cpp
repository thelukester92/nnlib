// force debugging asserts
#ifdef OPTIMIZE
	#warning Debugging asserts have been re-enabled for testing.
	#undef OPTIMIZE
#endif

#include "serialization/test_archive.h"

#include "test_storage.h"
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
#include "nn/test_logsoftmax.h"
#include "nn/test_lstm.h"
#include "nn/test_recurrent.h"
#include "nn/test_relu.h"
#include "nn/test_sequencer.h"
#include "nn/test_sequential.h"
#include "nn/test_tanh.h"
using namespace nnlib;

#include <iostream>
#include <string>
#include <initializer_list>
#include <tuple>
#include <functional>
using namespace std;

#define TEST(Class) { string("Testing ") + #Class, Test##Class }

int main()
{
	int ret = 0;
	
	initializer_list<pair<string, function<void()>>> tests = {
		// top level
		TEST(Storage),
		
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
		TEST(LogSoftMax),
		TEST(LSTM),
		TEST(Recurrent),
		TEST(ReLU),
		TEST(Sequencer),
		TEST(Sequential),
		TEST(TanH)
	};
	
	try
	{
		for(auto test : tests)
		{
			cout << test.first << "..." << flush;
			test.second();
			cout << " Passed!" << endl;
		}
		
		cout << "All tests passed!" << endl;
	}
	catch(const Error &e)
	{
		cerr << endl << "An error occurred:\n\t" << e.what() << endl;
		ret = 1;
	}
	return ret;
}
