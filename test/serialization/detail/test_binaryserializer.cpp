#include "../test_binaryserializer.hpp"
#include "nnlib/serialization/binaryserializer.hpp"
#include "nnlib/serialization/serialized.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/nn/sequential.hpp"
#include "nnlib/nn/tanh.hpp"
using namespace nnlib;

#include <bitset>

void TestBinarySerializer()
{
    // basic serialization

    {
        Serialized s;
        s.set("library", "nnlib");
        s.set("awesome", true);
        s.set("notAwesome", false);
        s.set("number", 32);
        s.set("number", 42);
        s.set("nothing", Serialized::Null);
        s.set("emptyArray", Serialized::Array);
        s.set("emptyObject", Serialized::Object);
        s.set("nested", Serialized::Object);

        s.get("nested")->set("indented", true);
        s.get("nested")->set("powerLevel", "> 9000");

        {
            std::stringstream ss;
            BinarySerializer::write(s, ss);

            Serialized d = BinarySerializer::read(ss);

            NNAssertEquals(d.get<std::string>("library"), "nnlib", "BinarySerializer failed!");
            NNAssertEquals(d.get<bool>("awesome"), true, "BinarySerializer failed!");
            NNAssertEquals(d.get<bool>("notAwesome"), false, "BinarySerializer failed!");
            NNAssertEquals(d.get<size_t>("number"), 42, "BinarySerializer failed!");
            NNAssertAlmostEquals(d.get<float>("number"), 42.0, 1e-12, "BinarySerializer failed!");
            NNAssertEquals(d.type("nothing"), Serialized::Null, "BinarySerializer failed!");
            NNAssertEquals(d.size("emptyArray"), 0, "BinarySerializer failed!");
            NNAssertEquals(d.size("emptyObject"), 0, "BinarySerializer failed!");

            NNAssertEquals(d.get("nested")->get<bool>("indented"), true, "BinarySerializer failed!");
            NNAssertEquals(d.get("nested")->get<std::string>("powerLevel"), "> 9000", "BinarySerializer failed!");
        }
    }

    // neural network serialization

    {
        Sequential<NN_REAL_T> nn(
            new Linear<NN_REAL_T>(10, 5),
            new TanH<NN_REAL_T>(),
            new Linear<NN_REAL_T>(5, 10),
            new TanH<NN_REAL_T>()
        );

        std::stringstream ss;
        BinarySerializer::write(nn, ss);

        Sequential<NN_REAL_T> *deserialized = BinarySerializer::read(ss).get<Sequential<NN_REAL_T> *>();

        auto &p1 = nn.params();
        auto &p2 = deserialized->params();

        for(auto i = p1.begin(), j = p2.begin(), end = p1.end(); i != end; ++i, ++j)
        {
            NNAssertAlmostEquals(*i, *j, 1e-12, "BinarySerializer failed!");
        }

        delete deserialized;
    }
}
