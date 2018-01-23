#include "../test_serialized.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/serialization/serialized.hpp"
#include <vector>
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Serialized)
{
    NNTestMethod(Serialized)
    {
        NNTestParams()
        {
            NNTestEquals(Serialized().type(), Serialized::Null);
        }

        NNTestParams(const Serialized &)
        {
            Serialized s;
            s.set("null", nullptr);
            s.set("bool", true);
            s.set("int", 32);
            s.set("int", 42);
            s.set("double", 3.14);
            s.set("string", "nnlib");
            s.set("array", Serialized::Array);
            s.get("array").push("array_element");
            s.set("object", Serialized::Object);
            s.get("object").set("object_prop1", 3.14);
            s.get("object").set("object_prop2", "value");

            Serialized t(const_cast<const Serialized &>(s));
            NNTestEquals(t.type("null"), Serialized::Null);
            NNTestEquals(t.get<bool>("bool"), true);
            NNTestEquals(t.get<int>("int"), 42);
            NNTestAlmostEquals(t.get<double>("double"), 3.14, 1e-12);
            NNTestEquals(t.get<std::string>("string"), "nnlib");
            NNTestEquals(t.size("array"), 1);
            NNTestEquals(t.get("array").get<std::string>(0), "array_element");
            NNTestEquals(t.size("object"), 2);
            NNTestAlmostEquals(t.get("object").get<double>("object_prop1"), 3.14, 1e-12);
            NNTestEquals(t.get("object").get<std::string>("object_prop2"), "value");
        }

        NNTestParams(Serialized &)
        {
            Serialized s;
            s.set("null", nullptr);
            s.set("bool", true);
            s.set("int", 32);
            s.set("int", 42);
            s.set("double", 3.14);
            s.set("string", "nnlib");
            s.set("array", Serialized::Array);
            s.get("array").push("array_element");
            s.set("object", Serialized::Object);
            s.get("object").set("object_prop1", 3.14);
            s.get("object").set("object_prop2", "value");

            Serialized t(s);
            NNTestEquals(t.type("null"), Serialized::Null);
            NNTestEquals(t.get<bool>("bool"), true);
            NNTestEquals(t.get<int>("int"), 42);
            NNTestAlmostEquals(t.get<double>("double"), 3.14, 1e-12);
            NNTestEquals(t.get<std::string>("string"), "nnlib");
            NNTestEquals(t.size("array"), 1);
            NNTestEquals(t.get("array").get<std::string>(0), "array_element");
            NNTestEquals(t.size("object"), 2);
            NNTestAlmostEquals(t.get("object").get<double>("object_prop1"), 3.14, 1e-12);
            NNTestEquals(t.get("object").get<std::string>("object_prop2"), "value");
        }

        NNTestParams(Serialized &&)
        {
            Serialized s;
            s.set("null", nullptr);
            s.set("bool", true);
            s.set("int", 32);
            s.set("int", 42);
            s.set("double", 3.14);
            s.set("string", "nnlib");
            s.set("array", Serialized::Array);
            s.get("array").push("array_element");
            s.set("object", Serialized::Object);
            s.get("object").set("object_prop1", 3.14);
            s.get("object").set("object_prop2", "value");

            Serialized t(std::move(s));
            NNTestEquals(t.type("null"), Serialized::Null);
            NNTestEquals(t.get<bool>("bool"), true);
            NNTestEquals(t.get<int>("int"), 42);
            NNTestAlmostEquals(t.get<double>("double"), 3.14, 1e-12);
            NNTestEquals(t.get<std::string>("string"), "nnlib");
            NNTestEquals(t.size("array"), 1);
            NNTestEquals(t.get("array").get<std::string>(0), "array_element");
            NNTestEquals(t.size("object"), 2);
            NNTestAlmostEquals(t.get("object").get<double>("object_prop1"), 3.14, 1e-12);
            NNTestEquals(t.get("object").get<std::string>("object_prop2"), "value");
        }

        NNTestParams(Serialized::Type)
        {
            NNTestEquals(Serialized(Serialized::Null).type(), Serialized::Null);
            NNTestEquals(Serialized(Serialized::Boolean).type(), Serialized::Boolean);
            NNTestEquals(Serialized(Serialized::Integer).type(), Serialized::Integer);
            NNTestEquals(Serialized(Serialized::Float).type(), Serialized::Float);
            NNTestEquals(Serialized(Serialized::String).type(), Serialized::String);
            NNTestEquals(Serialized(Serialized::Array).type(), Serialized::Array);
            NNTestEquals(Serialized(Serialized::Object).type(), Serialized::Object);
        }

        NNTestParams(nullptr_t)
        {
            NNTestEquals(Serialized(nullptr).type(), Serialized::Null);
        }

        NNTestParams(bool)
        {
            NNTestEquals(Serialized(true).type(), Serialized::Boolean);
        }

        NNTestParams(int)
        {
            NNTestEquals(Serialized(42).type(), Serialized::Integer);
        }

        NNTestParams(double)
        {
            NNTestEquals(Serialized(3.14).type(), Serialized::Float);
        }

        NNTestParams(const char *)
        {
            NNTestEquals(Serialized("hello").type(), Serialized::String);
        }

        NNTestParams(T, const T &)
        {
            std::vector<int> v(10);
            v[0] = 5;

            Serialized s(v.begin(), v.end());
            NNTestEquals(s.type(), Serialized::Array);
            NNTestEquals(s.size(), 10);
            NNTestEquals(s.get<int>(0), 5);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Serialized &)
        {
            NNTestParams(Serialized &)
            {
                Serialized s, t;
                s.set("null", nullptr);
                s.set("bool", true);
                s.set("int", 32);
                s.set("int", 42);
                s.set("double", 3.14);
                s.set("string", "nnlib");
                s.set("array", Serialized::Array);
                s.get("array").push("array_element");
                s.set("object", Serialized::Object);
                s.get("object").set("object_prop1", 3.14);
                s.get("object").set("object_prop2", "value");

                t = s;
                NNTestEquals(t.type("null"), Serialized::Null);
                NNTestEquals(t.get<bool>("bool"), true);
                NNTestEquals(t.get<int>("int"), 42);
                NNTestAlmostEquals(t.get<double>("double"), 3.14, 1e-12);
                NNTestEquals(t.get<std::string>("string"), "nnlib");
                NNTestEquals(t.size("array"), 1);
                NNTestEquals(t.get("array").get<std::string>(0), "array_element");
                NNTestEquals(t.size("object"), 2);
                NNTestAlmostEquals(t.get("object").get<double>("object_prop1"), 3.14, 1e-12);
                NNTestEquals(t.get("object").get<std::string>("object_prop2"), "value");
            }

            NNTestParams(Serialized &&)
            {
                Serialized t(0);
                t = Serialized();
                NNTestEquals(t.type(), Serialized::Null);
                t = Serialized(true);
                NNTestEquals(t.get<bool>(), true);
                t = Serialized(-42);
                NNTestEquals(t.get<int>(), -42);
                t = Serialized(3.14);
                NNTestAlmostEquals(t.get<double>(), 3.14, 1e-12);
                t = Serialized("hello");
                NNTestEquals(t.get<std::string>(), "hello");
                t = Serialized(Serialized::Array);
                NNTestEquals(t.type(), Serialized::Array);
                NNTestEquals(t.size(), 0);
                t = Serialized(Serialized::Object);
                NNTestEquals(t.type(), Serialized::Object);
                NNTestEquals(t.size(), 0);
            }
        }

        NNTestMethod(type)
        {
            NNTestParams(Serialized::Type)
            {
                Serialized s;
                s.type(Serialized::Null);
                NNTestEquals(s.type(), Serialized::Null);
                s.type(Serialized::Boolean);
                NNTestEquals(s.type(), Serialized::Boolean);
                s.type(Serialized::Integer);
                NNTestEquals(s.type(), Serialized::Integer);
                s.type(Serialized::Float);
                NNTestEquals(s.type(), Serialized::Float);
                s.type(Serialized::String);
                NNTestEquals(s.type(), Serialized::String);
                s.type(Serialized::Array);
                NNTestEquals(s.type(), Serialized::Array);
                s.type(Serialized::Object);
                NNTestEquals(s.type(), Serialized::Object);
            }

            NNTestParams(size_t, Serialized::Type)
            {
                Serialized s;
                s.push(0);
                s.type(0, Serialized::Null);
                NNTestEquals(s.type(0), Serialized::Null);
                s.type(0, Serialized::Boolean);
                NNTestEquals(s.type(0), Serialized::Boolean);
                s.type(0, Serialized::Integer);
                NNTestEquals(s.type(0), Serialized::Integer);
                s.type(0, Serialized::Float);
                NNTestEquals(s.type(0), Serialized::Float);
                s.type(0, Serialized::String);
                NNTestEquals(s.type(0), Serialized::String);
                s.type(0, Serialized::Array);
                NNTestEquals(s.type(0), Serialized::Array);
                s.type(0, Serialized::Object);
                NNTestEquals(s.type(0), Serialized::Object);
            }

            NNTestParams(const std::string &, Serialized::Type)
            {
                Serialized s;
                s.set("foo", 0);
                s.type("foo", Serialized::Null);
                NNTestEquals(s.type("foo"), Serialized::Null);
                s.type("foo", Serialized::Boolean);
                NNTestEquals(s.type("foo"), Serialized::Boolean);
                s.type("foo", Serialized::Integer);
                NNTestEquals(s.type("foo"), Serialized::Integer);
                s.type("foo", Serialized::Float);
                NNTestEquals(s.type("foo"), Serialized::Float);
                s.type("foo", Serialized::String);
                NNTestEquals(s.type("foo"), Serialized::String);
                s.type("foo", Serialized::Array);
                NNTestEquals(s.type("foo"), Serialized::Array);
                s.type("foo", Serialized::Object);
                NNTestEquals(s.type("foo"), Serialized::Object);
            }
        }

        NNTestMethod(size)
        {
            NNTestParams()
            {
                Serialized s;
                NNTestEquals(s.size(), 1);
                s.push(1);
                s.push(2);
                NNTestEquals(s.size(), 2);
                s.set("foo", 1);
                s.set("bar", 2);
                s.set("baz", 3);
                NNTestEquals(s.size(), 3);
            }

            NNTestParams(size_t)
            {
                Serialized s;
                s.push(Serialized::Array);
                s.get(0).push(1);
                s.get(0).push(2);
                NNTestEquals(s.size(0), 2);
                s.get(0).set("foo", 1);
                s.get(0).set("bar", 2);
                s.get(0).set("baz", 3);
                NNTestEquals(s.size(0), 3);
            }

            NNTestParams(const std::string &)
            {
                Serialized s;
                s.set("foo", Serialized::Array);
                s.get("foo").push(1);
                s.get("foo").push(2);
                NNTestEquals(s.size("foo"), 2);
                s.get("foo").set("foo", 1);
                s.get("foo").set("bar", 2);
                s.get("foo").set("baz", 3);
                NNTestEquals(s.size("foo"), 3);
            }
        }

        NNTestMethod(resize)
        {
            NNTestParams(size_t)
            {
                Serialized s;
                s.resize(10);
                NNTestEquals(s.type(0), Serialized::Null);
                NNTestEquals(s.size(), 10);
            }
        }

        NNTestMethod(get<int>)
        {
            NNTestParams()
            {
                NNTestEquals(Serialized().get<int>(), 0);
                NNTestEquals(Serialized(true).get<int>(), 1);
                NNTestEquals(Serialized(2).get<int>(), 2);
                NNTestEquals(Serialized(3.14).get<int>(), 3);
                try
                {
                    Serialized("string").get<int>();
                    NNTest(false);
                }
                catch(const Error &)
                {}
            }

            NNTestParams(size_t)
            {
                Serialized s;
                const Serialized &t = s;
                s.push(12);
                NNTestEquals(s.get<int>(0), 12);
                NNTestEquals(t.get<int>(0), 12);
            }

            NNTestParams(const std::string &)
            {
                Serialized s;
                const Serialized &t = s;
                s.set("foo", -12);
                NNTestEquals(s.get<int>("foo"), -12);
                NNTestEquals(t.get<int>("foo"), -12);
            }
        }

        NNTestMethod(get<double>)
        {
            NNTestParams()
            {
                NNTestAlmostEquals(Serialized().get<double>(), 0, 1e-12);
                NNTestAlmostEquals(Serialized(true).get<double>(), 1, 1e-12);
                NNTestAlmostEquals(Serialized(2).get<double>(), 2, 1e-12);
                NNTestAlmostEquals(Serialized(3.14).get<double>(), 3.14, 1e-12);
                try
                {
                    Serialized("string").get<double>();
                    NNTest(false);
                }
                catch(const Error &)
                {}
            }
        }

        NNTestMethod(get<std::string>)
        {
            NNTestParams()
            {
                NNTestEquals(Serialized().get<std::string>(), "null");
                NNTestEquals(Serialized(false).get<std::string>(), "false");
                NNTestEquals(Serialized(true).get<std::string>(), "true");
                NNTestEquals(Serialized(2).get<std::string>(), "2");
                NNTestEquals(Serialized(3.14).get<std::string>(), "3.14");
                NNTestEquals(Serialized("string").get<std::string>(), "string");
                try
                {
                    Serialized(Serialized::Array).get<std::string>();
                    NNTest(false);
                }
                catch(const Error &)
                {}
            }
        }

        NNTestMethod(get<Storage>)
        {
            NNTestParams()
            {
                Storage<size_t> s = { 0, 1, 2, 3, 4, 5 };
                Storage<size_t> t = Serialized(s).get<Storage<size_t>>();
                NNTestEquals(s, t);
            }
        }

        NNTestMethod(get<const Serialized *>)
        {
            NNTestParams()
            {
                Serialized s;
                NNTestEquals(&s, s.get<const Serialized *>());
            }
        }

        NNTestMethod(get<Serialized *>)
        {
            NNTestParams()
            {
                Serialized s;
                NNTestEquals(&s, s.get<Serialized *>());
            }
        }

        NNTestMethod(get<const Serialized &>)
        {
            NNTestParams()
            {
                Serialized s;
                NNTestEquals(&s, &s.get<const Serialized &>());
            }
        }

        NNTestMethod(get<Serialized &>)
        {
            NNTestParams()
            {
                Serialized s;
                NNTestEquals(&s, &s.get<Serialized &>());
            }
        }

        NNTestMethod(get<Module *>)
        {
            NNTestParams()
            {
                Linear<T> orig(2, 3);

                auto copy = Serialized(orig).get<Module<T> *>();
                forEach([&](T orig, T copy)
                {
                    NNTestAlmostEquals(orig, copy, 1e-12);
                }, orig.params(), copy->params());
                delete copy;

                NNTestEquals(Serialized().get<Module<T> *>(), nullptr);
            }
        }

        NNTestMethod(get<Linear *>)
        {
            NNTestParams()
            {
                Linear<T> orig(2, 3);

                auto copy1 = Serialized(orig).get<Linear<T> *>();
                forEach([&](T orig, T copy)
                {
                    NNTestAlmostEquals(orig, copy, 1e-12);
                }, orig.params(), copy1->params());
                delete copy1;

                Serialized s;
                orig.save(s);
                auto copy2 = s.get<Linear<T> *>();
                forEach([&](T orig, T copy)
                {
                    NNTestAlmostEquals(orig, copy, 1e-12);
                }, orig.params(), copy2->params());
                delete copy2;

                NNTestEquals(Serialized().get<Linear<T> *>(), nullptr);
            }
        }

        NNTestMethod(get<T>)
        {
            NNTestParams(T, const T &)
            {
                Serialized s;
                s.push(0);
                s.push(1);
                s.push(2);
                s.push(3);
                s.push(4);
                s.push(5);

                std::vector<int> v(6);
                s.get(v.begin(), v.end());

                for(int i = 0; i < 6; ++i)
                    NNTestEquals(v[i], i);
            }

            NNTestParams(size_t, T, const T &)
            {
                Serialized s;
                s.push(Serialized::Array);
                s.get(0).push(0);
                s.get(0).push(1);
                s.get(0).push(2);
                s.get(0).push(3);
                s.get(0).push(4);
                s.get(0).push(5);

                std::vector<int> v(6);
                s.get(0, v.begin(), v.end());

                for(int i = 0; i < 6; ++i)
                    NNTestEquals(v[i], i);
            }

            NNTestParams(const std::string &, T, const T &)
            {
                Serialized s;
                s.set("foo", Serialized::Array);
                s.get("foo").push(0);
                s.get("foo").push(1);
                s.get("foo").push(2);
                s.get("foo").push(3);
                s.get("foo").push(4);
                s.get("foo").push(5);

                std::vector<int> v(6);
                s.get("foo", v.begin(), v.end());

                for(int i = 0; i < 6; ++i)
                    NNTestEquals(v[i], i);
            }
        }

        NNTestMethod(set)
        {
            NNTestParams(Serialized::Type)
            {
                Serialized s;
                s.set(Serialized::Null);
                NNTestEquals(s.type(), Serialized::Null);
                s.set(Serialized::Boolean);
                NNTestEquals(s.type(), Serialized::Boolean);
                s.set(Serialized::Integer);
                NNTestEquals(s.type(), Serialized::Integer);
                s.set(Serialized::Float);
                NNTestEquals(s.type(), Serialized::Float);
                s.set(Serialized::String);
                NNTestEquals(s.type(), Serialized::String);
                s.set(Serialized::Array);
                NNTestEquals(s.type(), Serialized::Array);
                s.set(Serialized::Object);
                NNTestEquals(s.type(), Serialized::Object);
            }

            NNTestParams(bool)
            {
                Serialized s;
                s.set(true);
                NNTestEquals(s.type(), Serialized::Boolean);
                NNTestEquals(s.get<bool>(), true);
                s.set(false);
                NNTestEquals(s.get<bool>(), false);
            }

            NNTestParams(int)
            {
                Serialized s;
                s.set(-12);
                NNTestEquals(s.type(), Serialized::Integer);
                NNTestEquals(s.get<int>(), -12);
            }

            NNTestParams(double)
            {
                Serialized s;
                s.set(3.14159);
                NNTestEquals(s.type(), Serialized::Float);
                NNTestEquals(s.get<double>(), 3.14159);
            }

            NNTestParams(const char *)
            {
                Serialized s;
                s.set("string");
                NNTestEquals(s.type(), Serialized::String);
                NNTestEquals(s.get<std::string>(), "string");
            }

            NNTestParams(Storage)
            {
                Storage<size_t> orig = { 0, 1, 2, 3, 4, 5 };
                Serialized s;
                s.set(orig);

                Storage<size_t> copy = s.get<Storage<size_t>>();
                for(int i = 0; i < 6; ++i)
                    NNTestEquals(copy[i], i);
            }

            NNTestParams(Linear)
            {
                Linear<T> orig(2, 3);
                Serialized s;
                s.set(orig);
                Linear<T> copy = s.get<Linear<T>>();
                forEach([&](T orig, T copy)
                {
                    NNTestAlmostEquals(orig, copy, 1e-12);
                }, orig.params(), copy.params());
            }

            NNTestParams(Module *)
            {
                Serialized s;
                Module<T> *m = nullptr;
                s.set(m);
                NNTestEquals(s.type(), Serialized::Null);

                m = new Linear<T>(2, 3);
                s.set(m);
                Linear<T> copy = s.get<Linear<T>>();
                forEach([&](T orig, T copy)
                {
                    NNTestAlmostEquals(orig, copy, 1e-12);
                }, m->params(), copy.params());

                delete m;
            }

            NNTestParams(nullptr_t)
            {
                Serialized s(12);
                s.set(nullptr);
                NNTestEquals(s.type(), Serialized::Null);
            }

            NNTestParams(const Serialized &)
            {
                Serialized s(12), t;
                t.set(s);
                NNTestEquals(t.type(), Serialized::Integer);
                NNTestEquals(t.get<int>(), 12);
            }

            NNTestParams(T, const T &)
            {
                std::vector<int> v(10);
                v[0] = 5;

                Serialized s;
                s.set(v.begin(), v.end());
                NNTestEquals(s.type(), Serialized::Array);
                NNTestEquals(s.size(), 10);
                NNTestEquals(s.get<int>(0), 5);
            }

            NNTestParams(size_t, int)
            {
                Serialized s;
                s.push(0);
                s.set(0, 12);
                NNTestEquals(s.get<int>(0), 12);
            }

            NNTestParams(const std::string &, int)
            {
                Serialized s;
                s.set("foo", 0);
                s.set("foo", -1.5);
                NNTestAlmostEquals(s.get<double>("foo"), -1.5, 1e-12);
            }
        }

        NNTestMethod(push)
        {
            NNTestParams(int)
            {
                Serialized s;
                s.push(12);
                NNTestEquals(s.type(), Serialized::Array);
                NNTestEquals(s.size(), 1);
                NNTestEquals(s.get<int>(0), 12);
            }

            NNTestParams(Serialized &)
            {
                Serialized s, t(12);
                s.push(t);
                NNTestEquals(s.type(), Serialized::Array);
                NNTestEquals(s.size(), 1);
                NNTestEquals(s.get<int>(0), 12);
            }

            NNTestParams(Serialized &)
            {
                Serialized s;
                s.push(Serialized(12));
                NNTestEquals(s.type(), Serialized::Array);
                NNTestEquals(s.size(), 1);
                NNTestEquals(s.get<int>(0), 12);
            }
        }

        NNTestMethod(pop)
        {
            NNTestParams()
            {
                Serialized s;
                s.push(12);
                s.push("string");
                Serialized t = s.pop();
                NNTestEquals(t.get<std::string>(), "string");
                t = s.pop();
                NNTestEquals(t.get<int>(), 12);
            }
        }

        NNTestMethod(has)
        {
            NNTestParams(const std::string &)
            {
                Serialized s;
                s.set("foo", 42);
                NNTestEquals(s.has("foo"), true);
                NNTestEquals(s.has("bar"), false);
            }
        }

        NNTestMethod(keys)
        {
            NNTestParams()
            {
                Serialized s;
                s.set("foo", 42);
                s.set("bar", 3.14);
                auto keys = s.keys();
                NNTestEquals(keys[0], "foo");
                NNTestEquals(keys[1], "bar");
            }
        }
    }
}
