#include "../test_module.hpp"
#include "../test_container.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestAbstractClassImpl(Container, Container<T>)
{
    NNRunAbstractTest(Module, Container, nnImpl.copy());

    NNTestMethod(forget)
    {
        NNTestParams()
        {
            for(size_t i = 0; i < nnImpl.components(); ++i)
            {
                Module<T> &component = *nnImpl.component(i);
                component.state().fill(i);
                auto oldState = component.state().copy();
                nnImpl.forget();
                forEach([&](T oldState, T newState)
                {
                    NNTestAlmostEquals(oldState, i, 1e-12);
                    NNTestAlmostEquals(newState, 0, 1e-12);
                }, oldState, component.state());
            }
        }
    }

    NNTestMethod(save)
    {
        NNTestParams(Serialized &)
        {
            Serialized s;
            nnImpl.save(s);
            auto components = s.get<Storage<Module<T> *>>("components");
            NNTestEquals(components.size(), nnImpl.components());
            for(size_t i = 0; i < nnImpl.components(); ++i)
            {
                forEach([&](T orig, T copy)
                {
                    NNTestAlmostEquals(orig, copy, 1e-12);
                }, nnImpl.component(i)->params(), components[i]->params());
            }
        }
    }

    NNTestMethod(add)
    {
        NNTestParams(Module *)
        {
            size_t count = nnImpl.components();
            auto last = nnImpl.remove(count - 1);
            NNTestEquals(count - 1, nnImpl.components());
            nnImpl.add(last);
            NNTestEquals(count, nnImpl.components());
        }
    }

    NNTestMethod(remove)
    {
        NNTestParams(Module *)
        {
            size_t count = nnImpl.components();
            auto last = nnImpl.remove(count - 1);
            NNTestEquals(count - 1, nnImpl.components());
            nnImpl.add(last);
            NNTestEquals(count, nnImpl.components());
        }
    }

    NNTestMethod(clear)
    {
        NNTestParams(Module *)
        {
            auto copy = (Container<T> *) nnImpl.copy();
            copy->clear();
            NNTestEquals(copy->components(), 0);
            delete copy;
        }
    }

    NNTestMethod(paramsList)
    {
        NNTestParams()
        {
            for(size_t i = 0; i < nnImpl.components(); ++i)
                nnImpl.component(i)->params().fill(1);
            forEach([&](T param)
            {
                NNTestAlmostEquals(param, 1, 1e-12);
            }, nnImpl.params());
        }
    }

    NNTestMethod(gradList)
    {
        NNTestParams()
        {
            for(size_t i = 0; i < nnImpl.components(); ++i)
                nnImpl.component(i)->grad().fill(1);
            forEach([&](T param)
            {
                NNTestAlmostEquals(param, 1, 1e-12);
            }, nnImpl.grad());
        }
    }

    NNTestMethod(stateList)
    {
        NNTestParams()
        {
            for(size_t i = 0; i < nnImpl.components(); ++i)
                nnImpl.component(i)->state().fill(1);
            forEach([&](T param)
            {
                NNTestAlmostEquals(param, 1, 1e-12);
            }, nnImpl.state());
        }
    }
}
