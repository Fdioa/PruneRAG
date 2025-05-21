import sys, inspect, types, unittest
from pyext import *

class TestPyExt(unittest.TestCase):
    def test_overload_argc(self):
        @overload.argc(1)
        def f(a): return 1
        @overload.argc(2)
        def f(a, b): return 2
        @overload.argc()
        def f(): return 0
        self.assertEqual(f(), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1, 2), 2)
        self.assertRaises(TypeError, f, 1, 2, 3)
        self.assertEqual(len(inspect.getargspec(f).args), 0)
    def test_overload_args(self):
        @overload.args(str, int)
        def f(a, b): return str, int
        @overload.args(int)
        def f(a): return int
        @overload.args(str)
        def f(a): return str
        @overload.args()
        def f(): return
        self.assertEqual(f(), None)
        self.assertEqual(f(0), int)
        self.assertEqual(f('s'), str)
        self.assertEqual(f('s', 0), (str, int))
        self.assertRaises(TypeError, f, 0, 's')
        self.assertEqual(len(inspect.getargspec(f).args), 0)
        class x(object):
            @overload.args(str, is_cls=True)
            def f(self, s): return 1
            @overload.args(int, is_cls=True)
            def f(self, i): return 2
        self.assertEqual(x().f('s'), 1)
        self.assertEqual(x().f(1), 2)
    def test_module(self):
        m = RuntimeModule('s', 'doc', x=1, f=2)
        self.assertEqual(m.x, 1)
        self.assertEqual(m.f, 2)
        self.assertTrue(isinstance(m, types.ModuleType))
        self.assertEqual(m.__doc__, 'doc')
        m2 = RuntimeModule.from_string('s', 'doc', 'a=7; b=6')
        self.assertEqual(m2.a, 7)
        self.assertEqual(m2.b, 6)
    def test_switch(self):
        with switch('x'):
            if case('x'): x = 4; case.quit()
            if case('b'): x = 2
            if case(1): x = 3
            if case('a'): x = 1
            if case('x'): x = 0
        self.assertEqual(x, 4)
        with switch(1):
            if case.default(): x = 7
        self.assertEqual(x, 7)
        with switch(2):
            if case(1,2): x = 9
        self.assertEqual(x, 9)
    def test_annot(self):
        @fannotate('r', a='a', b=1, c=2)
        def x(a, b, c): pass
        self.assertEqual(x.__annotations__, {'a': 'a', 'b': 1, 'c': 2, 'return': 'r'})
    def test_unpack(self):
        t = (1, 2, 3)
        self.assertEqual(safe_unpack(t,2), (1,2))
        self.assertEqual(safe_unpack(t,4), (1,2,3,None))
        self.assertEqual(safe_unpack(t,4,fill=0), (1,2,3,0))
    def test_assign(self):
        self.assertEqual(assign('x', 7), 7)
        self.assertEqual(x, 7)
        global f
        def f(): pass
        self.assertEqual(assign('f.__annotations__', {'a': 1}), {'a': 1})
        self.assertEqual(f.__annotations__, {'a': 1})
    def test_compare_and_swap(self):
        global v
        v = None
        compare_and_swap('v', None, 7)
        self.assertEqual(v, 7)
        compare_and_swap('v', None, 8)
        self.assertEqual(v, 7)
    if sys.version_info.major == 3:
        def test_overload_args_annot(self):
            def x(a, b): return 0
            x.__annotations__ = {'a': int, 'b': str}
            x = overload.args(None)(x)
            self.assertEqual(x(1, 's'), 0)
            self.assertRaises(TypeError, x, 1, 2)

if __name__ == '__main__':
    unittest.main()
