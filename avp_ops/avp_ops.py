import time
import math
import inspect
import fractions
import decimal
import array
try:
	import Numpy
	__numpy_import__ = True
except ImportError:
	__numpy_import__ = False


__version__ = "1.5.2"  # don't change this to remove a warning


class OpTwo:
	"""two input custom operator"""
	def __init__(self, function, help_str="", help_dict=None):
		self.function = function
		self.help_str = help_str
		self.help_dict = help_dict

	def __ror__(self, other):
		return OpTwo(lambda x, self=self, other=other: self.function(other, x))

	def __or__(self, other):
		return self.function(other)

	def __rlshift__(self, other):
		return OpTwo(lambda x, self=self, other=other: self.function(other, x))

	def __rshift__(self, other):
		return self.function(other)

	def __rrshift__(self, other):
		return OpTwo(lambda x, self=self, other=other: self.function(other, x))

	def __lshift__(self, other):
		return self.function(other)

	def __rmod__(self, other):
		return OpTwo(lambda x, self=self, other=other: self.function(other, x))

	def __mod__(self, other):
		return self.function(other)

	def __rmatmul__(self, other):
		return OpTwo(lambda x, self=self, other=other: self.function(other, x))

	def __matmul__(self, other):
		return self.function(other)

	def __call__(self, value1, value2):
		return self.function(value1, value2)

	def help(self):
		if self.help_dict is not None:
			if self.help_str == "":
				data = inspect.getsource(self.function).replace("\t", "").replace("\n", "")
			else:
				data = inspect.getsource(self.function).replace("\t", "").replace("\n", "") + " <?> " + self.help_str
		else:
			data = self.help_dict["name"] + " - "
			try:
				data += self.help_dict["notes"] + " - "
			except KeyError:
				pass
			data += self.help_dict["type"] + " - "
			data += inspect.getsource(self.function).replace("\t", "").replace("\n", "")
		print(data)
		return data


class Op:
	"""one input custom operator"""
	def __init__(self, function, help_str="", help_dict=None):
		self.function = function
		self.help_str = help_str
		self.help_dict = help_dict

	def __ror__(self, other):
		return self.function(other)

	def __rlshift__(self, other):
		return self.function(other)

	def __rrshift__(self, other):
		return self.function(other)

	def __rmod__(self, other):
		return self.function(other)

	def __rmatmul__(self, other):
		return self.function(other)

	def __or__(self, other):
		return self.function(other)

	def __lshift__(self, other):
		return self.function(other)

	def __rshift__(self, other):
		return self.function(other)

	def __mod__(self, other):
		return self.function(other)

	def __lt__(self, other):
		return self.function(other)

	def __gt__(self, other):
		return self.function(other)

	def __matmul__(self, other):
		return self.function(other)

	def __call__(self, value1):
		return self.function(value1)

	def help(self):
		if self.help_dict is not None:
			if self.help_str == "":
				data = inspect.getsource(self.function).replace("\t", "").replace("\n", "")
			else:
				data = inspect.getsource(self.function).replace("\t", "").replace("\n", "") + " <?> " + self.help_str
		else:
			data = self.help_dict["name"] + " - "
			try:
				data += self.help_dict["notes"] + " - "
			except KeyError:
				pass
			data += self.help_dict["type"] + " - "
			data += inspect.getsource(self.function).replace("\t", "").replace("\n", "")
		print(data)
		return data


def bit_not(n):
	return (1 << n.bit_length()) - 1 - n


# i___ is iterator ___. the second is applied to the first, which should be a type that supports iteration
# z___ is zipped iterator ___. the iterators get applied element-wise, as opposed to sequentially


class I:
	I_help_dict = {
		"div": {"name": "div: division", "type": "iter to non-iter"},
		"rdiv": {"name": "rdiv: division, reverse", "type": "iter to non-iter"},
		"mul": {"name": "mul: multiplication", "type": "iter to non-iter"},
		"add": {"name": "add: addition", "type": "iter to non-iter"},
		"sub": {"name": "sub: subtraction", "type": "iter to non-iter"},
		"rsub": {"name": "rsub: subtraction, reverse", "type": "iter to non-iter"},
		"pwr": {"name": "pwr: power", "type": "iter to non-iter"},
		"rpwr": {"name": "rpwr: power, reverse", "type": "iter to non-iter"},
		"mod": {"name": "mod: modulo", "type": "iter to non-iter"},
		"rmod": {"name": "rmod: modulo, reverse", "type": "iter to non-iter"},
		"fmod": {"name": "fmod: fmod from math", "type": "iter to non-iter"},
		"rfmod": {"name": "rfmod: fmod from math, reversed", "type": "iter to non-iter"},
		"sign": {"name": "sign: sets sign", "type": "iter to non-iter"},
		"gcd": {"name": "gcd: greatest common denominator", "type": "iter to non-iter"},
		"log": {"name": "log: logarithm base n", "type": "iter to non-iter"},
		"rlog": {"name": "rlog: logarithm base n, reversed", "type": "iter to non-iter"},
		"atan2": {"name": "atan2: 2d atan", "type": "iter to non-iter"},
		"ratan2": {"name": "ratan2: 2d atan, reversed", "type": "iter to non-iter"},
		"hypot": {"name": "hypot: euclidean distance", "type": "iter to non-iter"},
		"rhypot": {"name": "rhypot: euclidean distance, reversed", "type": "iter to non-iter"},
		"avg": {"name": "avg: average", "type": "iter to non-iter", "notes": "second input is necessary but unused"},
		"fact": {"name": "fact: factorial (!)", "type": "iter to non-iter"},
		"repl": {"name": "repl: replace", "type": "iter to non-iter", "notes": "use a list or tuple with two items, first is looked for to be replaces, second is what replaces it"},
		"replm": {"name": "replm: multi replace", "type": "iter to non-iter", "notes": "use a dictionary with a key for the item to be found to replace, and it's key'd item is swapped in"},
		"set": {"name": "set: replace entire array", "type": "iter to non-iter"}
	}
	div = OpTwo(lambda x, y: [n / y for n in x], I_help_dict["div"])
	rdiv = OpTwo(lambda x, y: [y / n for n in x], I_help_dict["rdiv"])
	mul = OpTwo(lambda x, y: [n * y for n in x], I_help_dict["mul"])
	add = OpTwo(lambda x, y: [n + y for n in x], I_help_dict["add"])
	sub = OpTwo(lambda x, y: [n - y for n in x], I_help_dict["sub"])
	rsub = OpTwo(lambda x, y: [y - n for n in x], I_help_dict["rsub"])
	pwr = OpTwo(lambda x, y: [n ** y for n in x], I_help_dict["pwr"])
	rpwr = OpTwo(lambda x, y: [y ** n for n in x], I_help_dict["rpwr"])
	mod = OpTwo(lambda x, y: [n % y for n in x], I_help_dict["mod"])
	rmod = OpTwo(lambda x, y: [y % n for n in x], I_help_dict["rmod"])
	fmod = OpTwo(lambda x, y: [math.fmod(n, y) for n in x], I_help_dict["fmod"])
	rfmod = OpTwo(lambda x, y: [math.fmod(y, n) for n in x], I_help_dict["rfmod"])
	sign = OpTwo(lambda x, y: [math.copysign(n, y) for n in x], I_help_dict["sign"])
	gcd = OpTwo(lambda x, y: [math.gcd(n, y) for n in x], I_help_dict["gcd"])
	log = OpTwo(lambda x, y: [math.log(n, y) for n in x], I_help_dict["log"])
	rlog = OpTwo(lambda x, y: [math.log(y, n) for n in x], I_help_dict["rlog"])
	atan2 = OpTwo(lambda x, y: [math.atan2(n, y) for n in x], I_help_dict["atan2"])
	ratan2 = OpTwo(lambda x, y: [math.atan2(y, n) for n in x], I_help_dict["ratan2"])
	hypot = OpTwo(lambda x, y: [math.hypot(n, y) for n in x], I_help_dict["hypot"])
	rhypot = OpTwo(lambda x, y: [math.hypot(y, n) for n in x], I_help_dict["rhypot"])
	avg = OpTwo(lambda x, y: [(n + y) / 2 for n in x], I_help_dict["avg"])
	fact = Op(lambda x: [math.factorial(n) for n in x], I_help_dict["fact"])  # Single input
	repl = OpTwo(lambda x, y: [y[1] if n == y[0] else n for n in x], I_help_dict["repl"])
	replm = OpTwo(lambda x, y: [y[n] if n in y else n for n in x], I_help_dict["replm"])
	set = OpTwo(lambda x, y: [y] * len(x), I_help_dict["set"])
	types = Op(lambda x: [type(n) for n in x])  # Single input
	ca = OpTwo(lambda x, y: combine_any(x, y))

	class Div:
		Div_help_dict = {}
		addmul = OpTwo(lambda x, y: [(n + y) / (n * y) for n in x])
		addsub = OpTwo(lambda x, y: [(n + y) / (n - y) for n in x])
		addmod = OpTwo(lambda x, y: [(n + y) / (n % y) for n in x])
		addpwr = OpTwo(lambda x, y: [(n + y) / (n ** y) for n in x])
		adddiv = OpTwo(lambda x, y: [(n + y) / (n / y) for n in x])
		submul = OpTwo(lambda x, y: [(n - y) / (n * y) for n in x])
		subadd = OpTwo(lambda x, y: [(n - y) / (n + y) for n in x])
		submod = OpTwo(lambda x, y: [(n - y) / (n % y) for n in x])
		subpwr = OpTwo(lambda x, y: [(n - y) / (n ** y) for n in x])
		subdiv = OpTwo(lambda x, y: [(n - y) / (n / y) for n in x])
		muladd = OpTwo(lambda x, y: [(n * y) / (n + y) for n in x])
		mulsub = OpTwo(lambda x, y: [(n * y) / (n - y) for n in x])
		mulmod = OpTwo(lambda x, y: [(n * y) / (n % y) for n in x])
		mulpwr = OpTwo(lambda x, y: [(n * y) / (n ** y) for n in x])
		muldiv = OpTwo(lambda x, y: [(n * y) / (n / y) for n in x])
		modmul = OpTwo(lambda x, y: [(n % y) / (n * y) for n in x])
		modsub = OpTwo(lambda x, y: [(n % y) / (n - y) for n in x])
		modadd = OpTwo(lambda x, y: [(n % y) / (n + y) for n in x])
		modpwr = OpTwo(lambda x, y: [(n % y) / (n ** y) for n in x])
		moddiv = OpTwo(lambda x, y: [(n % y) / (n / y) for n in x])
		pwrmul = OpTwo(lambda x, y: [(n ** y) / (n * y) for n in x])
		pwrsub = OpTwo(lambda x, y: [(n ** y) / (n - y) for n in x])
		pwrmod = OpTwo(lambda x, y: [(n ** y) / (n % y) for n in x])
		pwradd = OpTwo(lambda x, y: [(n ** y) / (n + y) for n in x])
		pwrdiv = OpTwo(lambda x, y: [(n ** y) / (n / y) for n in x])
		divmul = OpTwo(lambda x, y: [(n / y) / (n * y) for n in x])
		divsub = OpTwo(lambda x, y: [(n / y) / (n - y) for n in x])
		divmod = OpTwo(lambda x, y: [(n / y) / (n % y) for n in x])
		divpwr = OpTwo(lambda x, y: [(n / y) / (n ** y) for n in x])
		divadd = OpTwo(lambda x, y: [(n / y) / (n + y) for n in x])

	class DInv:
		disub = OpTwo(lambda x, y: [1 / ((1 / n) - (1 / y)) for n in x])
		diadd = OpTwo(lambda x, y: [1 / ((1 / n) + (1 / y)) for n in x])
		dimul = OpTwo(lambda x, y: [1 / ((1 / n) * (1 / y)) for n in x])
		didiv = OpTwo(lambda x, y: [1 / ((1 / n) / (1 / y)) for n in x])
		dimod = OpTwo(lambda x, y: [1 / ((1 / n) % (1 / y)) for n in x])
		dipwr = OpTwo(lambda x, y: [1 / ((1 / n) ** (1 / y)) for n in x])
		disubr = OpTwo(lambda x, y: [1 / ((1 / y) - (1 / n)) for n in x])
		diaddr = OpTwo(lambda x, y: [1 / ((1 / y) + (1 / n)) for n in x])
		dimulr = OpTwo(lambda x, y: [1 / ((1 / y) * (1 / n)) for n in x])
		didivr = OpTwo(lambda x, y: [1 / ((1 / y) / (1 / n)) for n in x])
		dimodr = OpTwo(lambda x, y: [1 / ((1 / y) % (1 / n)) for n in x])
		dipwrr = OpTwo(lambda x, y: [1 / ((1 / y) ** (1 / n)) for n in x])

	class Bin:
		and_ = OpTwo(lambda x, y: [n & y for n in x])
		xor = OpTwo(lambda x, y: [n ^ y for n in x])
		xnor = OpTwo(lambda x, y: [bit_not(n ^ y) for n in x])
		xnor_uns = OpTwo(lambda x, y: [~(n ^ y) for n in x])
		or_ = OpTwo(lambda x, y: [n | y for n in x])
		ls = OpTwo(lambda x, y: [n << y for n in x])
		rs = OpTwo(lambda x, y: [n >> y for n in x])
		rls = OpTwo(lambda x, y: [y << n for n in x])
		rrs = OpTwo(lambda x, y: [y >> n for n in x])
		inv = Op(lambda x: [bit_not(n) for n in x])  # Single input
		inv_uns = Op(lambda x: [~n for n in x])  # Single input


class Z:
	div = OpTwo(lambda x, y: [a / b for a, b in zip(x, y)])
	rdiv = OpTwo(lambda x, y: [b / a for a, b in zip(x, y)])
	mul = OpTwo(lambda x, y: [a * b for a, b in zip(x, y)])
	add = OpTwo(lambda x, y: [a + b for a, b in zip(x, y)])
	sub = OpTwo(lambda x, y: [a - b for a, b in zip(x, y)])
	rsub = OpTwo(lambda x, y: [b - a for a, b in zip(x, y)])
	pwr = OpTwo(lambda x, y: [a ** b for a, b in zip(x, y)])
	rpwr = OpTwo(lambda x, y: [b ** a for a, b in zip(x, y)])
	mod = OpTwo(lambda x, y: [a % b for a, b in zip(x, y)])
	rmod = OpTwo(lambda x, y: [b % a for a, b in zip(x, y)])
	fmod = OpTwo(lambda x, y: [math.fmod(a, b) for a, b in zip(x, y)])
	rfmod = OpTwo(lambda x, y: [math.fmod(b, a) for a, b in zip(x, y)])
	addstr = OpTwo(lambda x, y: [float(str(a).split(".")[0] + str(b)) for a, b in zip(x, y)])
	sign = OpTwo(lambda x, y: [math.copysign(a, b) for a, b in zip(x, y)])
	gcd = OpTwo(lambda x, y: [math.gcd(a, b) for a, b in zip(x, y)])
	log = OpTwo(lambda x, y: [math.log(a, b) for a, b in zip(x, y)])
	rlog = OpTwo(lambda x, y: [math.log(b, a) for a, b in zip(x, y)])
	atan2 = OpTwo(lambda x, y: [math.atan2(a, b) for a, b in zip(x, y)])
	ratan2 = OpTwo(lambda x, y: [math.atan2(b, a) for a, b in zip(x, y)])
	hypot = OpTwo(lambda x, y: [math.hypot(a, b) for a, b in zip(x, y)])
	rhypot = OpTwo(lambda x, y: [math.hypot(b, a) for a, b in zip(x, y)])
	avg = OpTwo(lambda x, y: [(a + b) / 2 for a, b in zip(x, y)])
	intersect = OpTwo(lambda x, y: list(set(x).intersection(y)))
	types = Op(lambda x: [type(n) for n in x])  # Single input
	ca = OpTwo(lambda x, y: combine_any(x, y))

	class Div:
		addmul = OpTwo(lambda x, y: [(a + b) / (a * b) for a, b in zip(x, y)])
		addsub = OpTwo(lambda x, y: [(a + b) / (a - b) for a, b in zip(x, y)])
		addmod = OpTwo(lambda x, y: [(a + b) / (a % b) for a, b in zip(x, y)])
		addpwr = OpTwo(lambda x, y: [(a + b) / (a ** b) for a, b in zip(x, y)])
		adddiv = OpTwo(lambda x, y: [(a + b) / (a / b) for a, b in zip(x, y)])
		submul = OpTwo(lambda x, y: [(a - b) / (a * b) for a, b in zip(x, y)])
		subadd = OpTwo(lambda x, y: [(a - b) / (a + b) for a, b in zip(x, y)])
		submod = OpTwo(lambda x, y: [(a - b) / (a % b) for a, b in zip(x, y)])
		subpwr = OpTwo(lambda x, y: [(a - b) / (a ** b) for a, b in zip(x, y)])
		subdiv = OpTwo(lambda x, y: [(a - b) / (a / b) for a, b in zip(x, y)])
		muladd = OpTwo(lambda x, y: [(a * b) / (a + b) for a, b in zip(x, y)])
		mulsub = OpTwo(lambda x, y: [(a * b) / (a - b) for a, b in zip(x, y)])
		mulmod = OpTwo(lambda x, y: [(a * b) / (a % b) for a, b in zip(x, y)])
		mulpwr = OpTwo(lambda x, y: [(a * b) / (a ** b) for a, b in zip(x, y)])
		muldiv = OpTwo(lambda x, y: [(a * b) / (a / b) for a, b in zip(x, y)])
		modmul = OpTwo(lambda x, y: [(a % b) / (a * b) for a, b in zip(x, y)])
		modsub = OpTwo(lambda x, y: [(a % b) / (a - b) for a, b in zip(x, y)])
		modadd = OpTwo(lambda x, y: [(a % b) / (a + b) for a, b in zip(x, y)])
		modpwr = OpTwo(lambda x, y: [(a % b) / (a ** b) for a, b in zip(x, y)])
		moddiv = OpTwo(lambda x, y: [(a % b) / (a / b) for a, b in zip(x, y)])
		pwrmul = OpTwo(lambda x, y: [(a ** b) / (a * b) for a, b in zip(x, y)])
		pwrsub = OpTwo(lambda x, y: [(a ** b) / (a - b) for a, b in zip(x, y)])
		pwrmod = OpTwo(lambda x, y: [(a ** b) / (a % b) for a, b in zip(x, y)])
		pwradd = OpTwo(lambda x, y: [(a ** b) / (a + b) for a, b in zip(x, y)])
		pwrdiv = OpTwo(lambda x, y: [(a ** b) / (a / b) for a, b in zip(x, y)])
		divmul = OpTwo(lambda x, y: [(a / b) / (a * b) for a, b in zip(x, y)])
		divsub = OpTwo(lambda x, y: [(a / b) / (a - b) for a, b in zip(x, y)])
		divmod = OpTwo(lambda x, y: [(a / b) / (a % b) for a, b in zip(x, y)])
		divpwr = OpTwo(lambda x, y: [(a / b) / (a ** b) for a, b in zip(x, y)])
		divadd = OpTwo(lambda x, y: [(a / b) / (a + b) for a, b in zip(x, y)])

	class DInv:
		disub = OpTwo(lambda x, y: [1 / ((1 / a) - (1 / b)) for a, b in zip(x, y)])
		diadd = OpTwo(lambda x, y: [1 / ((1 / a) + (1 / b)) for a, b in zip(x, y)])
		dimul = OpTwo(lambda x, y: [1 / ((1 / a) * (1 / b)) for a, b in zip(x, y)])
		didiv = OpTwo(lambda x, y: [1 / ((1 / a) / (1 / b)) for a, b in zip(x, y)])
		dimod = OpTwo(lambda x, y: [1 / ((1 / a) % (1 / b)) for a, b in zip(x, y)])
		dipwr = OpTwo(lambda x, y: [1 / ((1 / a) ** (1 / b)) for a, b in zip(x, y)])
		disubr = OpTwo(lambda x, y: [1 / ((1 / b) - (1 / a)) for a, b in zip(x, y)])
		diaddr = OpTwo(lambda x, y: [1 / ((1 / b) + (1 / a)) for a, b in zip(x, y)])
		dimulr = OpTwo(lambda x, y: [1 / ((1 / b) * (1 / a)) for a, b in zip(x, y)])
		didivr = OpTwo(lambda x, y: [1 / ((1 / b) / (1 / a)) for a, b in zip(x, y)])
		dimodr = OpTwo(lambda x, y: [1 / ((1 / b) % (1 / a)) for a, b in zip(x, y)])
		dipwrr = OpTwo(lambda x, y: [1 / ((1 / b) ** (1 / a)) for a, b in zip(x, y)])

	class Bin:
		and_ = OpTwo(lambda x, y: [a & b for a, b in zip(x, y)])
		xor = OpTwo(lambda x, y: [a ^ b for a, b in zip(x, y)])
		xnor = OpTwo(lambda x, y: [bit_not(a ^ b) for a, b in zip(x, y)])
		xnor_uns = OpTwo(lambda x, y: [~(a ^ b) for a, b in zip(x, y)])
		or_ = OpTwo(lambda x, y: [a | b for a, b in zip(x, y)])
		ls = OpTwo(lambda x, y: [a << b for a, b in zip(x, y)])
		rs = OpTwo(lambda x, y: [a >> b for a, b in zip(x, y)])
		rls = OpTwo(lambda x, y: [b << a for a, b in zip(x, y)])
		rrs = OpTwo(lambda x, y: [b >> a for a, b in zip(x, y)])


avg = OpTwo(lambda x, y: (x + y) / 2)


class B:
	not_ = Op(lambda x: [not n for n in x])  # Single input

	class I:
		equ = OpTwo(lambda x, y: [n == y for n in x])
		nequ = OpTwo(lambda x, y: [n == y for n in x])
		not_ = Op(lambda x: [not n for n in x])  # Single input
		and_ = OpTwo(lambda x, y: [n and y for n in x])
		nand = OpTwo(lambda x, y: [not(n and y) for n in x])
		or_ = OpTwo(lambda x, y: [n or y for n in x])
		nor = OpTwo(lambda x, y: [not(n or y) for n in x])
		xor = OpTwo(lambda x, y: [n ^ y for n in x])
		xnor = OpTwo(lambda x, y: [not(n ^ y) for n in x])

	class Z:
		equ = OpTwo(lambda x, y: [a == b for a, b in zip(x, y)])
		nequ = OpTwo(lambda x, y: [not(a == b) for a, b in zip(x, y)])
		not_ = Op(lambda x: [not n for n in x])  # Single input
		and_ = OpTwo(lambda x, y: [a and b for a, b in zip(x, y)])
		nand = OpTwo(lambda x, y: [not(a and b) for a, b in zip(x, y)])
		or_ = OpTwo(lambda x, y: [a or b for a, b in zip(x, y)])
		nor = OpTwo(lambda x, y: [not(a or b) for a, b in zip(x, y)])
		xor = OpTwo(lambda x, y: [a ^ b for a, b in zip(x, y)])
		xnor = OpTwo(lambda x, y: [not(a ^ b) for a, b in zip(x, y)])


class N:
	types = Op(lambda x: [type(n) for n in x])  # Single input
	fact = Op(lambda x: [math.factorial(n) for n in x])  # Single input
	itype = Op(lambda x: [is_iter(n) for n in x])  # Single input
	mca = Op(lambda x: combine_any(*x))

	class Bin:
		inv = Op(lambda x: [bit_not(n) for n in x])  # Single input
		inv_uns = Op(lambda x: [~n for n in x])  # Single input


def combine(existing: dict, new: dict) -> dict:
	out = {}
	for key in existing.keys():
		out[key] = existing[key]
	for key in new.keys():
		out[key] = new[key]
	return out


def rsum(n, *m):  # mmmm efficiency
	print("no just use sum()")
	try:
		s = n + m[0]
		return rsum(s, *m[1:])
	except IndexError:
		return n


class Iter:
	@staticmethod
	def eq(other):
		t = type(other)
		if t == type:
			t = other
		if __numpy_import__:
			if (t == Numpy.ndarray) or (t == list) or (t == tuple) or (t == range) or (t == array.array) or (t == memoryview):
				return True
			else:
				return False
		else:
			if (t == list) or (t == tuple) or (t == range) or (t == array.array) or (t == memoryview):
				return True
			else:
				return False

	def __mod__(self, other):
		return self.eq(other)

	def __rmod__(self, other):
		return self.eq(other)

	def __matmul__(self, other):
		return self.eq(other)

	def __rmatmul__(self, other):
		return self.eq(other)

	def __or__(self, other):
		return self.eq(other)

	def __ror__(self, other):
		return self.eq(other)

	def __and__(self, other):
		return self.eq(other)

	def __rand__(self, other):
		return self.eq(other)

	def __eq__(self, other):
		return self.eq(other)

	@staticmethod
	def type(o):
		return o if o == type else type(o)

	def __call__(self, o):
		return self.type(o)


def is_iter(o):
	if o @ Iter():
		return type(Iter())
	else:
		return Iter().type(o)


def combine_any(*items, dict_key_add=""):
	items_types = items % N.itype  # returns list of type of each element in input list
	dict_key_add = dict_key_add.strip()
	set_items = set(items_types)
	set_list_append = {Iter, int, float, str, decimal.Decimal, fractions.Fraction, complex}
	set_dict_append = {dict, Iter, int, float, str, decimal.Decimal, fractions.Fraction, complex}
	set_str_append = {int, float, str, decimal.Decimal, fractions.Fraction, complex}
	set_num_append = {int, float, decimal.Decimal, fractions.Fraction, complex}
	if set_num_append >= set_items:
		return sum(items)
	elif set_str_append >= set_items:
		temp = ""
		for i, t in zip(items, items_types):
			if t is str:
				temp += i  # works for two strings
			else:
				temp += str(i)
		return temp
	elif (set_list_append >= set_items) and (Iter in set_items):
		temp = []
		for i, t in zip(items, items_types):
			if t is Iter:
				temp += i  # works for two lists
			else:
				temp.append(i)
		print(temp)
		return temp
	elif (set_dict_append >= set_items) and (dict in set_items):
		temp, ctr = dict(), 0
		new_items = []
		for i, t in zip(items, items_types):
			if t is Iter:
				new_items.append(*i)
			elif t is dict:
				temp = {**temp, **i}  # combines dictionaries
			else:
				new_items.append(i)
		new_types = new_items % N.itype  # returns list of type of each element in list
		for i, t in zip(new_items, new_types):
			if t is dict:
				temp = {**temp, **i}  # combines dictionaries
			else:
				temp["avp_ca_" + str(ctr) + dict_key_add] = i  # makes new dictionary key/value pairs for items not in a dictionary
			ctr += 1
		return temp


def dict_intersect(x, y, appendpair=("_l", "_r")):
	xk, yk, xy = list(x.keys()), list(y.keys()), {}
	for key in list(set(xk).intersection(yk)):
		xy[str(key) + appendpair[0]] = x.pop(key)
		xy[str(key) + appendpair[1]] = y.pop(key)
	return xy


def dict_symdiff(x, y):
	xk, yk, xy = list(x.keys()), list(y.keys()), {}
	for key in list(set(xk).symmetric_difference(yk)):
		xy[str(key)] = x.pop(key)
		xy[str(key)] = y.pop(key)
	return xy


def string_sub_l(string, remove):
	string = list(string)
	for i in range(len(remove)): string[i] = "\v" if string[i] == remove[i] else string[i]
	return "".join(string).replace("\v", "")


def string_sub_r(string, remove):
	string, remove = list(string)[::-1], remove[::-1]
	for i in range(len(remove)): string[i] = "\v" if string[i] == remove[i] else string[i]
	return "".join(string[::-1]).replace("\v", "")


def string_sub_m(x, y):
	for n in list(y.keys()):
		x = x.replace(n, y[n])
	return x


class D:
	append = OpTwo(lambda x, y: {**x, **y})
	combine = append
	intersect = OpTwo(lambda x, y: dict_intersect(x, y))
	and_ = intersect
	mutex = OpTwo(lambda x, y: dict_symdiff(x, y))
	ca = OpTwo(lambda x, y: combine_any(x, y))
	symdiff = mutex
	xor = mutex


class S:
	subt_l = OpTwo(lambda x, y: string_sub_l(x, y))
	subt_r = OpTwo(lambda x, y: string_sub_r(x, y))
	rsubt_l = OpTwo(lambda x, y: string_sub_l(x, y))
	rsubt_r = OpTwo(lambda x, y: string_sub_r(x, y))
	sub = OpTwo(lambda x, y: x.replace(y, ""))
	repl_t = OpTwo(lambda x, y: [x.replace(str(y[0]), str(y[1]))])
	repl_d = OpTwo(lambda x, y: string_sub_m(x, y))
	ca = OpTwo(lambda x, y: combine_any(x, y))
	repl = repl_t
	repl_m = repl_d


def timeme(method, total_var=None):
	def wrapper(*args, **kw):
		if not total_var.enable:
			return method(*args, **kw)
		start_time = int(round(time.time() * 1000))
		result = method(*args, **kw)
		end_time = int(round(time.time() * 1000))
		# print(end_time - start_time, 'ms')
		total_var.timer += (end_time - start_time)
		return result
	return wrapper


def log_fn(var):
	def log_decorator(func):
		def func_wrapper(*args, **kw):
			if not var.enable:
				return func(*args, **kw)
			print(var.count)
			var.count[func.__name__] += 1
			print(var.count)
			return func(*args, **kw)
		return func_wrapper
	return log_decorator


class TestCounter:
	def __init__(self, *args):
		self.enable = True
		self.count = {}
		if args:
			for n in args:
				self.count[n] = 0


'''
testtest = TestCounter("testing")


@log_fn(testtest)
def testing(n):
	print(n)


# testtest.count["testing"] = 0
for n in range(10):
	print("/-", n)
	testing(n)
	print("\\-", n)
	print()
print(testtest.count["testing"])
# '''

# print([2, 3, 4] | z_pwr | [3, 2, 4])

