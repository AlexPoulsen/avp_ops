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


__version__ = "1.6.4"  # don't change this to remove a warning


def _mkdocstr(function, help_str, help_dict=None):
	"""
	_mkdocstr(callable, dict)
	_mkdocstr(callable, str, dict)
	"""
	if not isinstance(help_str, str):
		help_str, help_dict = "", help_str

	if help_dict is None:
		if help_str == "":
			data = inspect.getsource(function).replace("\t", "").replace("\n", "")
		else:
			data = inspect.getsource(function).replace("\t", "").replace("\n", "") + " <?> " + help_str
	else:
		data = help_dict["name"] + " - "
		try:
			data += help_dict["notes"] + " - "
		except KeyError:
			pass
		data += help_dict["type"] + " - "
		data += inspect.getsource(function).replace("\t", "").replace("\n", "")
	return data


class BaseOp:
	def __new__(cls, function, help_str="", help_dict=None):
		newcls = type(cls.__name__, (cls,), {'__doc__': _mkdocstr(function, help_str, help_dict)})
		# This bypasses the MRO, but encodes the assumption that we're calling object
		return object.__new__(newcls)

	def __init__(self, function, help_str="", help_dict=None):
		self.function = function
		self.help_str = help_str
		self.help_dict = help_dict


class OpTwo(BaseOp):
	"""two input custom operator"""
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
		data = _mkdocstr(self.function, self.help_str, self.help_dict)
		print(data)
		return data


class Op(BaseOp):
	"""one input custom operator"""
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
		data = _mkdocstr(self.function, self.help_str, self.help_dict)
		print(data)
		return data


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


class TestCounter:
	def __init__(self, *args):
		self.enable = True
		self.count = {}
		if args:
			for n in args:
				self.count[n] = 0


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


def bit_not(n):
	return (1 << n.bit_length()) - 1 - n


def safe_len(n, float_digits_split=False):
	try:
		return len(n)
	except TypeError:
		if float_digits_split:
			if type(n) == int:
				return len(str(n))
			elif (type(n) == decimal.Decimal) or (type(n) == float):
				try:
					s = str(n).split(".")
					return len(s[0]), len(s[1])
				except IndexError:
					s = str(n).split(".")
					return len(s[0]), 0
			elif type(n) == fractions.Fraction:
				s = str(n).split("/")
				return len(s[0]), len(s[1])
			else:
				return 1
		else:
			return 1


def is_iter(o):
	if o @ Iter():
		return type(Iter())
	else:
		return Iter().type(o)


def combine(existing: dict, new: dict) -> dict:
	out = {}
	for key in existing.keys():
		out[key] = existing[key]
	for key in new.keys():
		out[key] = new[key]
	return out


def combine_iters(*items) -> range:
	lens = items % N.mlen
	types = items % N.types
	total_len = sum(lens)
	pos = 0
	item = 0
	print(lens, types, total_len)
	for n in range(total_len):
		print(pos, item)
		if (types[item] != int) and (types[item] != float) and (types[item] != fractions.Fraction) and (types[item] != decimal.Decimal) and (types[item] != complex):
			yield items[item][pos]
		else:
			yield items[item]
		if (pos + 1) >= lens[item]:
			item += 1
			pos = 0
		else:
			pos += 1


def combine_any(*items, dict_key_add=""):
	items_types = items % N.itype  # returns list of type of each element in input list
	dict_key_add = dict_key_add.strip()
	set_items = set(items_types)
	set_list_append = {Iter, int, float, str, decimal.Decimal, fractions.Fraction, complex}
	set_dict_append = {dict, Iter, int, float, str, decimal.Decimal, fractions.Fraction, complex}
	set_str_append = {int, float, str, decimal.Decimal, fractions.Fraction, complex}
	set_num_append = {int, float, decimal.Decimal, fractions.Fraction, complex}
	if type(range) in items_types:
		return combine_iters(*items)
	else:
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


def curry(f, x):
	def curried_function(*args, **kw):
		return f(*((x, )+args), **kw)
	return curried_function


class I:
	"""Iterator and Non-Iterator operators/methods"""
	I_help_dictionary = {
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
	div = OpTwo(lambda x, y: [n / y for n in x], I_help_dictionary["div"])
	rdiv = OpTwo(lambda x, y: [y / n for n in x], I_help_dictionary["rdiv"])
	mul = OpTwo(lambda x, y: [n * y for n in x], I_help_dictionary["mul"])
	add = OpTwo(lambda x, y: [n + y for n in x], I_help_dictionary["add"])
	sub = OpTwo(lambda x, y: [n - y for n in x], I_help_dictionary["sub"])
	rsub = OpTwo(lambda x, y: [y - n for n in x], I_help_dictionary["rsub"])
	pwr = OpTwo(lambda x, y: [n ** y for n in x], I_help_dictionary["pwr"])
	rpwr = OpTwo(lambda x, y: [y ** n for n in x], I_help_dictionary["rpwr"])
	mod = OpTwo(lambda x, y: [n % y for n in x], I_help_dictionary["mod"])
	rmod = OpTwo(lambda x, y: [y % n for n in x], I_help_dictionary["rmod"])
	fmod = OpTwo(lambda x, y: [math.fmod(n, y) for n in x], I_help_dictionary["fmod"])
	rfmod = OpTwo(lambda x, y: [math.fmod(y, n) for n in x], I_help_dictionary["rfmod"])
	sign = OpTwo(lambda x, y: [math.copysign(n, y) for n in x], I_help_dictionary["sign"])
	gcd = OpTwo(lambda x, y: [math.gcd(n, y) for n in x], I_help_dictionary["gcd"])
	log = OpTwo(lambda x, y: [math.log(n, y) for n in x], I_help_dictionary["log"])
	rlog = OpTwo(lambda x, y: [math.log(y, n) for n in x], I_help_dictionary["rlog"])
	atan2 = OpTwo(lambda x, y: [math.atan2(n, y) for n in x], I_help_dictionary["atan2"])
	ratan2 = OpTwo(lambda x, y: [math.atan2(y, n) for n in x], I_help_dictionary["ratan2"])
	hypot = OpTwo(lambda x, y: [math.hypot(n, y) for n in x], I_help_dictionary["hypot"])
	rhypot = OpTwo(lambda x, y: [math.hypot(y, n) for n in x], I_help_dictionary["rhypot"])
	avg = OpTwo(lambda x, y: [(n + y) / 2 for n in x], I_help_dictionary["avg"])
	repl = OpTwo(lambda x, y: [y[1] if n == y[0] else n for n in x], I_help_dictionary["repl"])
	replm = OpTwo(lambda x, y: [y[n] if n in y else n for n in x], I_help_dictionary["replm"])
	set = OpTwo(lambda x, y: [y] * len(x), I_help_dictionary["set"])
	ca = OpTwo(lambda x, y: combine_any(x, y))
	isa = OpTwo(lambda x, y: [y.__class__ == n.__class__ for n in x])  # ADD MORE HELP INFO
	curry_r_sf = OpTwo(lambda x, y: [curry(y, n) for n in x])  # takes function on right and curries is with the inputs on the left, returns list of curried functions
	curry_l_sf = OpTwo(lambda x, y: [curry(x, n) for n in y])  # takes function on left and curries is with the inputs on the right, returns list of curried functions
	curry_r_mf = OpTwo(lambda x, y: [curry(n, x) for n in y])  # takes function on right and curries is with the inputs on the left, returns list of curried functions
	curry_l_mf = OpTwo(lambda x, y: [curry(n, y) for n in x])  # takes function on left and curries is with the inputs on the right, returns list of curried functions
	eval_r_sf = OpTwo(lambda x, y: [y(n) for n in x])  # takes function on right and curries is with the inputs on the left, returns list of curried functions
	eval_l_sf = OpTwo(lambda x, y: [x(n) for n in y])  # takes function on left and curries is with the inputs on the right, returns list of curried functions
	eval_r_mf = OpTwo(lambda x, y: [n(x) for n in y])  # takes function on right and curries is with the inputs on the left, returns list of curried functions
	eval_l_mf = OpTwo(lambda x, y: [n(y) for n in x])  # takes function on left and curries is with the inputs on the right, returns list of curried functions
	eval_ni = Op(lambda x: [n() for n in x])

	class Div:
		"""A series of operators involving two parenthesis groups divided, using a variety of operators on each side"""
		Div_help_dict = {}
		add_mul = OpTwo(lambda x, y: [(n + y) / (n * y) for n in x])
		add_sub = OpTwo(lambda x, y: [(n + y) / (n - y) for n in x])
		add_subr = OpTwo(lambda x, y: [(n + y) / (y - n) for n in x])
		add_mod = OpTwo(lambda x, y: [(n + y) / (n % y) for n in x])
		add_modr = OpTwo(lambda x, y: [(n + y) / (y % n) for n in x])
		add_pwr = OpTwo(lambda x, y: [(n + y) / (n ** y) for n in x])
		add_pwrr = OpTwo(lambda x, y: [(n + y) / (y ** n) for n in x])
		add_div = OpTwo(lambda x, y: [(n + y) / (n / y) for n in x])
		add_divr = OpTwo(lambda x, y: [(n + y) / (y / n) for n in x])
		sub_mul = OpTwo(lambda x, y: [(n - y) / (n * y) for n in x])
		sub_add = OpTwo(lambda x, y: [(n - y) / (n + y) for n in x])
		sub_mod = OpTwo(lambda x, y: [(n - y) / (n % y) for n in x])
		sub_modr = OpTwo(lambda x, y: [(n - y) / (y % n) for n in x])
		sub_pwr = OpTwo(lambda x, y: [(n - y) / (n ** y) for n in x])
		sub_pwrr = OpTwo(lambda x, y: [(n - y) / (y ** n) for n in x])
		sub_div = OpTwo(lambda x, y: [(n - y) / (n / y) for n in x])
		sub_divr = OpTwo(lambda x, y: [(n - y) / (y / n) for n in x])
		mul_add = OpTwo(lambda x, y: [(n * y) / (n + y) for n in x])
		mul_sub = OpTwo(lambda x, y: [(n * y) / (n - y) for n in x])
		mul_subr = OpTwo(lambda x, y: [(n * y) / (y - n) for n in x])
		mul_mod = OpTwo(lambda x, y: [(n * y) / (n % y) for n in x])
		mul_modr = OpTwo(lambda x, y: [(n * y) / (y % n) for n in x])
		mul_pwr = OpTwo(lambda x, y: [(n * y) / (n ** y) for n in x])
		mul_pwrr = OpTwo(lambda x, y: [(n * y) / (y ** n) for n in x])
		mul_div = OpTwo(lambda x, y: [(n * y) / (n / y) for n in x])
		mul_divr = OpTwo(lambda x, y: [(n * y) / (y / n) for n in x])
		mod_mul = OpTwo(lambda x, y: [(n % y) / (n * y) for n in x])
		mod_sub = OpTwo(lambda x, y: [(n % y) / (n - y) for n in x])
		mod_subr = OpTwo(lambda x, y: [(n % y) / (y - n) for n in x])
		mod_add = OpTwo(lambda x, y: [(n % y) / (n + y) for n in x])
		mod_pwr = OpTwo(lambda x, y: [(n % y) / (n ** y) for n in x])
		mod_pwrr = OpTwo(lambda x, y: [(n % y) / (y ** n) for n in x])
		mod_div = OpTwo(lambda x, y: [(n % y) / (n / y) for n in x])
		mod_divr = OpTwo(lambda x, y: [(n % y) / (y / n) for n in x])
		pwr_mul = OpTwo(lambda x, y: [(n ** y) / (n * y) for n in x])
		pwr_sub = OpTwo(lambda x, y: [(n ** y) / (n - y) for n in x])
		pwr_subr = OpTwo(lambda x, y: [(n ** y) / (y - n) for n in x])
		pwr_mod = OpTwo(lambda x, y: [(n ** y) / (n % y) for n in x])
		pwr_modr = OpTwo(lambda x, y: [(n ** y) / (y % n) for n in x])
		pwr_add = OpTwo(lambda x, y: [(n ** y) / (n + y) for n in x])
		pwr_div = OpTwo(lambda x, y: [(n ** y) / (n / y) for n in x])
		pwr_divr = OpTwo(lambda x, y: [(n ** y) / (y / n) for n in x])
		div_mul = OpTwo(lambda x, y: [(n / y) / (n * y) for n in x])
		div_sub = OpTwo(lambda x, y: [(n / y) / (n - y) for n in x])
		div_subr = OpTwo(lambda x, y: [(n / y) / (y - n) for n in x])
		div_mod = OpTwo(lambda x, y: [(n / y) / (n % y) for n in x])
		div_modr = OpTwo(lambda x, y: [(n / y) / (y % n) for n in x])
		div_pwr = OpTwo(lambda x, y: [(n / y) / (n ** y) for n in x])
		div_pwrr = OpTwo(lambda x, y: [(n / y) / (y ** n) for n in x])
		div_add = OpTwo(lambda x, y: [(n / y) / (n + y) for n in x])
		subr_mul = OpTwo(lambda x, y: [(y - n) / (n * y) for n in x])
		subr_add = OpTwo(lambda x, y: [(y - n) / (n + y) for n in x])
		subr_mod = OpTwo(lambda x, y: [(y - n) / (n % y) for n in x])
		subr_modr = OpTwo(lambda x, y: [(y - n) / (y % n) for n in x])
		subr_pwr = OpTwo(lambda x, y: [(y - n) / (n ** y) for n in x])
		subr_pwrr = OpTwo(lambda x, y: [(y - n) / (y ** n) for n in x])
		subr_div = OpTwo(lambda x, y: [(y - n) / (n / y) for n in x])
		subr_divr = OpTwo(lambda x, y: [(y - n) / (y / n) for n in x])
		modr_mul = OpTwo(lambda x, y: [(y % n) / (n * y) for n in x])
		modr_sub = OpTwo(lambda x, y: [(y % n) / (n - y) for n in x])
		modr_subr = OpTwo(lambda x, y: [(y % n) / (y - n) for n in x])
		modr_add = OpTwo(lambda x, y: [(y % n) / (n + y) for n in x])
		modr_pwr = OpTwo(lambda x, y: [(y % n) / (n ** y) for n in x])
		modr_pwrr = OpTwo(lambda x, y: [(y % n) / (y ** n) for n in x])
		modr_div = OpTwo(lambda x, y: [(y % n) / (n / y) for n in x])
		modr_divr = OpTwo(lambda x, y: [(y % n) / (y / n) for n in x])
		pwrr_mul = OpTwo(lambda x, y: [(y ** n) / (n * y) for n in x])
		pwrr_sub = OpTwo(lambda x, y: [(y ** n) / (n - y) for n in x])
		pwrr_subr = OpTwo(lambda x, y: [(y ** n) / (y - n) for n in x])
		pwrr_mod = OpTwo(lambda x, y: [(y ** n) / (n % y) for n in x])
		pwrr_modr = OpTwo(lambda x, y: [(y ** n) / (y % n) for n in x])
		pwrr_add = OpTwo(lambda x, y: [(y ** n) / (n + y) for n in x])
		pwrr_div = OpTwo(lambda x, y: [(y ** n) / (n / y) for n in x])
		pwrr_divr = OpTwo(lambda x, y: [(y ** n) / (y / n) for n in x])
		divr_mul = OpTwo(lambda x, y: [(y / n) / (n * y) for n in x])
		divr_sub = OpTwo(lambda x, y: [(y / n) / (n - y) for n in x])
		divr_subr = OpTwo(lambda x, y: [(y / n) / (y - n) for n in x])
		divr_mod = OpTwo(lambda x, y: [(y / n) / (n % y) for n in x])
		divr_modr = OpTwo(lambda x, y: [(y / n) / (y % n) for n in x])
		divr_pwr = OpTwo(lambda x, y: [(y / n) / (n ** y) for n in x])
		divr_pwrr = OpTwo(lambda x, y: [(y / n) / (y ** n) for n in x])
		divr_add = OpTwo(lambda x, y: [(y / n) / (n + y) for n in x])

	class DInv:
		"""A series of operators involving manipulating two inverted inputs then inverting again"""
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
		"""Binary related"""
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
	"""Iterator and Iterator operators/methods, zipped"""
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
	ca = OpTwo(lambda x, y: combine_any(x, y))
	isa = OpTwo(lambda x, y: [a.__class__ == b.__class__ for a, b in zip(x, y)])  # ADD MORE HELP INFO
	curry_r = OpTwo(lambda x, y: [curry(b, a) for a, b in zip(x, y)])  # takes function on right and curries is with the inputs on the left, returns list of curried functions
	curry_l = OpTwo(lambda x, y: [curry(a, b) for a, b in zip(x, y)])  # takes function on ;eft and curries is with the inputs on the right, returns list of curried functions
	curry = curry_l
	eval_l = OpTwo(lambda x, y: [a(b) for a, b in zip(x, y)])
	eval_r = OpTwo(lambda x, y: [b(a) for a, b in zip(x, y)])
	eval_ni = Op(lambda x: [n() for n in x])

	class Div:
		"""A series of operators involving two parenthesis groups divided, using a variety of operators on each side"""
		add_mul = OpTwo(lambda x, y: [(a + b) / (a * b) for a, b in zip(x, y)])
		add_sub = OpTwo(lambda x, y: [(a + b) / (a - b) for a, b in zip(x, y)])
		add_subr = OpTwo(lambda x, y: [(a + b) / (b - a) for a, b in zip(x, y)])
		add_mod = OpTwo(lambda x, y: [(a + b) / (a % b) for a, b in zip(x, y)])
		add_modr = OpTwo(lambda x, y: [(a + b) / (b % a) for a, b in zip(x, y)])
		add_pwr = OpTwo(lambda x, y: [(a + b) / (a ** b) for a, b in zip(x, y)])
		add_pwrr = OpTwo(lambda x, y: [(a + b) / (b ** a) for a, b in zip(x, y)])
		add_div = OpTwo(lambda x, y: [(a + b) / (a / b) for a, b in zip(x, y)])
		add_divr = OpTwo(lambda x, y: [(a + b) / (b / a) for a, b in zip(x, y)])
		sub_mul = OpTwo(lambda x, y: [(a - b) / (a * b) for a, b in zip(x, y)])
		sub_add = OpTwo(lambda x, y: [(a - b) / (a + b) for a, b in zip(x, y)])
		sub_mod = OpTwo(lambda x, y: [(a - b) / (a % b) for a, b in zip(x, y)])
		sub_modr = OpTwo(lambda x, y: [(a - b) / (b % a) for a, b in zip(x, y)])
		sub_pwr = OpTwo(lambda x, y: [(a - b) / (a ** b) for a, b in zip(x, y)])
		sub_pwrr = OpTwo(lambda x, y: [(a - b) / (b ** a) for a, b in zip(x, y)])
		sub_div = OpTwo(lambda x, y: [(a - b) / (a / b) for a, b in zip(x, y)])
		sub_divr = OpTwo(lambda x, y: [(a - b) / (b / a) for a, b in zip(x, y)])
		mul_add = OpTwo(lambda x, y: [(a * b) / (a + b) for a, b in zip(x, y)])
		mul_sub = OpTwo(lambda x, y: [(a * b) / (a - b) for a, b in zip(x, y)])
		mul_subr = OpTwo(lambda x, y: [(a * b) / (b - a) for a, b in zip(x, y)])
		mul_mod = OpTwo(lambda x, y: [(a * b) / (a % b) for a, b in zip(x, y)])
		mul_modr = OpTwo(lambda x, y: [(a * b) / (b % a) for a, b in zip(x, y)])
		mul_pwr = OpTwo(lambda x, y: [(a * b) / (a ** b) for a, b in zip(x, y)])
		mul_pwrr = OpTwo(lambda x, y: [(a * b) / (b ** a) for a, b in zip(x, y)])
		mul_div = OpTwo(lambda x, y: [(a * b) / (a / b) for a, b in zip(x, y)])
		mul_divr = OpTwo(lambda x, y: [(a * b) / (b / a) for a, b in zip(x, y)])
		mod_mul = OpTwo(lambda x, y: [(a % b) / (a * b) for a, b in zip(x, y)])
		mod_sub = OpTwo(lambda x, y: [(a % b) / (a - b) for a, b in zip(x, y)])
		mod_subr = OpTwo(lambda x, y: [(a % b) / (b - a) for a, b in zip(x, y)])
		mod_add = OpTwo(lambda x, y: [(a % b) / (a + b) for a, b in zip(x, y)])
		mod_pwr = OpTwo(lambda x, y: [(a % b) / (a ** b) for a, b in zip(x, y)])
		mod_pwrr = OpTwo(lambda x, y: [(a % b) / (b ** a) for a, b in zip(x, y)])
		mod_div = OpTwo(lambda x, y: [(a % b) / (a / b) for a, b in zip(x, y)])
		mod_divr = OpTwo(lambda x, y: [(a % b) / (b / a) for a, b in zip(x, y)])
		pwr_mul = OpTwo(lambda x, y: [(a ** b) / (a * b) for a, b in zip(x, y)])
		pwr_sub = OpTwo(lambda x, y: [(a ** b) / (a - b) for a, b in zip(x, y)])
		pwr_subr = OpTwo(lambda x, y: [(a ** b) / (b - a) for a, b in zip(x, y)])
		pwr_mod = OpTwo(lambda x, y: [(a ** b) / (a % b) for a, b in zip(x, y)])
		pwr_modr = OpTwo(lambda x, y: [(a ** b) / (b % a) for a, b in zip(x, y)])
		pwr_add = OpTwo(lambda x, y: [(a ** b) / (a + b) for a, b in zip(x, y)])
		pwr_div = OpTwo(lambda x, y: [(a ** b) / (a / b) for a, b in zip(x, y)])
		pwr_divr = OpTwo(lambda x, y: [(a ** b) / (b / a) for a, b in zip(x, y)])
		div_mul = OpTwo(lambda x, y: [(a / b) / (a * b) for a, b in zip(x, y)])
		div_sub = OpTwo(lambda x, y: [(a / b) / (a - b) for a, b in zip(x, y)])
		div_subr = OpTwo(lambda x, y: [(a / b) / (b - a) for a, b in zip(x, y)])
		div_mod = OpTwo(lambda x, y: [(a / b) / (a % b) for a, b in zip(x, y)])
		div_modr = OpTwo(lambda x, y: [(a / b) / (b % a) for a, b in zip(x, y)])
		div_pwr = OpTwo(lambda x, y: [(a / b) / (a ** b) for a, b in zip(x, y)])
		div_pwrr = OpTwo(lambda x, y: [(a / b) / (a ** b) for a, b in zip(x, y)])
		div_add = OpTwo(lambda x, y: [(a / b) / (a + b) for a, b in zip(x, y)])
		subr_mul = OpTwo(lambda x, y: [(b - a) / (a * b) for a, b in zip(x, y)])
		subr_add = OpTwo(lambda x, y: [(b - a) / (a + b) for a, b in zip(x, y)])
		subr_mod = OpTwo(lambda x, y: [(b - a) / (a % b) for a, b in zip(x, y)])
		subr_modr = OpTwo(lambda x, y: [(b - a) / (b % a) for a, b in zip(x, y)])
		subr_pwr = OpTwo(lambda x, y: [(b - a) / (a ** b) for a, b in zip(x, y)])
		subr_pwrr = OpTwo(lambda x, y: [(b - a) / (b ** a) for a, b in zip(x, y)])
		subr_div = OpTwo(lambda x, y: [(b - a) / (a / b) for a, b in zip(x, y)])
		subr_divr = OpTwo(lambda x, y: [(b - a) / (b / a) for a, b in zip(x, y)])
		modr_mul = OpTwo(lambda x, y: [(b % a) / (a * b) for a, b in zip(x, y)])
		modr_sub = OpTwo(lambda x, y: [(b % a) / (a - b) for a, b in zip(x, y)])
		modr_subr = OpTwo(lambda x, y: [(b % a) / (b - a) for a, b in zip(x, y)])
		modr_add = OpTwo(lambda x, y: [(b % a) / (a + b) for a, b in zip(x, y)])
		modr_pwr = OpTwo(lambda x, y: [(b % a) / (a ** b) for a, b in zip(x, y)])
		modr_pwrr = OpTwo(lambda x, y: [(b % a) / (b ** a) for a, b in zip(x, y)])
		modr_div = OpTwo(lambda x, y: [(b % a) / (a / b) for a, b in zip(x, y)])
		modr_divr = OpTwo(lambda x, y: [(b % a) / (b / a) for a, b in zip(x, y)])
		pwrr_mul = OpTwo(lambda x, y: [(b ** a) / (a * b) for a, b in zip(x, y)])
		pwrr_sub = OpTwo(lambda x, y: [(b ** a) / (a - b) for a, b in zip(x, y)])
		pwrr_subr = OpTwo(lambda x, y: [(b ** a) / (b - a) for a, b in zip(x, y)])
		pwrr_mod = OpTwo(lambda x, y: [(b ** a) / (a % b) for a, b in zip(x, y)])
		pwrr_modr = OpTwo(lambda x, y: [(b ** a) / (b % a) for a, b in zip(x, y)])
		pwrr_add = OpTwo(lambda x, y: [(b ** a) / (a + b) for a, b in zip(x, y)])
		pwrr_div = OpTwo(lambda x, y: [(b ** a) / (a / b) for a, b in zip(x, y)])
		pwrr_divr = OpTwo(lambda x, y: [(b ** a) / (b / a) for a, b in zip(x, y)])
		divr_mul = OpTwo(lambda x, y: [(b / a) / (a * b) for a, b in zip(x, y)])
		divr_sub = OpTwo(lambda x, y: [(b / a) / (a - b) for a, b in zip(x, y)])
		divr_subr = OpTwo(lambda x, y: [(b / a) / (b - a) for a, b in zip(x, y)])
		divr_mod = OpTwo(lambda x, y: [(b / a) / (a % b) for a, b in zip(x, y)])
		divr_modr = OpTwo(lambda x, y: [(b / a) / (b % a) for a, b in zip(x, y)])
		divr_pwr = OpTwo(lambda x, y: [(b / a) / (a ** b) for a, b in zip(x, y)])
		divr_pwrr = OpTwo(lambda x, y: [(b / a) / (a ** b) for a, b in zip(x, y)])
		divr_add = OpTwo(lambda x, y: [(b / a) / (a + b) for a, b in zip(x, y)])

	class DInv:
		"""A series of operators involving manipulating two inverted inputs then inverting again"""
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
		"""Binary related"""
		and_ = OpTwo(lambda x, y: [a & b for a, b in zip(x, y)])
		xor = OpTwo(lambda x, y: [a ^ b for a, b in zip(x, y)])
		xnor = OpTwo(lambda x, y: [bit_not(a ^ b) for a, b in zip(x, y)])
		xnor_uns = OpTwo(lambda x, y: [~(a ^ b) for a, b in zip(x, y)])
		or_ = OpTwo(lambda x, y: [a | b for a, b in zip(x, y)])
		ls = OpTwo(lambda x, y: [a << b for a, b in zip(x, y)])
		rs = OpTwo(lambda x, y: [a >> b for a, b in zip(x, y)])
		rls = OpTwo(lambda x, y: [b << a for a, b in zip(x, y)])
		rrs = OpTwo(lambda x, y: [b >> a for a, b in zip(x, y)])
		inv = Op(lambda x: [bit_not(n) for n in x])  # Single input
		inv_uns = Op(lambda x: [~n for n in x])  # Single input


class B:
	"""Boolean operators/methods"""
	not_ = Op(lambda x: [not n for n in x])  # Single input

	class I:
		"""Iterator and Non-Iterator operators/methods"""
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
		"""Iterator and Iterator operators/methods, zipped"""
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
	"""
	Non-Iterator and Non-Iterator operators/methods
	Additionally and Iterator single input operators/methods
	"""
	types = Op(lambda x: [type(n) for n in x])  # Single input
	fact = Op(lambda x: [math.factorial(n) for n in x])  # Single input
	itype = Op(lambda x: [is_iter(n) for n in x])  # Single input
	mca = Op(lambda x: combine_any(*x))  # 'Single' input of a list containing all the items to combine
	avg = OpTwo(lambda x, y: (x + y) / 2)
	mlen = Op(lambda x: [safe_len(n) for n in x])
	mca_i = Op(lambda x: combine_iters(*x))  # 'Single' input of a list containing all the items to combine, will return a generator, even if input does not contain a range or other generator. mca will call combine_iters() if an input is a range
	isa = Op(lambda x, y: x.__class__ == y.__class__)  # ADD MORE HELP INFO
	curry_r = OpTwo(lambda x, y: curry(y, x))  # takes function on right and curries is with the inputs on the left, returns list of curried functions
	curry_l = OpTwo(lambda x, y: curry(x, y))  # takes function on ;eft and curries is with the inputs on the right, returns list of curried functions
	curry = curry_l
	eval_l = OpTwo(lambda x, y: x(y))
	eval_r = OpTwo(lambda x, y: y(x))
	eval_ni = Op(lambda x: x())

	class Bin:
		"""Binary related"""
		inv = Op(lambda x: [bit_not(n) for n in x])  # Single input
		inv_uns = Op(lambda x: [~n for n in x])  # Single input


class D:
	"""Dictionary and Dictionary operators/methods"""
	append = OpTwo(lambda x, y: {**x, **y})
	combine = append
	intersect = OpTwo(lambda x, y: dict_intersect(x, y))
	and_ = intersect
	mutex = OpTwo(lambda x, y: dict_symdiff(x, y))
	ca = OpTwo(lambda x, y: combine_any(x, y))
	symdiff = mutex
	xor = mutex


class S:
	"""String and String operators/methods"""
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
