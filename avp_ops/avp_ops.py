import time
import math
import inspect


class Infix:
	def __init__(self, function, help_str="", help_dict=None):
		self.function = function
		self.help_str = help_str
		self.help_dict = help_dict

	def __ror__(self, other):
		return Infix(lambda x, self=self, other=other: self.function(other, x))

	def __or__(self, other):
		return self.function(other)

	def __rlshift__(self, other):
		return Infix(lambda x, self=self, other=other: self.function(other, x))

	def __rshift__(self, other):
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
	div = Infix(lambda x, y: [n / y for n in x], I_help_dict["div"])
	rdiv = Infix(lambda x, y: [y / n for n in x], I_help_dict["rdiv"])
	mul = Infix(lambda x, y: [n * y for n in x], I_help_dict["mul"])
	add = Infix(lambda x, y: [n + y for n in x], I_help_dict["add"])
	sub = Infix(lambda x, y: [n - y for n in x], I_help_dict["sub"])
	rsub = Infix(lambda x, y: [y - n for n in x], I_help_dict["rsub"])
	pwr = Infix(lambda x, y: [n ** y for n in x], I_help_dict["pwr"])
	rpwr = Infix(lambda x, y: [y ** n for n in x], I_help_dict["rpwr"])
	mod = Infix(lambda x, y: [n % y for n in x], I_help_dict["mod"])
	rmod = Infix(lambda x, y: [y % n for n in x], I_help_dict["rmod"])
	fmod = Infix(lambda x, y: [math.fmod(n, y) for n in x], I_help_dict["fmod"])
	rfmod = Infix(lambda x, y: [math.fmod(y, n) for n in x], I_help_dict["rfmod"])
	sign = Infix(lambda x, y: [math.copysign(n, y) for n in x], I_help_dict["sign"])
	gcd = Infix(lambda x, y: [math.gcd(n, y) for n in x], I_help_dict["gcd"])
	log = Infix(lambda x, y: [math.log(n, y) for n in x], I_help_dict["log"])
	rlog = Infix(lambda x, y: [math.log(y, n) for n in x], I_help_dict["rlog"])
	atan2 = Infix(lambda x, y: [math.atan2(n, y) for n in x], I_help_dict["atan2"])
	ratan2 = Infix(lambda x, y: [math.atan2(y, n) for n in x], I_help_dict["ratan2"])
	hypot = Infix(lambda x, y: [math.hypot(n, y) for n in x], I_help_dict["hypot"])
	rhypot = Infix(lambda x, y: [math.hypot(y, n) for n in x], I_help_dict["rhypot"])
	avg = Infix(lambda x, y: [(n + y) / 2 for n in x], I_help_dict["avg"])
	fact = Infix(lambda x, y: [math.factorial(n) for n in x], I_help_dict["fact"])  # second term is necessary but unused
	repl = Infix(lambda x, y: [y[1] if n == y[0] else n for n in x], I_help_dict["repl"])
	replm = Infix(lambda x, y: [y[n] if n in y else n for n in x], I_help_dict["replm"])
	set = Infix(lambda x, y: [y * len(x)], I_help_dict["set"])

	class Div:
		Div_help_dict = {}
		addmul = Infix(lambda x, y: [(n + y) / (n * y) for n in x])
		addsub = Infix(lambda x, y: [(n + y) / (n - y) for n in x])
		addmod = Infix(lambda x, y: [(n + y) / (n % y) for n in x])
		addpwr = Infix(lambda x, y: [(n + y) / (n ** y) for n in x])
		adddiv = Infix(lambda x, y: [(n + y) / (n / y) for n in x])
		submul = Infix(lambda x, y: [(n - y) / (n * y) for n in x])
		subadd = Infix(lambda x, y: [(n - y) / (n + y) for n in x])
		submod = Infix(lambda x, y: [(n - y) / (n % y) for n in x])
		subpwr = Infix(lambda x, y: [(n - y) / (n ** y) for n in x])
		subdiv = Infix(lambda x, y: [(n - y) / (n / y) for n in x])
		muladd = Infix(lambda x, y: [(n * y) / (n + y) for n in x])
		mulsub = Infix(lambda x, y: [(n * y) / (n - y) for n in x])
		mulmod = Infix(lambda x, y: [(n * y) / (n % y) for n in x])
		mulpwr = Infix(lambda x, y: [(n * y) / (n ** y) for n in x])
		muldiv = Infix(lambda x, y: [(n * y) / (n / y) for n in x])
		modmul = Infix(lambda x, y: [(n % y) / (n * y) for n in x])
		modsub = Infix(lambda x, y: [(n % y) / (n - y) for n in x])
		modadd = Infix(lambda x, y: [(n % y) / (n + y) for n in x])
		modpwr = Infix(lambda x, y: [(n % y) / (n ** y) for n in x])
		moddiv = Infix(lambda x, y: [(n % y) / (n / y) for n in x])
		pwrmul = Infix(lambda x, y: [(n ** y) / (n * y) for n in x])
		pwrsub = Infix(lambda x, y: [(n ** y) / (n - y) for n in x])
		pwrmod = Infix(lambda x, y: [(n ** y) / (n % y) for n in x])
		pwradd = Infix(lambda x, y: [(n ** y) / (n + y) for n in x])
		pwrdiv = Infix(lambda x, y: [(n ** y) / (n / y) for n in x])
		divmul = Infix(lambda x, y: [(n / y) / (n * y) for n in x])
		divsub = Infix(lambda x, y: [(n / y) / (n - y) for n in x])
		divmod = Infix(lambda x, y: [(n / y) / (n % y) for n in x])
		divpwr = Infix(lambda x, y: [(n / y) / (n ** y) for n in x])
		divadd = Infix(lambda x, y: [(n / y) / (n + y) for n in x])

	class DInv:
		disub = Infix(lambda x, y: [1 / ((1 / n) - (1 / y)) for n in x])
		diadd = Infix(lambda x, y: [1 / ((1 / n) + (1 / y)) for n in x])
		dimul = Infix(lambda x, y: [1 / ((1 / n) * (1 / y)) for n in x])
		didiv = Infix(lambda x, y: [1 / ((1 / n) / (1 / y)) for n in x])
		dimod = Infix(lambda x, y: [1 / ((1 / n) % (1 / y)) for n in x])
		dipwr = Infix(lambda x, y: [1 / ((1 / n) ** (1 / y)) for n in x])
		disubr = Infix(lambda x, y: [1 / ((1 / y) - (1 / n)) for n in x])
		diaddr = Infix(lambda x, y: [1 / ((1 / y) + (1 / n)) for n in x])
		dimulr = Infix(lambda x, y: [1 / ((1 / y) * (1 / n)) for n in x])
		didivr = Infix(lambda x, y: [1 / ((1 / y) / (1 / n)) for n in x])
		dimodr = Infix(lambda x, y: [1 / ((1 / y) % (1 / n)) for n in x])
		dipwrr = Infix(lambda x, y: [1 / ((1 / y) ** (1 / n)) for n in x])

	class Bin:
		and_ = Infix(lambda x, y: [n & y for n in x])
		xor = Infix(lambda x, y: [n ^ y for n in x])
		xnor = Infix(lambda x, y: [bit_not(n ^ y) for n in x])
		xnor_uns = Infix(lambda x, y: [~(n ^ y) for n in x])
		or_ = Infix(lambda x, y: [n | y for n in x])
		ls = Infix(lambda x, y: [n << y for n in x])
		rs = Infix(lambda x, y: [n >> y for n in x])
		rls = Infix(lambda x, y: [y << n for n in x])
		rrs = Infix(lambda x, y: [y >> n for n in x])
		inv = Infix(lambda x, y: [bit_not(n) for n in x])  # second term is necessary but unused
		inv_uns = Infix(lambda x, y: [~n for n in x])  # second term is necessary but unused


class Z:
	div = Infix(lambda x, y: [a / b for a, b in zip(x, y)])
	rdiv = Infix(lambda x, y: [b / a for a, b in zip(x, y)])
	mul = Infix(lambda x, y: [a * b for a, b in zip(x, y)])
	add = Infix(lambda x, y: [a + b for a, b in zip(x, y)])
	sub = Infix(lambda x, y: [a - b for a, b in zip(x, y)])
	rsub = Infix(lambda x, y: [b - a for a, b in zip(x, y)])
	pwr = Infix(lambda x, y: [a ** b for a, b in zip(x, y)])
	rpwr = Infix(lambda x, y: [b ** a for a, b in zip(x, y)])
	mod = Infix(lambda x, y: [a % b for a, b in zip(x, y)])
	rmod = Infix(lambda x, y: [b % a for a, b in zip(x, y)])
	fmod = Infix(lambda x, y: [math.fmod(a, b) for a, b in zip(x, y)])
	rfmod = Infix(lambda x, y: [math.fmod(b, a) for a, b in zip(x, y)])
	addstr = Infix(lambda x, y: [float(str(a).split(".")[0] + str(b)) for a, b in zip(x, y)])
	sign = Infix(lambda x, y: [math.copysign(a, b) for a, b in zip(x, y)])
	gcd = Infix(lambda x, y: [math.gcd(a, b) for a, b in zip(x, y)])
	log = Infix(lambda x, y: [math.log(a, b) for a, b in zip(x, y)])
	rlog = Infix(lambda x, y: [math.log(b, a) for a, b in zip(x, y)])
	atan2 = Infix(lambda x, y: [math.atan2(a, b) for a, b in zip(x, y)])
	ratan2 = Infix(lambda x, y: [math.atan2(b, a) for a, b in zip(x, y)])
	hypot = Infix(lambda x, y: [math.hypot(a, b) for a, b in zip(x, y)])
	rhypot = Infix(lambda x, y: [math.hypot(b, a) for a, b in zip(x, y)])
	avg = Infix(lambda x, y: [(a + b) / 2 for a, b in zip(x, y)])
	intersect = Infix(lambda x, y: list(set(x).intersection(y)))

	class Div:
		addmul = Infix(lambda x, y: [(a + b) / (a * b) for a, b in zip(x, y)])
		addsub = Infix(lambda x, y: [(a + b) / (a - b) for a, b in zip(x, y)])
		addmod = Infix(lambda x, y: [(a + b) / (a % b) for a, b in zip(x, y)])
		addpwr = Infix(lambda x, y: [(a + b) / (a ** b) for a, b in zip(x, y)])
		adddiv = Infix(lambda x, y: [(a + b) / (a / b) for a, b in zip(x, y)])
		submul = Infix(lambda x, y: [(a - b) / (a * b) for a, b in zip(x, y)])
		subadd = Infix(lambda x, y: [(a - b) / (a + b) for a, b in zip(x, y)])
		submod = Infix(lambda x, y: [(a - b) / (a % b) for a, b in zip(x, y)])
		subpwr = Infix(lambda x, y: [(a - b) / (a ** b) for a, b in zip(x, y)])
		subdiv = Infix(lambda x, y: [(a - b) / (a / b) for a, b in zip(x, y)])
		muladd = Infix(lambda x, y: [(a * b) / (a + b) for a, b in zip(x, y)])
		mulsub = Infix(lambda x, y: [(a * b) / (a - b) for a, b in zip(x, y)])
		mulmod = Infix(lambda x, y: [(a * b) / (a % b) for a, b in zip(x, y)])
		mulpwr = Infix(lambda x, y: [(a * b) / (a ** b) for a, b in zip(x, y)])
		muldiv = Infix(lambda x, y: [(a * b) / (a / b) for a, b in zip(x, y)])
		modmul = Infix(lambda x, y: [(a % b) / (a * b) for a, b in zip(x, y)])
		modsub = Infix(lambda x, y: [(a % b) / (a - b) for a, b in zip(x, y)])
		modadd = Infix(lambda x, y: [(a % b) / (a + b) for a, b in zip(x, y)])
		modpwr = Infix(lambda x, y: [(a % b) / (a ** b) for a, b in zip(x, y)])
		moddiv = Infix(lambda x, y: [(a % b) / (a / b) for a, b in zip(x, y)])
		pwrmul = Infix(lambda x, y: [(a ** b) / (a * b) for a, b in zip(x, y)])
		pwrsub = Infix(lambda x, y: [(a ** b) / (a - b) for a, b in zip(x, y)])
		pwrmod = Infix(lambda x, y: [(a ** b) / (a % b) for a, b in zip(x, y)])
		pwradd = Infix(lambda x, y: [(a ** b) / (a + b) for a, b in zip(x, y)])
		pwrdiv = Infix(lambda x, y: [(a ** b) / (a / b) for a, b in zip(x, y)])
		divmul = Infix(lambda x, y: [(a / b) / (a * b) for a, b in zip(x, y)])
		divsub = Infix(lambda x, y: [(a / b) / (a - b) for a, b in zip(x, y)])
		divmod = Infix(lambda x, y: [(a / b) / (a % b) for a, b in zip(x, y)])
		divpwr = Infix(lambda x, y: [(a / b) / (a ** b) for a, b in zip(x, y)])
		divadd = Infix(lambda x, y: [(a / b) / (a + b) for a, b in zip(x, y)])

	class DInv:
		disub = Infix(lambda x, y: [1 / ((1 / a) - (1 / b)) for a, b in zip(x, y)])
		diadd = Infix(lambda x, y: [1 / ((1 / a) + (1 / b)) for a, b in zip(x, y)])
		dimul = Infix(lambda x, y: [1 / ((1 / a) * (1 / b)) for a, b in zip(x, y)])
		didiv = Infix(lambda x, y: [1 / ((1 / a) / (1 / b)) for a, b in zip(x, y)])
		dimod = Infix(lambda x, y: [1 / ((1 / a) % (1 / b)) for a, b in zip(x, y)])
		dipwr = Infix(lambda x, y: [1 / ((1 / a) ** (1 / b)) for a, b in zip(x, y)])
		disubr = Infix(lambda x, y: [1 / ((1 / b) - (1 / a)) for a, b in zip(x, y)])
		diaddr = Infix(lambda x, y: [1 / ((1 / b) + (1 / a)) for a, b in zip(x, y)])
		dimulr = Infix(lambda x, y: [1 / ((1 / b) * (1 / a)) for a, b in zip(x, y)])
		didivr = Infix(lambda x, y: [1 / ((1 / b) / (1 / a)) for a, b in zip(x, y)])
		dimodr = Infix(lambda x, y: [1 / ((1 / b) % (1 / a)) for a, b in zip(x, y)])
		dipwrr = Infix(lambda x, y: [1 / ((1 / b) ** (1 / a)) for a, b in zip(x, y)])

	class Bin:
		and_ = Infix(lambda x, y: [a & b for a, b in zip(x, y)])
		xor = Infix(lambda x, y: [a ^ b for a, b in zip(x, y)])
		xnor = Infix(lambda x, y: [bit_not(a ^ b) for a, b in zip(x, y)])
		xnor_uns = Infix(lambda x, y: [~(a ^ b) for a, b in zip(x, y)])
		or_ = Infix(lambda x, y: [a | b for a, b in zip(x, y)])
		ls = Infix(lambda x, y: [a << b for a, b in zip(x, y)])
		rs = Infix(lambda x, y: [a >> b for a, b in zip(x, y)])
		rls = Infix(lambda x, y: [b << a for a, b in zip(x, y)])
		rrs = Infix(lambda x, y: [b >> a for a, b in zip(x, y)])


avg = Infix(lambda x, y: (x + y) / 2)


class B:
	not_ = Infix(lambda x, y: [not n for n in x])  # second term is necessary but unused

	class I:
		equ = Infix(lambda x, y: [n == y for n in x])
		nequ = Infix(lambda x, y: [n == y for n in x])
		not_ = Infix(lambda x, y: [not n for n in x])  # second term is necessary but unused
		and_ = Infix(lambda x, y: [n and y for n in x])
		nand = Infix(lambda x, y: [not(n and y) for n in x])
		or_ = Infix(lambda x, y: [n or y for n in x])
		nor = Infix(lambda x, y: [not(n or y) for n in x])
		xor = Infix(lambda x, y: [n ^ y for n in x])
		xnor = Infix(lambda x, y: [not(n ^ y) for n in x])

	class Z:
		equ = Infix(lambda x, y: [a == b for a, b in zip(x, y)])
		nequ = Infix(lambda x, y: [not(a == b) for a, b in zip(x, y)])
		not_ = Infix(lambda x, y: [not n for n in x])  # second term is necessary but unused
		and_ = Infix(lambda x, y: [a and b for a, b in zip(x, y)])
		nand = Infix(lambda x, y: [not(a and b) for a, b in zip(x, y)])
		or_ = Infix(lambda x, y: [a or b for a, b in zip(x, y)])
		nor = Infix(lambda x, y: [not(a or b) for a, b in zip(x, y)])
		xor = Infix(lambda x, y: [a ^ b for a, b in zip(x, y)])
		xnor = Infix(lambda x, y: [not(a ^ b) for a, b in zip(x, y)])


def combine(new: dict, existing: dict) -> dict:
	out = {}
	for key in new.keys():
		out[key] = new[key]
	for key in existing.keys():
		out[key] = existing[key]
	return out


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
	append = Infix(lambda x, y: dict(**x, **y))
	combine = append
	append_non_str = Infix(lambda x, y: combine(x, y))
	combine_non_str = append_non_str
	intersect = Infix(lambda x, y: dict_intersect(x, y))
	and_ = intersect
	mutex = Infix(lambda x, y: dict_symdiff(x, y))
	symdiff = mutex
	xor = mutex


class S:
	subt_l = Infix(lambda x, y: string_sub_l(x, y))
	subt_r = Infix(lambda x, y: string_sub_r(x, y))
	rsubt_l = Infix(lambda x, y: string_sub_l(x, y))
	rsubt_r = Infix(lambda x, y: string_sub_r(x, y))
	sub = Infix(lambda x, y: x.replace(y, ""))
	repl_t = Infix(lambda x, y: [x.replace(str(y[0]), str(y[1]))])
	repl_d = Infix(lambda x, y: string_sub_m(x, y))
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

