import time
import math


class Infix:
	def __init__(self, function):
		self.function = function

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


# t___ is tuple ___. the second is applied to the first, which should be a list/tuple/thing that supports iteration
# z___ is zip ___. the iterators get applied element-wise, as opposed to sequentially, i.e. for addition

i_div = Infix(lambda x, y: [n / y for n in x])
i_mul = Infix(lambda x, y: [n * y for n in x])
i_add = Infix(lambda x, y: [n + y for n in x])
i_sub = Infix(lambda x, y: [n - y for n in x])
i_rsub = Infix(lambda x, y: [y - n for n in x])
i_pwr = Infix(lambda x, y: [n ** y for n in x])
i_rpwr = Infix(lambda x, y: [y ** n for n in x])
i_mod = Infix(lambda x, y: [n % y for n in x])
i_rmod = Infix(lambda x, y: [y % n for n in x])
i_fmod = Infix(lambda x, y: [math.fmod(n, y) for n in x])
i_rfmod = Infix(lambda x, y: [math.fmod(y, n) for n in x])
ib_and = Infix(lambda x, y: [n & y for n in x])
ib_xor = Infix(lambda x, y: [n ^ y for n in x])
ib_or = Infix(lambda x, y: [n | y for n in x])
ib_ls = Infix(lambda x, y: [n << y for n in x])
ib_rs = Infix(lambda x, y: [n >> y for n in x])
i_sign = Infix(lambda x, y: [math.copysign(n, y) for n in x])
i_gcd = Infix(lambda x, y: [math.gcd(n, y) for n in x])
i_log = Infix(lambda x, y: [math.log(n, y) for n in x])
i_rlog = Infix(lambda x, y: [math.log(y, n) for n in x])
i_atan2 = Infix(lambda x, y: [math.atan2(n, y) for n in x])
i_ratan2 = Infix(lambda x, y: [math.atan2(y, n) for n in x])
i_hypot = Infix(lambda x, y: [math.hypot(n, y) for n in x])
i_rhypot = Infix(lambda x, y: [math.hypot(y, n) for n in x])

z_div = Infix(lambda x, y: [a / b for a, b in zip(x, y)])
z_mul = Infix(lambda x, y: [a * b for a, b in zip(x, y)])
z_add = Infix(lambda x, y: [a + b for a, b in zip(x, y)])
z_sub = Infix(lambda x, y: [a - b for a, b in zip(x, y)])
z_rsub = Infix(lambda x, y: [b - a for a, b in zip(x, y)])
z_pwr = Infix(lambda x, y: [a ** b for a, b in zip(x, y)])
z_rpwr = Infix(lambda x, y: [b ** a for a, b in zip(x, y)])
z_mod = Infix(lambda x, y: [a % b for a, b in zip(x, y)])
z_rmod = Infix(lambda x, y: [b % a for a, b in zip(x, y)])
z_fmod = Infix(lambda x, y: [math.fmod(a, b) for a, b in zip(x, y)])
z_rfmod = Infix(lambda x, y: [math.fmod(b, a) for a, b in zip(x, y)])
zb_and = Infix(lambda x, y: [a & b for a, b in zip(x, y)])
zb_xor = Infix(lambda x, y: [a ^ b for a, b in zip(x, y)])
zb_or = Infix(lambda x, y: [a | b for a, b in zip(x, y)])
zb_ls = Infix(lambda x, y: [a << b for a, b in zip(x, y)])
zb_rs = Infix(lambda x, y: [a >> b for a, b in zip(x, y)])
zbr_ls = Infix(lambda x, y: [b << a for a, b in zip(x, y)])
zbr_rs = Infix(lambda x, y: [b >> a for a, b in zip(x, y)])
z_addstr = Infix(lambda x, y: [float(str(a).split(".")[0] + str(b)) for a, b in zip(x, y)])
z_sign = Infix(lambda x, y: [math.copysign(a, b) for a, b in zip(x, y)])
z_gcd = Infix(lambda x, y: [math.gcd(a, b) for a, b in zip(x, y)])
z_log = Infix(lambda x, y: [math.log(a, b) for a, b in zip(x, y)])
z_rlog = Infix(lambda x, y: [math.log(b, a) for a, b in zip(x, y)])
z_atan2 = Infix(lambda x, y: [math.atan2(a, b) for a, b in zip(x, y)])
z_ratan2 = Infix(lambda x, y: [math.atan2(b, a) for a, b in zip(x, y)])
z_hypot = Infix(lambda x, y: [math.hypot(a, b) for a, b in zip(x, y)])
z_rhypot = Infix(lambda x, y: [math.hypot(b, a) for a, b in zip(x, y)])

avg = Infix(lambda x, y: (x + y) / 2)
inv = Infix(lambda x, y: [~ n for n in x])  # second term is necessary but unused
fact = Infix(lambda x, y: [math.factorial(n) for n in x])  # second term is necessary but unused
replace = Infix(lambda x, y: [n.replace(y[0], y[1]) for n in x])
i_set = Infix(lambda x, y: [y * len(x)])
i_equ = Infix(lambda x, y: [n == y for n in x])
i_nequ = Infix(lambda x, y: [n == y for n in x])
z_equ = Infix(lambda x, y: [a == b for a, b in zip(x, y)])
z_nequ = Infix(lambda x, y: [not(a == b) for a, b in zip(x, y)])
i_not = Infix(lambda x, y: [not n for n in x])  # second term is necessary but unused
i_and = Infix(lambda x, y: [n and y for n in x])
i_nand = Infix(lambda x, y: [not(n and y) for n in x])
i_or = Infix(lambda x, y: [n or y for n in x])
i_nor = Infix(lambda x, y: [not(n or y) for n in x])
z_and = Infix(lambda x, y: [a and b for a, b in zip(x, y)])
z_nand = Infix(lambda x, y: [not(a and b) for a, b in zip(x, y)])
z_or = Infix(lambda x, y: [a or b for a, b in zip(x, y)])
z_nor = Infix(lambda x, y: [not(a or b) for a, b in zip(x, y)])


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

