# AVP Custom Operators

How to use:
`variable_a | z_pwr | variable_b`
or: `variable_a << z_pwr >> variable_b`

Example:
`[4, 5, 6, 8] | z_pwr | [2, 4, 3, 2] -> [16, 625, 216, 64]`
`4 ** 2 = 16, 5 ** 4 = 625, 6 ** 3 = 216, 8 ** 2 = 64`

Credit to [Ferdinand Jamitzky](https://code.activestate.com/recipes/384122/) for the hack that makes this possible.

Currently there are 68 operators. I am working on a PyCharm plugin to add syntax highlighting for them.

Some operators don't make use of the second input. It is still necessary but can be anything. (Labelled with ***◊***)

### Operators (iterable & non-iterable)

* `i_div = [n / y for n in x]`
* `i_mul = [n * y for n in x]`
* `i_add = [n + y for n in x]`
* `i_sub = [n - y for n in x]`
* `i_rsub = [y - n for n in x]`
* `i_pwr = [n ** y for n in x]`
* `i_rpwr = [y ** n for n in x]`
* `i_mod = [n % y for n in x]`
* `i_rmod = [y % n for n in x]`
* `i_fmod = [math.fmod(n, y) for n in x]`
* `i_rfmod = [math.fmod(y, n) for n in x]`
* `ib_and = [n & y for n in x]`
* `ib_xor = [n ^ y for n in x]`
* `ib_or = [n | y for n in x]`
* `ib_ls = [n << y for n in x]`
* `ib_rs = [n >> y for n in x]`
* `i_sign = [math.copysign(n, y) for n in x]`
* `i_gcd = [math.gcd(n, y) for n in x]`
* `i_log = [math.log(n, y) for n in x]`
* `i_rlog = [math.log(y, n) for n in x]`
* `i_atan2 = [math.atan2(n, y) for n in x]`
* `i_ratan2 = [math.atan2(y, n) for n in x]`
* `i_hypot = [math.hypot(n, y) for n in x]`
* `i_rhypot = [math.hypot(y, n) for n in x]`

### Operators (iterable & iterable)

* `z_div = [a / b for a, b in zip(x, y)]`
* `z_mul = [a * b for a, b in zip(x, y)]`
* `z_add = [a + b for a, b in zip(x, y)]`
* `z_sub = [a - b for a, b in zip(x, y)]`
* `z_rsub = [b - a for a, b in zip(x, y)]`
* `z_pwr = [a ** b for a, b in zip(x, y)]`
* `z_rpwr = [b ** a for a, b in zip(x, y)]`
* `z_mod = [a % b for a, b in zip(x, y)]`
* `z_rmod = [b % a for a, b in zip(x, y)]`
* `z_fmod = [math.fmod(a, b) for a, b in zip(x, y)]`
* `z_rfmod = [math.fmod(b, a) for a, b in zip(x, y)]`
* `zb_and = [a & b for a, b in zip(x, y)]`
* `zb_xor = [a ^ b for a, b in zip(x, y)]`
* `zb_or = [a | b for a, b in zip(x, y)]`
* `zb_ls = [a << b for a, b in zip(x, y)]`
* `zb_rs = [a >> b for a, b in zip(x, y)]`
* `zbr_ls = [b << a for a, b in zip(x, y)]`
* `zbr_rs = [b >> a for a, b in zip(x, y)]`
* `z_addstr = [float(str(a).split(".")[0] + str(b)) for a, b in zip(x, y)]`
* `z_sign = [math.copysign(a, b) for a, b in zip(x, y)]`
* `z_gcd = [math.gcd(a, b) for a, b in zip(x, y)]`
* `z_log = [math.log(a, b) for a, b in zip(x, y)]`
* `z_rlog = [math.log(b, a) for a, b in zip(x, y)]`
* `z_atan2 = [math.atan2(a, b) for a, b in zip(x, y)]`
* `z_ratan2 = [math.atan2(b, a) for a, b in zip(x, y)]`
* `z_hypot = [math.hypot(a, b) for a, b in zip(x, y)]`
* `z_rhypot = [math.hypot(b, a) for a, b in zip(x, y)]`

#### Other Operators, including boolean focused operators

* `avg = (x + y) / 2)`
* `inv = [~ n for n in x]` ***◊***
* `fact = [math.factorial(n) for n in x]` ***◊***
* `replace = [n.replace(y[0], y[1] for n in x]`
* `i_set = [y * len(x)]` ***◊***
* `i_equ = [n == y for n in x]`
* `i_nequ = [n == y for n in x]`
* `z_equ = [a == b for a, b in zip(x, y)]`
* `z_nequ = [not(a == b) for a, b in zip(x, y)]`
* `i_not = [not n for n in x]` ***◊***
* `i_and = [n and y for n in x]`
* `i_nand = [not(n and y) for n in x]`
* `i_or = [n or y for n in x]`
* `i_nor = [not(n or y) for n in x]`
* `z_and = [a and b for a, b in zip(x, y)]`
* `z_nand = [not(a and b) for a, b in zip(x, y)]`
* `z_or = [a or b for a, b in zip(x, y)]`
* `z_nor = [not(a or b) for a, b in zip(x, y)]`

##### If you email me i may respond. If you have a bug report and want to email me, make a gihub issue first. Don't sign me up for spam or email me in a spammy way.