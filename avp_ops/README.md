# AVP Custom Operators

Credit to [Ferdinand Jamitzky](https://code.activestate.com/recipes/384122/) for the hack that makes this possible.

### Operators:
Currently there are 315 operators. That is too many to list in great detail when I'd rather add more. Please read the code.

If you want to see the function for one and can't use an IDE to go to the code definition, you can use `.help()` on the operator and it will return its function, and for some it will return more help (which i am also slowly working on)

---

### OpTwo

Operators defined with OpTwo take in two inputs `input_one % operator % input_two` and can use `<<`, `>>`, `|`, `%`, and `@`.

All operators can take in inputs as a function. Simply call `Z.pwr([1, 2, 3], [4, 5, 6])` or `N.types(["string", 1, 2, [54, 32, 10]])` to call it just as if you used it as an operator.

#### Example:

```
>>> [4, 5, 6, 8] | Z.pwr | [2, 4, 3, 2]
    [16, 625, 216, 64]

>>> 4 ** 2
    16

>>> 5 ** 4
    625

>>> 6 ** 3
    216

>>> 8 ** 2
    64
```

### Op

Operators defined with Op take in one input `input % operator` or `operator % input` and can use `<<`, `>>`, `<`, `>`, `|`, `%`, and `@`.

#### Example:

```
>>> [1, 2, 3, ["5", 5]] % N.types
    [<class 'int'>, <class 'int'>, <class 'int'>, <class 'list'>]
    
>>> type(1)
    <class 'int'>

>>> type(2)
    <class 'int'>

>>> type(3)
    <class 'int'>

>>> type(["5", 5])
    <class 'list'>
```

##### The curry and eval set of functions are powerful, but can easily lead to confusing code.

```
N.curry_l     N.curry_r     N.eval_l     N.eval_r
Z.curry_l     Z.curry_r     Z.eval_l     Z.eval_r
I.curry_l_sf  I.curry_r_sf  I.eval_l_sf  I.eval_r_sf
I.curry_l_mf  I.curry_r_mf  I.eval_l_mf  I.eval_r_mf
```

##### If you email me i may respond. If you have a bug report and want to email me, make a gihub issue first. Don't sign me up for spam or email me in a spammy way.