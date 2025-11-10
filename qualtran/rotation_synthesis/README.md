# Rotation Synthesis

Rotation-Synthesis implements the state of the art rotation synthesis protocols for compiling $SU(2)$ unitaries into clifford+T gates.


## Minimal Examples

### Common Setup
The examples assume the following lines have been executed.

```py3
>>> import mpmath
>>> import qualtran.rotation_synthesis as rs
>>> from qualtran.rotation_synthesis import math_config as mc
>>> from qualtran.rotation_synthesis.protocols import clifford_t_synthesis as cts
>>> # 300 digits of precision is enough for eps as low as 10^-30.
>>> # for big enough eps (>= 10^-10) prefer dps ~100 to reduce the runtime
>>> # A high dps increases the runtime but a low dps risks missing solutions.
>>> config = mc.with_dps(300)
>>> theta = 0.1  # theta = 0.1 rad
>>> eps = mpmath.mpf(1e-8)  # epsilon = 1e-8
```

### Diagonal Protocol

```py3
>>> diagonal = cts.diagonal_unitary_approx(theta=theta, eps=eps, max_n=400, config=config)
>>> diagonal
Unitary(p=ZW(coords=(1509861905169903736208, 1174754259083368969221, 151491500481353346337, -960512924518388809952)), q=ZW(coords=(-4263827284335, -6564144819191, -5019275344346, -534182446068)), n=80, twirl=False)

>>> 'number of T gates: %d'%diagonal.n  # Number of T gates used
'number of T gates: 80'

>>> 'actual diamond distance: %e'%diagonal.diamond_norm_distance_to_rz(theta, config)
'actual diamond distance: 8.854162e-09'

>>> diagonal.to_matrix()  # Print matrix form
SU2CliffordT(matrix=array([[ZW(coords=(1509861905169903736208, 1174754259083368969221, 151491500481353346337, -960512924518388809952)),
        ZW(coords=(4263827284335, -534182446068, -5019275344346, -6564144819191))],
       [ZW(coords=(-4263827284335, -6564144819191, -5019275344346, -534182446068)),
        ZW(coords=(1509861905169903736208, 960512924518388809952, -151491500481353346337, -1174754259083368969221))]],
      dtype=object), gates=())

>>> rs.to_sequence(diagonal.to_matrix())  # Print gate names
('S', 'H', 'Tz', 'Tx', 'Ty', 'Tz', 'Tx', 'Tz', 'Tx', 'Tz', 'Ty', 'Tz', 'Ty', 'Tx', 'Tz', 'Tx', 'Ty', 'Tx', 'Ty', 'Tx', 'Tz', 'Ty', 'Tz', 'Ty', 'Tz', 'Ty', 'Tz', 'Ty', 'Tx', 'Tz', 'Ty', 'Tz', 'Ty', 'Tz', 'Tx', 'Tz', 'Ty', 'Tz', 'Ty', 'Tx', 'Ty', 'Tz', 'Tx', 'Tz', 'Ty', 'Tx', 'Ty', 'Tx', 'Tz', 'Tx', 'Ty', 'Tz', 'Ty', 'Tx', 'Ty', 'Tx', 'Ty', 'Tz', 'Tx', 'Tz', 'Ty', 'Tz', 'Ty', 'Tz', 'Tx', 'Ty', 'Tz', 'Tx', 'Tz', 'Ty', 'Tx', 'Ty', 'Tz', 'Tx', 'Ty', 'Tx', 'Ty', 'Tx', 'Ty', 'Tz', 'Ty', 'Tz')

>>> print(diagonal.to_cirq())  # Export to Cirq
0: ───S───Rx(0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───

>>> print(diagonal.to_quirk(fmt='xz'))  # Export to quirk
https://algassert.com/quirk#circuit={"cols":[["Z^½"],["X^¼"],["Z^-¼"],["X^¼"],["Z^¼"],["X^¼"],["Z^-¼"],["X^¼"],["Z^¼"],["X^¼"],["Z^-¼"],["X^-¼"],["Z^¼"],["X^¼"],["Z^¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^¼"],["X^¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^¼"],["X^¼"],["Z^¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^¼"],["X^¼"],["Z^¼"],["X^¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^¼"],["Z^¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^¼"],["X^¼"],["Z^¼"],["X^¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^¼"],["X^-¼"],["Z^-¼"],["X^-¼"],["Z^-¼"],["X^¼"],["Z^¼"],["X^-¼"],["Z^-¼"]]}
```


### Fallback (a.k.a Repeat-Until-Success (RUS)) Protocol

```py3
>>> fallback = cts.fallback_protocol(theta, eps, success_probability=0.99, max_n=200, config=config)
>>> fallback
ProjectiveChannel(rotation=Unitary(p=ZW(coords=(239242444, 186143567, 24004313, -152196342)), q=ZW(coords=(11711997, 17295059, 12746910, 731794)), n=32, twirl=False), correction=Unitary(p=ZW(coords=(113393374677276065, 8759190556798315, -101006008596441546, -151603257795059376)), q=ZW(coords=(-13979001969, 2234592382, 17139192822, 22003886555)), n=65, twirl=False))

>>> print('expected number of T gates is %.3f divided into\n%d T gates used in the projective step and\n%d T gates used in the correction step which gets executed with probability %.6f'%(fallback.expected_num_ts(config), fallback.rotation.n, fallback.correction.n, 1 - fallback.success_probability(config)))
expected number of T gates is 32.335 divided into
32 T gates used in the projective step and
65 T gates used in the correction step which gets executed with probability 0.005156

>>> 'actual diamond distance: %e'%fallback.diamond_norm_distance_to_rz(theta, config)
'actual diamond distance: 8.358389e-09'

>>> print(fallback.to_quirk(fmt='xz'))  # Export to Quirk
https://algassert.com/quirk#circuit={"cols":[["•","X"],["Z^½",1],["Z^½",1],["Z^½",1],["H",1],["Z^½",1],["H",1],["X^¼",1],["Z^¼",1],["X^¼",1],["Z^-¼",1],["X^¼",1],["Z^-¼",1],["X^-¼",1],["Z^¼",1],["X^¼",1],["Z^¼",1],["X^¼",1],["Z^¼",1],["X^¼",1],["Z^¼",1],["X^-¼",1],["Z^-¼",1],["X^-¼",1],["Z^-¼",1],["X^¼",1],["Z^-¼",1],["X^-¼",1],["Z^¼",1],["X^-¼",1],["Z^-¼",1],["X^¼",1],["Z^-¼",1],["X^-¼",1],["Z^-¼",1],["X^-¼",1],["Z^-¼",1],["X^-¼",1],["Z^-¼",1],["•","X"],[1,"Measure"],["H","•"],["Z^½","•"],["H","•"],["X^-¼","•"],["Z^-¼","•"],["X^-¼","•"],["Z^-¼","•"],["X^-¼","•"],["Z^-¼","•"],["X^-¼","•"],["Z^-¼","•"],["X^-¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^¼","•"],["X^¼","•"],["Z^¼","•"],["X^¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^¼","•"],["X^¼","•"],["Z^¼","•"],["X^¼","•"],["Z^¼","•"],["X^¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^¼","•"],["X^¼","•"],["Z^-¼","•"],["X^-¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^-¼","•"],["X^¼","•"],["Z^¼","•"],["X^¼","•"],["Z^-¼","•"],["X^-¼","•"],["Z^-¼","•"],["X^¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^-¼","•"],["X^¼","•"],["Z^-¼","•"],["X^-¼","•"],["Z^-¼","•"],["X^-¼","•"],["Z^-¼","•"],["X^¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^¼","•"],["X^¼","•"],["Z^¼","•"],["X^¼","•"],["Z^-¼","•"],["X^¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^¼","•"],["X^-¼","•"],["Z^¼","•"],["X^-¼","•"],["H","•"]]}

>>> print(fallback.to_cirq())  # Export to Cirq.
```

<details>
<summary> Cirq circuit </summary>

```sh

0: ───@───[ 0: ───S───S───S───H───S───H───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)─── ]───@───────[ 0: ───H───S───H───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(-0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(0.25π)───Rz(-0.25π)───Rx(0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(-0.25π)───Rz(0.25π)───Rx(-0.25π)───H─── ].with_classical_controls(m)───
      │                                                                                                                                                                                                                                                                                                                                                                                                                                                          │       ║
1: ───X──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────X───M───╫───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ║   ║
m: ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════@═══╩═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
```
</details>


### Mixed Diagonal Protocol

```py3
>>> mixed_diagonal = cts.mixed_diagonal_protocol(theta, eps, max_n=300, config=config)
>>> mixed_diagonal
ProbabilisticChannel(c1=Unitary(p=ZW(coords=(111279824866, 86577736003, 11159583589, -70795701541)), q=ZW(coords=(2905774, -827421, -4075924, -4936806)), n=42, twirl=True), c2=Unitary(p=ZW(coords=(32592863748, 25359568315, 3270981699, -20733701634)), q=ZW(coords=(-261947, -1668702, -2097954, -1298253)), n=40, twirl=True), probability=mpf('0.322751359448635267114326370859256829689458745026081702686971426030895682599474514910816660886778364762073662905955339813205441513151138367919667045726178096784779764992341606213189667302147688085007915854895131323239141860542004025041462985374870176351764143883625569021092174011359447964257700140487645'))

>>> print('expected number of T gates is %.3f '%(mixed_diagonal.expected_num_ts(config)))
expected number of T gates is 40.646

>>> 'actual diamond distance: %e'%mixed_diagonal.diamond_norm_distance_to_rz(theta, config)
'actual diamond distance: 9.336685e-09'
```


### Mixed Fallback Protocol


```py3
>>> mixed_fallback = cts.mixed_fallback_protocol(theta, eps, success_probability=0.99, max_n=300, config=config)
>>> mixed_fallback
ProbabilisticChannel(c1=ProjectiveChannel(rotation=Unitary(p=ZW(coords=(81907, 63728, 8218, -52106)), q=ZW(coords=(-1154, 985, 2547, 2617)), n=19, twirl=False), correction=ProbabilisticChannel(c1=Unitary(p=ZW(coords=(-263391098, -737624638, -779767669, -365133375)), q=ZW(coords=(-33461, 136222, 226108, 183543)), n=34, twirl=True), c2=Unitary(p=ZW(coords=(-77065381, -216007734, -228415686, -107020827)), q=ZW(coords=(-69300, -100383, -72663, -2378)), n=32, twirl=True), probability=mpf('0.045325686665484547940062502757622971144169613023747430900082177527088724087189898382813946924494697326621166855704833076057576200695347588135413318392991377437019817846667142336189311546588693427566709307270541970137781795585088761789630584519324449933375058208979540812676838676732065743027397323424595'))), c2=ProjectiveChannel(rotation=Unitary(p=ZW(coords=(23978, 18657, 2407, -15253)), q=ZW(coords=(-956, -1084, -577, 268)), n=17, twirl=False), correction=ProbabilisticChannel(c1=Unitary(p=ZW(coords=(-1373879826, -510348019, 652138736, 1432611464)), q=ZW(coords=(-189284, 278185, 582697, 545873)), n=35, twirl=True), c2=Unitary(p=ZW(coords=(-402460569, -149610333, 190879607, 419554862)), q=ZW(coords=(-69300, -31083, 25342, 66922)), n=33, twirl=True), probability=mpf('0.525040802635724724816097410427482676809933478764171151369355514657076375140052487037506999832799609679457396659076441254494844846449599140703045579066152308112059014940560931326583883941250186082725677082913867579961221230890451563733787938393583123811396292716244823945079739397111073899481930671270029'))), probability=mpf('0.972077582319464271880109129184631244429176680508313282994368258672700642062726782223015417086951750128434612291521852730028085519115453255242801194500490513953599832009476998145136699780645059022125864590137472901641917927185402638252104301881958927225784570356720997394377273869421589887668348541958305'))

>>> print('expected number of T gates is %.3f '%(mixed_fallback.expected_num_ts(config)))
expected number of T gates is 18.982

>>> 'actual diamond distance: %e'%mixed_fallback.diamond_norm_distance_to_rz(theta, config)
'actual diamond distance: 5.397363e-10'
```

## Effect of digits of precision
The number of digits of precision used (i.e. rs.with_dps(digits_of_precision)) affects the result of the synthesis as follows:

- very low => A math error will be raised by one of the checks|
- low => A valid solution may be missed, in other words you get either a solution that has more T gates or None|
- just right => A correct solution that has a number of T gates on par with the current state of the art|
- high => same solution as above but in more time|

Essentially, the code will either return a valid synthesis or None. If the code returns a result then it may be improved by increasing the number of digits of precision and if the code returns None then we need to increase the number of digits of precisions or `max_n` or both.

As a rule of thumb, the number of digits of precision should be close to $9\log_{10}{1/\epsilon}$. This works for large $epsilon$ and is an upperbound for very small $\epsilon$, for example for $\epsilon=10^{-50}$ we need 400 digits and for $\epsilon=10^{-100}$ we need 800 digits.
