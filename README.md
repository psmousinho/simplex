# simplex
Implementation of simplex algorithm

## Instaces format:
```
N_VAR: <INT: Number of variables>
N_RES: <INT: Number of restrictions>
OBJ_FUN: <STR: "MIN|MAX"> <LIST[FLOAT]: List separated by space containing N_VAR coefficients>
RESTRICTIONS:
N_RES * <LIST[FLOAT]: List separated by space containing N_VAR coefficients> <STR: "<= | = | >=" > <FLOAT: Left side of restriction>
N_VAR * <STR: x{i}> <STR: "<= | = | >="> <FLOAT: Left side of bound restriction>
```
### Exemple:
```
N_VAR: 2
N_RES: 3
OBJ_FUN: MAX 3 5 
RESTRICTIONS:
1.0 0.0 <= 4.0
0.0 2.0 <= 12.0
3.0 2.0 <= 18.0
x1 >= 0
x2 >= 0
```
