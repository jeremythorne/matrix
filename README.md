a simple Linear Algebra (Matrix) library in C++

able to
solve linear equations of the form Ax=b for unknown x

decompose A -> L * U
where L is lower triangular and U is upper triangular
then        solve Ly = b
and finally solve Ux = y

and able to use that to perform linear least squares regression
i.e. find model parameters B that minimise ||Y-XB||^2

references

https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/
https://en.wikipedia.org/wiki/LU_decomposition
https://en.wikipedia.org/wiki/Least_squares
