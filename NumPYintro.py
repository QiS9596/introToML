"""
This module is used to practice the fundamental knowledge of Numpy.
Most codes are re-typing code from the Numpy official tutorial at https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
"""

from numpy import *

a = arange(20)
print(a)

a = a.reshape(5, 2, 2)  # reshape a ndarray into a 5*2*2 3d ndarray
print(a)

print(a.ndim)  # the dimension of the data

a = a.reshape(20)
print(a)

print(a.dtype)  # data type of the element

print(a.itemsize)

print(type(a))

a = array([1, 2, 3])

print(type(a))  # in numpy, array is a alias of ndarray

a = linspace(0, 2, 9)
print(type(a))
print(a)
x = linspace(9, 2 * pi, 100)
f = sin(x)
print(x)
print(f)

a = array([[1, 1], [0, 1]])
b = array([[2, 0], [3, 4]])

print(a * b)  # * operator are elementwise in numpy

print(a.dot(b))  # dot method are used for mutiplication

# assignment do not copy the ndarray at all
# view is shallow copy, the data resource stays the same, but the shape can be changed without interfering the others
# .copy() method of ndarray returns a deep copy instance

# fancy indexing
# indexing with ndarray, the output remains the same shape with the
# indexing array

a = arange(12)**2
j = array([[3, 4], [9, 7]])
print(a[j])
print(j.shape == a[j].shape)

# indexing with multidimensional, i stands for the first dimension, j stands for the second dimension, the shape of i,j
# should be the same, and will be the same as the result
a = arange(12).reshape(3, 4)
print(a)
i = array([[0, 1], [1, 2]])
j = array([[2, 1], [3, 3]])
print(a[i, j])
# i, j can be put into a python list
k = [i, j]
print(a[i, j] == a[k])
# but can not be put into a nd array

# indexing with boolean array
b = a > 4
print(b)
a[b] = 0
print(a)
# all element greater than 4 are changed to 0

import matplotlib.pyplot as plt


def mandelbrot(h, w, maxit=20):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y,x = ogrid[-1.4:1.4:h * 1j, -2:0.8:w * 1j]
    c = x + y * 1j
    z = c
    divtime = maxit + zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z * conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime == maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime


#plt.imshow(mandelbrot(400, 400))
#plt.show()

# linear algebra

a = array([[1.0, 2.0], [3.0, 4.0]])
print(a)
print(a.transpose())
print(linalg.inv(a))
print(linalg.inv(a).dot(a))
b = linalg.inv(a).dot(a)
print(dot(b,b))
print(trace(b))
print(eye(2))
b = array([[5.],[7.]])
print(linalg.solve(a,b))

# tricks and tips
# automatic reshaping: use -1 in reshaping