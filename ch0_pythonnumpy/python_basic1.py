# -*- coding: utf-8 -*-
"""
Python basic  Demo

Created on Thu Mar 12 18:21:43 2020

@author: Ahn
"""

#
# 1. Vriable and type 
#
x = 1
print(type(x))

x = 1.2
print(type(x))

x = "hello"
print(type(x))

print(len(x)) #this will work because x is a string

x = 1.2
#print(len(x)) #what will happen here?


p = 1
print (type(p)) #the parentheses around the argument to print are opt
q = .2
print (type(q))
r = p + q
print (type(r))
print ("value of r: {}".format(r)) #compare this to printf!


# 2. Control 

if r < 3:
    print ("x")
else:
    print ("y")
    
    
if r < 1:
    print ("x")
    print ("less than 1")
elif r < 2:
    print ("y")
    print ("less than 2")
elif r < 3:
    print ("z")
    print ("less than 3")
else:
    print ("w")
    print ("otherwise!")
    
    
x = 1
y = 1
while (x <= 10) :
    y *= x
    x += 1
print(y)
    
data = [1, 4, 9, 0, 4, 2, 6, 1, 2, 8, 4, 5, 0, 7]
print(data)

hist = 5 * [0]
print(hist)

length = len(data)
print( "data length: {} data[{}] = {}".format(length, length - 1, data[length - 1]))

data.append(8)
length = len(data)
print("data length: {} data[{}] = {}".format(length, length - 1, data[length - 1]))

for d in data :
    print (d)
    
for d in data :
    hist[d//2] += 1
print( hist)
  
    
    
r = range(0,5)
print( r)

for i in range(0, 5):
    print( i)

    
def foo(x) :
    return x * 2

print(foo(10))


def foo2(x) :
    return x * 2, x * 4

(a, b) = foo2(10)
print(a, b)


def foo3(x, y) :
    return 2 * x + y

print (foo3(7, 10))
print (foo3(y = 10, x = 7))


