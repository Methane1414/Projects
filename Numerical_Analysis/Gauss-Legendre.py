import numpy as np

a,b =1,2

def f(xk):  
    t=xk
    #Transformation from [1,2] to [-1,1]
    x=((b-a)/2)*t + ((b+a)/2)
    f=2*x/(1+pow(x,4))
    return f

#Jacobian of mapping from [1,2] to [-1,1]
J = (b-a)/2
#1 point formula
I1 = J*2*f(0)
print("I1=",I1)

#2 Point Formula
I2 = J*(f(-1/np.sqrt(3)) + f(1/np.sqrt(3)))
print("\nI2=",I2)

#3 point Formula
I3 = J*(1/9*(5*f(-np.sqrt(3/5)) + 8*f(0) + 5*f(np.sqrt(3/5))))
print("\nI3=",I3)

#exact solution
I =np.arctan(4)- np.pi/4
print("\nI=",I)
