####################################################################
# Advanced Machine Learning - MDS
# Lluís A. Belanche

# LAB 5: Hand-made kernels
#        Novelty detection with the SVM
#        {LiblineaR}
# version of November 2022
####################################################################


#################################################
# EXAMPLE 1: Creating our own hand-made kernels
#################################################

library(kernlab)

## Now we are going to understand better the class 'kernel' in {kernlab}

## An object of class 'kernel' is simply a function with an additional slot 'kpar' for kernel parameters

## We can start by looking at two built-in kernels to see how they were created
vanilladot
rbfdot

## Let us create a RBF kernel and look at its attributes
rbf <- rbfdot(sigma=10^-2)

rbf
rbf@.Data # the kernel function itself
rbf@kpar  # the kernel paramters
rbf@class # the class

## Once we have a kernel object such as rbf, we can do several things, eg:

N <- 20
M <- 25
x <- matrix( rnorm(N*M,mean=0,sd=1), N, M)
t <- rbinom(n=20, size=1, prob=0.5)

## 1) Compute the kernel between two vectors
rbf(x[1,],x[2,])

## 2) Compute a kernel matrix between two sets of vectors
K <- kernelMatrix(rbf,x[1:5,],x[6:20,])
dim(K)

## or between a set of vectors with itself (this is the typical use)
K <- kernelMatrix(rbf,x)
dim(K)

## 3) Obviously we can train a SVM
m <- ksvm (x,t, kernel=rbf, type="C-svc", scale=c())

## Now we are going to make our own kernel and integrate it in kernlab: 

## To make things simple, we simply create a normalized "own version" of the linear kernel:

kval <- function(x, y = NULL) 
{
  if (is.null(y)) {
    crossprod(x)
  } else {
    crossprod(x,y) / sqrt(crossprod(x)*crossprod(y))
  }
}

## We then create the kernel object as follows
## Remember this kernel has no parameters, so we specify kpar=list(), an empty list

mylinearK <- new("kernel",.Data=kval,kpar=list())

## this is what we did
str(mylinearK)

## Now we can call different functions of kernlab right away

mylinearK (x[1,],x[2,])

kernelMatrix (mylinearK,x[1:5,])

m <- ksvm(x,t, kernel=mylinearK, type="C-svc")

## As a final example, we make a kernel that evaluates a precomputed kernel
## This is particularly useful when the kernel is very costly to evaluate
## so we do it once and store in a external file, for example

## The way we do this is by creating a new "kernel" whose parameter is a precomputed kernel matrix K
## The kernel function is then a function of integers i,j such that preK(i,j)=K[i,j]

mypreK <- function (preK=matrix())
{
  rval <- function(i, j = NULL) {
    ## i, j are just indices to be evaluated
    if (is.null(j)) 
    {
      preK[i,i]
    } else 
    {
      preK[i,j]
    }
  }
  return(new("kernel", .Data=rval, kpar=list(preK = preK)))
}

## To simplify matters, suppose we already loaded the kernel matrix from disk into
## our matrix 'myRBF.kernel' (the one we created at the start)

## We create it
myprecomputed.kernel <- mypreK(myRBF.kernel)
str(myprecomputed.kernel)

## We check that it works

myRBF.kernel[seq(5),seq(5)]                 # original matrix (seen just as a matrix)
kernelMatrix(myprecomputed.kernel,seq(5))   # our kernel

## We can of course use it to train SVMs

svm.pre <- ksvm(seq(N),t, type="C-svc", kernel=myprecomputed.kernel, scale=c())
svm.pre

## which should be equal to our initial 'svm1'
svm1

## compare the predictions are equal
p1 <- predict (svm.pre, seq(N))
p2 <- predict (svm1)[1:N]
table(p1,p2)

################################################
# EXAMPLE 2: Modelling 2D Outlier Detection data
################################################

## Now we switch to the {e1071} package
library(e1071)

## just a variation of the built-in example ...

N <- 1000

X <- data.frame(a = rnorm(N), b = rnorm(N))
attach(X)

# default nu = 0.5, to see how it works; gamma is the usual GRBF parameter
(m <- svm(X, gamma = 0.1))

newdata <- data.frame(a = c(0, 2.5,-2,-2), b = c(0, 2.5,2,0))

# visualize:
plot(X, col = 1:N %in% m$index + 1, xlim = c(-5,5), ylim=c(-5,5))
text(newdata[,1],newdata[,2],labels=row.names(newdata),pch = "?", col = 3, cex = 2)

# Areas marked by number are potential areas of outlier localization

# test:
!predict (m, newdata) # TRUE stands for predicted outliers, FALSE for the opposite

# now redo with nu = 0.01 (more in accordance with outlier detection)

(m <- svm(X, gamma = 0.1, nu = 0.01))

# visualize:
plot(X, col = 1:N %in% m$index + 1, xlim = c(-5,5), ylim=c(-5,5))
text(newdata[,1],newdata[,2],labels=row.names(newdata),pch = "?", col = 3, cex = 2)

# test:
!predict (m, newdata)


################################################
# EXAMPLE 3: Linear Predictive Models Based on 
#             the LIBLINEAR C/C++ Library
################################################

library(LiblineaR)

# A wrapper around the LIBLINEAR C/C++ library for machine learning 
# (available at <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>). 
# LIBLINEAR is a simple library for solving large-scale regularized linear 
# classification and regression. It currently supports L2-regularized classification 
# (such as logistic regression, L2-loss linear SVM and L1-loss linear SVM) as well 
# as L1-regularized classification (such as L2-loss linear SVM and logistic regression)
# and L2-regularized support vector regression (with L1- or L2-loss). The main features
# of LiblineaR include multi-class classification 
# (one-vs-the rest, and Crammer & Singer method), cross validation for model selection, 
# probability estimates (logistic regression only) or weights for unbalanced data. 
# The estimation of the models is particularly fast as compared to other libraries.

# More info at https://www.csie.ntu.edu.tw/~cjlin/liblinear/FAQ.html