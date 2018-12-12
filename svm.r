#Call data
data <- read.table("pb2.txt")

Data <- as.matrix(data, ncol=5)
Y <- Data[,1]
X <- as.matrix(Data[,2:5])
n <- length(Y)
for (i in 1:n){
  if (Y[i] > 1){
    Y[i]<--1
  }
}



require('quadprog')
## Defining the Gaussian kernel
rbf_kernel <- function(x1,x2,gamma){
  K<-exp(-(1/gamma^2)*t(x1-x2)%*%(x1-x2))
  return(K)
}

svmtrain <- function(X,Y,C=Inf, gamma=1.5,esp=1e-10){
  
  N<-length(Y)
  Dm<-matrix(0,N,N)
  X<-as.matrix(X);Y<-as.vector(Y)
  
  for(i in 1:N){
    for(j in 1:N){
      Dm[i,j]<-Y[i]*Y[j]*rbf_kernel(X[i,],X[j,],gamma)
    }
  }
  Dm<-Dm+diag(N)*1e-12 # adding a very small number to the diag, some trick
  
  dv<-t(rep(1,N))
  meq<-1
  Am<-cbind(matrix(Y,N),diag(N))
  bv<-rep(0,1+N) # the 1 is for the sum(alpha)==0, others for each alpha_i >= 0
  if(C!=Inf){
    # an upper bound is given
    Am<-cbind(Am,-1*diag(N))
    bv<-c(cbind(matrix(bv,1),matrix(rep(-C,N),1)))
  }
  alpha_org<-solve.QP(Dm,dv,Am,meq=meq,bvec=bv)$solution
  alphaindx<-which(alpha_org>esp,arr.ind=TRUE)
  alpha<-alpha_org[alphaindx]
  nSV<-length(alphaindx)
  if(nSV==0){
    throw("QP is not able to give a solution for these data points")
  }
  Xv<-X[alphaindx,]
  Yv<-Y[alphaindx]
  Yv<-as.vector(Yv)
  # choose one of the support vector to compute b. for safety reason,
  # select all support vectors and find average of the b'sthe one with max alpha********
    
  b <- numeric(nSV)
  ayK <- numeric(nSV)
  for (i in 1:nSV){
    for (m in 1:nSV){
      ayK[m] <- alpha[m]*Yv[m]*rbf_kernel(Xv[m,],Xv[i,],gamma)
    }
    b[i]<-Yv[i]-sum(ayK)
    
  }
  w0 <- mean(b)
  
  #list(alpha=alpha, wstar=w, b=w0, nSV=nSV, Xv=Xv, Yv=Yv, gamma=gamma)
  list(alpha=alpha, b=w0, nSV=nSV, Xv=Xv, Yv=Yv, gamma=gamma)
}

rbf_kernel(X[1,],X[2,],gamma=3)

### Predict the class of an object X



svmpredict <- function(x,model){
  alpha<-model$alpha
  b<-model$b
  Yv<-model$Yv
  Xv<-model$Xv
  nSV<-model$nSV
  gamma<-model$gamma
  #wstar<-model$wstar
  #result<-sign(rbf_kernel(wstar,x,gamma)+b)
  ayK <- numeric(nSV)
  for(i in 1:nSV){
    ayK[i]<-alpha[i]*Yv[i]*rbf_kernel(Xv[i,],x,gamma)
  }
  result <- sign(sum(ayK)+b)
  return(result)
}


#Problem 2.2

GAMMA <- c( 1:100)

NSV <- numeric(100)


for(i in 1:100){

  NSV[i] <- svmtrain(X,Y,C=2, gamma= i ,esp=1e-10)$nSV

}


plot(GAMMA,NSV)

C <- c( 1:10)

NSV_C <- numeric(10)


for(i in 1:10){

  NSV_C[i] <- svmtrain(X,Y,C=i, gamma= 20 ,esp=1e-10)$nSV

}


plot(C,NSV_C)

#Prediction

model23 <- svmtrain(X,Y,C=8, gamma=20,esp=1e-10)
model23

z <- c(18,17,33,26)

svmpredict(z,model23)



