setwd('C:/Users/wangx/OneDrive - Georgia State University/Spring 2019/8700 Categorical Data Analysis/HW')
install.packages('pROC')
library(car)
library(glmnet)
library(randomForest)
library(pROC)
###cleaning
##training data
data.raw=read.table('project.1.data.2.train.txt',sep=',',na.strings = '?')
head(data.raw)
str(data.raw)
apply(is.na(data.raw),2,sum)
id_no_miss=(apply(is.na(data.raw),1,sum)==0)
data.train=data.raw[id_no_miss,]
sum(is.na(data.train))
str(data.train)
data.train$V16=as.character(data.train$V16)
data.train$V16
data.train$V16[data.train$V16=='+']=1
data.train$V16[data.train$V16=='-']=0
data.train$V16=as.factor(data.train$V16)
data.train$V16
colnames(data.train)[16]="Y"
str(data.train)
##testing data
data.raw1=read.table('project.1.data.2.test.txt',sep=',',na.strings = '?')
head(data.raw1)
str(data.raw1)
apply(is.na(data.raw1),2,sum)
data.test=data.raw1
data.test$V16=as.character(data.test$V16)
data.test$V16
data.test$V16[data.test$V16=='+']=1
data.test$V16[data.test$V16=='-']=0
data.test$V16=as.factor(data.test$V16)
data.test$V16
colnames(data.test)[16]="Y"
str(data.test)

###data exploration
cor(data.frame(data.train$V2,data.train$V3,data.train$V8,data.train$V11,data.train$V14,data.train$V15))
##numerical data
par(mfrow=c(2,3))
hist(data.train$V2)
hist(data.train$V3)
hist(data.train$V8)
hist(data.train$V11)
hist(data.train$V14)
hist(data.train$V15)
##categorical data/bar charts
par(mfrow=c(3,3))
plot(data.train$V1)
plot(data.train$V4)
plot(data.train$V5)
plot(data.train$V6)
plot(data.train$V7)
plot(data.train$V9)
plot(data.train$V10)
plot(data.train$V12)
plot(data.train$V13)
##scatterplot

scatterplotMatrix(data.train,plot.points=F, diagonal="histogram")

##-----logistic regression method-----##

#model selection
full=glm(Y~.,data=data.train,family="binomial")
null=glm(Y~1,data=data.train,family = "binomial")
select=step(null,scope=list(lower=null,upper=full),direction="forward")
data1=(select$model)
pred=colnames(data1)[-1]
m=length(pred)
pred.names=NULL
for(i in 1:m){
  pred.names=c(pred.names, pred[i])
}
for(i in 1:(m-1)){
  for(j in (i+1):m){
    pred.names=c(pred.names, paste(pred[i], ":", pred[j]))
  }
}
Formula <- formula(paste("Y ~ ",
                         paste(pred.names, collapse=" + ")))
Formula
fit.full.1=glm(Formula,data=data1,family=binomial)
fit.null.1=glm(Y~1,data=data1,family=binomial)
select.1=step(fit.null.1,scope=list(lower=fit.null.1,upper=fit.full.1),direction='forward')
fit.1=glm(Y ~ V9 + V15 + V11 + V6 + V4 + V14 + V8 + V9:V15 + V9:V11 + V4:V14 + 
            V4:V8 + V14:V8,data=data1,family=binomial)
summary(fit.1)
fit.1$coefficients
Anova(fit.1)
vif(fit.1)
##calculate ROC and AUCs using LOO method
pi0=seq(1,0,length.out = 500)
nsample=nrow(data.train)
roc.1=NULL
for (k in 1:length(pi0)){
  nsample=nrow(data.train)
  n1_11=0
  n1_10=0
  n1_01=0
  n1_00=0
  for (i in 1:nsample){
    training=data.train[-i,]
    test=data.train[i,]
    fit.1.training=glm(fit.1,data=data.train,family = binomial)
    if(predict(fit.1.training, test, type='response')>=pi0[k]){
      Y.pred.1=1
    }else{
      Y.pred.1=0
    }
    if((test$Y==1) & (Y.pred.1==1)){
      n1_11=n1_11+1
    }
    if((test$Y==1) & (Y.pred.1==0)){
      n1_10=n1_10+1
    }
    if((test$Y==0) & (Y.pred.1==1)){
      n1_01=n1_01+1
    }
    if((test$Y==0) & (Y.pred.1==0)){
      n1_00=n1_00+1
    }
  }
  sen1=n1_11/(n1_10+n1_11)
  spe1=n1_00/(n1_00+n1_01)
  roc.1=rbind(roc.1,c(1-spe1,sen1))
}
plot(roc.1,type='s',xlim=c(0,1),ylim = c(0,1),col='red',lwd=3,main='ROC curve',xlab='1-Specificity',ylab='Sensitivity')
str(roc.1)
auc.1=sum(roc.1[-500,2]*(roc.1[-1,1]-roc.1[-500,1]))
auc.1
pred.accuracy.1=roc.1[,2]*sum(data.train$Y==1)/nsample+(1-roc.1[,1])*sum(data.train$Y==0)/nsample
##cut-off point of pi0
which.max(pred.accuracy.1)
pi0.1=pi0[(which.max(pred.accuracy.1))]
pi0.1
##prediction accuracy/training accuracy
pred.train=pred.accuracy.1[258]
pred.train
####prediction accuracy/test accuracy
nsample=nrow(data.test)
n_correct=0
for (i in 1:nsample){
  test=data.test[i,]
  if(predict(fit.1, test, type='response')>pi0.1){
    Y.pred=1
  }else{
    Y.pred=0
  }
  if(test$Y==Y.pred){
    n_correct=n_correct+1
  }
}
pred.test=n_correct/nsample
pred.test

##model regularity - Ridge penalty
x=cbind(data.train$V9,data.train$V15,data.train$V11,data.train$V6,data.train$V4,data.train$V14,data.train$V8)
y=data.train$Y
fit.0=glmnet(x=x, y=y, family='binomial',alpha=0)
lambda.seq=fit.0$lambda
pi0=seq(0,1, length.out = 100)
correct.num=matrix(0, length(lambda.seq), length(pi0))
nsample=nrow(data.train)
for(i in 1:nsample){
  print(c("i=", i))
  x.train=x[-i,]
  x.test=matrix(x[i,],1,ncol(x))
  y.train=y[-i]
  y.test=y[i]
  fit.0.training=glmnet(x.train, y.train, family="binomial", alpha=0)
  for(j in 1:length(lambda.seq)){
    pred.prob=predict(fit.0.training, newx=x.test, s=lambda.seq[j], type="response")
    for(k in 1:length(pi0)){
      if(pred.prob>=pi0[k]){
        Y.pred.1=1
      }else{Y.pred.1=0}
      if((y.test==1)&(Y.pred.1==1)){
        correct.num[j,k]=correct.num[j,k]+1
      }
      if((y.test==0)&(Y.pred.1==0)){
        correct.num[j,k]=correct.num[j,k]+1
      }
    }
  }
}
#output
accuracy=correct.num/nsample
max(accuracy)
which(accuracy==max(accuracy), arr.ind=TRUE)
j=92
k=55
lambda=lambda.seq[j]
pi0.2=pi0[k]
lambda
pi0.2

#prediction accuracy after regularity/test accuracy
nsample=nrow(data.test)
n_correct=0
for (i in 1:nsample){
  test=data.test[i,]
  if(predict(fit.1, test, s = lambda, type='response')$fit>pi0.2){
    Y.pred=1
  }else{
    Y.pred=0
  }
  if(test$Y==Y.pred){
    n_correct=n_correct+1
  }
}
n_correct/nsample
pred.test1=n_correct/nsample
pred.test1

##----randomforest method----##
# Create a Random Forest model with default parameters
set.seed(12)
model1=randomForest(Y~., data = data.train, importance=TRUE, keep.forest = TRUE)
model1

#Checking classification accuracy
# *test data is actually used as validation set, instead of splitting training set into train and validation sets.
table(predict(model1,data.train),data.train$Y)
combined=rbind(data.train,data.test)
str(combined)
data.test=combined[548:647,]
predValid = predict(model1, newdata = data.test, type = "class")
#test accuracy
mean(predValid == data.test$Y)
table(predValid, data.test$Y)
# To check important variables
importance(model1)
varImpPlot(model1)

# Fine tuning parameters of Random Forest model
#if using test accuracy as model evaluation
# using for loop to find optimal mtry
set.seed(12)
a=as.numeric()
for (i in 2:15){
  model2= randomForest(Y~ .,data = data.train, ntree=500, mtry=i, importance=TRUE)
  predValid2 = predict(model2, newdata = data.test, type = "class")
  a[i-1]= mean(predValid2 == data.test$Y)
}
a
plot(2:15,a)
#if using OOB as model evaluation
# - tuneRF in the randomForest package
x.train = data.train[,1:15]
y.train = data.train[,16]
set.seed(12)
tuneRF(
  x = x.train,
  y = y.train,
  ntreeTry = 500,
  mtryStart = 3,
  stepFactor = 1.5,
  improve = 0.01,
  trace = FALSE
)

set.seed(12)
model_RFtuned= randomForest(Y~ .,data = data.train, ntree=500, mtry=2, importance=TRUE)
model_Mtuned= randomForest(Y~ .,data = data.train, ntree=500, mtry=7, importance=TRUE)
model1
model_RFtuned
model_Mtuned
predValid_RFtuned = predict(model_RFtuned, newdata = data.test, type = "class")
mean(predValid_RFtuned == data.test$Y)
table(predValid_RFtuned, data.test$Y)
predValid_Mtuned = predict(model_Mtuned, newdata = data.test, type = "class")
mean(predValid_Mtuned == data.test$Y)
table(predValid_Mtuned, data.test$Y)
rf.roc_RFtuned = roc(data.train$Y,model_RFtuned$votes[,2], auc = TRUE, plot = TRUE)
plot(rf.roc_RFtuned)
auc(rf.roc_RFtuned)
rf.roc_Mtuned = roc(data.train$Y,model_Mtuned$votes[,2])
plot(rf.roc_Mtuned)

# https://stackoverflow.com/questions/30366143/how-to-compute-roc-and-auc-under-roc-after-training-using-caret-in-r
