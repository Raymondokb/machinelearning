library(caret)
library(kernlab)
data(spam)
inTrain <- createDataPartition(y = spam$type, p =0.75, list=FALSE)
training <- spam[inTrain, ]
testing <- spam[-inTrain,]
dim(training)

set.seed(32343)
modelFit <- train(type ~., data = training, method = "glm") #generalized linear model
modelFit
modelFit$finalModel

prediction <- predict(modelFit, newdata = testing)
prediction
confusionMatrix(prediction, testing$type)

###
#K-fold
#Cross validation
set.seed(32323)
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=TRUE)
sapply(folds,length)
folds[[1]][1:10]
###
#Return test
set.seed(32323)
folds <- createFolds(y=spam$type, k=10, list=TRUE, returnTrain=FALSE) #returns just testset
sapply(folds, length)
folds[[1]][1:20]

###
#Resampling
folds <- createResample(y=spam$type, times=10, list=TRUE)
sapply(folds, length)
folds[[1]][1:10]

###
#Time slices
tme <- 1:1000
folds <- createTimeSlices(y=tme, initialWindow = 20, horizon = 10)
names(folds)
folds$train[[1]]
folds$test[[1]]


####Train Options
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y = spam$type, p =0.75, list=FALSE)
training <- spam[inTrain, ]
testing <- spam[-inTrain,]
modelFit <- train(type ~., data = training, method = "glm")

args(trainControl)

set.seed(1235)
modelFit2 <- train(type ~., data=training, method="glm")
modelFit2

#predictingPlot
library(ISLR); library(ggplot2); library(caret)
data(Wage)
summary(Wage)
inTrain <- createDataPartition(y=Wage$wage, p = 0.7, list=FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain, ]
dim(training); dim(testing)

featurePlot(x=training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")

qplot(age,wage,data=training)
qplot(age,wage,colour=jobclass,data=training)
qq <- qplot(age,wage,colour=education,data=training)
qq + geom_smooth(method='lm',formula=y~x)

library(Hmisc); library(gridExtra)
cutWage <- cut2(training$wage,g=3) #cut into different categories; g = num of groups based on quantile groups
table(cutWage)
p1 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot"))
p1
p2 <- qplot(cutWage,age, data=training,fill=cutWage,
            geom=c("boxplot","jitter")) #jitter
grid.arrange(p1,p2,ncol=2)

t1 <- table(cutWage,training$jobclass)
t1
prop.table(t1,1)
qplot(wage,colour=education,data=training,geom="density")
#^plot continuous predictors


###Preprocessing predictor variables
#Transforming predictor variables
#standardizing -> (x - mean(x))/sd(x) [if mean(x) >> sd(x)]
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type,
                               p=0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain,]
hist(training$capitalAve,main="",xlab="ave. capital run length")
#also use preProcessFunction
preObj <- preProcess(training[,-58],method=c("center","scale"))
trainCapAveS <- predict(preObj,training[,-58])$capitalAve
mean(trainCapAveS)

preObj <- preProcess(training[-58], method =c("BoxCox") )
trainCapAveS <- predict(preObj,training[-58])$capitalAve
par(mfrow=c(1,2)); hist(trainCapAveS);  qqnorm(trainCapAveS)
#^still there's a hump at theleft histogram, and flat at qq plot

#If there are missing values in the data
set.seed(13343)
#Make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1],size=1, prob=0.05)==1
training$capAve[selectNA] <- NA

#Impute and standardize
preObj <- preProcess(training[,-58], method="knnImpute")
capAve <- predict(preObj, training[,-58])$capAve

###################
###################

# Covariate creation
library(kernlab); data(spam)
spam$capitalAveSq<- spam$capitalAve^2

# Level 1: Raw data -> covariates
# Level 2: Tidy covariates -> New covariates

library(ISLR)
library(caret); data(Wage)
inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
training <- Wage[inTrain, ]; testing <- Wage[-inTrain, ]

# Common covariates to add, dummy variables
table(training$jobclass)
dummies <- dummyVars(wage ~ jobclass, data=training)
head(predict(dummies, newdata=training))

# Remove zero covariates
#Identifies and removes almost constant columns
nsv <- nearZeroVar(training, saveMetrics=TRUE)
nsv
# see the freqRatio, neear zero are bad and should be removed
# ex: sex, region to be removed

# Spline basis
library(splines)
bsBasis <- bs(training$age, df=3)
bsBasis

lm1 <-lm(wage ~bsBasis, data=training)
plot(training$age, training$wage, pch=19, cex = 0.5)
points(training$age, predict(lm1, newdata=training),col="red", pch=19, cex=0.5)

#splines on the test set
predict(bsBasis, age=testing$age)


### Preprocessin with Principal Components Analysis (PCA)
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y=spam$type, p = 0.75, list=FALSE)
training <- spam[inTrain,]
testing <- spam[-inTrain, ]

M <- abs(cor(training[,-58])) # n by n matrix
diag(M) <- 0 #delete the correlation 1 entries at diagonal
which(M > 0.8, arr.ind=T)

# correlated predictors
names(spam)[c(34, 32)]
plot(spam[,34], spam[,32])
# ^ almost 1 on 1 correlation

smallSpam <- spam[,c(34,32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[,1], prComp$x[,2])
# ^ a more straight line as desired
# ^$sdev are the eigenvalues

prComp$rotation
# ^ square rotation matrix

# PCA on SPAM data
typeColor <- ((spam$type=="spam")*1 + 1)
prComp <- prcomp(log10(spam[, -58] + 1))
plot(prComp$x[,1], prComp$x[,2], col=typeColor, xlab="PC1", ylab="PC2")
# ^ a little divided,but we want black and red to completely separate

# PCA with caret
preProc <- preProcess(log10(spam[, -58]+1), method="pca", pcaComp = 3)
spamPC <- predict(preProc, log10(spam[,-58] + 1))
plot(spamPC[, 1], spamPC[,2],col=typeColor)
# ^ a little better

preProc <- preProcess(log10(training[, -58]+1), method="pca", pcaComp=2)
trainPC <- predict(preProc, log10(training[,-58]+ 1))
modelFit <- train(x = trainPC, y = training$type,method="glm")
#?
testPC <- predict(preProc, log10(testing[, -58]+ 1))
confusionMatrix(testing$type, predict(modelFit, testPC))


# Alternative set # of PC
#?
modelFit <- train(x = log10(training[,-58]+1), y=training$type, method="glm", preProcess="pca")
confusionMatrix(testing$type, predict(modelFit, testing))


### Prediting with regression
library(caret); data(faithful); set.seed(333)
inTrain <- createDataPartition(y=faithful$waiting, p = 0.5, list = FALSE)
trainFaith <- faithful[inTrain, ]; testFaith <- faithful[-inTrain, ]
head(trainFaith)

plot(trainFaith$waiting, trainFaith$eruptions, pch=19,col="blue",xlab="waiting",ylab="Duration")

# Fit a linear model
lm1 <- lm(eruptions ~ waiting, data = trainFaith)
summary(lm1)

plot(trainFaith$waiting, trainFaith$eruptions, pch=19,col="blue",xlab="waiting",ylab="Duration")
lines(trainFaith$waiting, lm1$fitted.values, lwd=3)

# Predict a new value
newdata <- data.frame(waiting=80)
predict(lm1, newdata)


#####################
#####################

# Regression with multiple covariates
# *exploring dataset and identify which variables to use
library(ISLR); library(kernlab)
# ^INTRODUCTION to statistical learning
library(ggplot2); library(caret);
data(Wage); Wage <- subset(Wage, select= -c(logwage)) # we will be predicting logwage, so removed
summary(Wage)

inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain, ]
dim(training); dim(testing)


#do all exploration on TRAINING set only
#looking only at trainign set, not touching testing set
featurePlot(x=training[, c("age","education","jobclass")],
            y= training$wage,
            plot="pairs")

qplot(age,wage,data=training)
#appears to be some kind of trend
qplot(age,wage,color=jobclass,data=training)
#color by another variable other than the 2 variables
qplot(age,wage,color=education,data=training)

modFit <- train(wage ~ age + jobclass + education,
                method = "lm", data = training)
finMod <- modFit$finalModel
print(finMod)

plot(finMod, 1, pch = 19, cex=0.5)

qplot(finMod$fitted.values, finMod$residuals,color = race, data=training)

# plot by index, rsidual
plot(finMod$residuals, pch = 19)
# if we see a trend wrt index, suggests there is a variable missing from model

pred <- predict(modFit, testing) 
qplot(wage, pred, color = year, data = testing)
#^ ideally a straight line

#to use all covariates
#well after checking summary(modFitAll for wage ~., lots of empty vars)
#fine tuning it to exclude the NA variables
#ex: sex as two categories but only males in this dataset though
#removing region and sex
modFitAll <- train(wage ~year+age+maritl+race+education+jobclass+health+health_ins
                   , data = training, method="lm")

pred <- predict(modFitAll, testing)
qplot(wage, pred, data=testing)


#Quiz time
#1
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]

#2
library(Hmisc) #use cut2() for turning cts variables into factors
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
qplot(y=CompressiveStrength, color=Cement, data = training)
qplot(y=CompressiveStrength, color=BlastFurnaceSlag, data = training)
qplot(y=CompressiveStrength, color=FlyAsh, data = training)
qplot(y=CompressiveStrength, color=Water, data = training)
qplot(y=CompressiveStrength, color=Superplasticizer, data = training)
qplot(y=CompressiveStrength, color=CoarseAggregate, data = training)
qplot(y=CompressiveStrength, color=FineAggregate, data = training)
qplot(y=CompressiveStrength, color=Age, data = training)

#3
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

qplot(x=Superplasticizer, data=training, bins=30)
#lots of zeroes

#4
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
library(dplyr)
q4 <- training %>% select(starts_with("IL"))
preproc2 <- preProcess(q4, thresh = 0.9)     
#checks eigenvalues
q4sdev2 <- preproc2$std[order(preproc2$std,decreasing=TRUE)]^2
q4sdev2/sum(q4sdev2)     
cumsum(q4sdev/sum(q4sdev2))

#####4 answer
IL_col_idx <- grep("^[Ii][Ll].*", names(training))
preObj <- preProcess(training[, IL_col_idx], method=c("center", "scale", "pca"), thresh=0.9)
preObj

#5
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# create predictive models
# diagnosis column is the y column
#using all IL columns
q5 <- cbind(training %>% select(starts_with("IL")), diagnosis  =training$diagnosis)
modFit1 <- train(diagnosis ~ ., data = q5, method="glm")
pred <- predict(modFit1, testing[,-1])
confusionMatrix(pred, testing$diagnosis)

# now using just the 8 variables capturing all 90% of the data
#asdfghjkl forgot to use PCA variables
pca2 <-names(q4sdev2/sum(q4sdev2))[1:8]
q5b <- training[, pca2]; q5b <- cbind(q5b,diagnosis = training$diagnosis)
q5b_test <- testing[, pca2]; q5b_test <- cbind(q5b_test, diagnosis = testing$diagnosis)
pca_data <- preProcess(q5b[, -9], method=c('center','scale','pca'),thresh=0.8)
pca_train_pred <- predict(pca_data, q5b[, -9])
pca_test_pred <- predict(pca_data, q5b_test[, -9])

pca_train_pred <- cbind(pca_train_pred, diagnosis2 = q5b$diagnosis)

modFit2 <- train(diagnosis2 ~., data=pca_train_pred, method="glm")
pred2test <- predict(modFit2, pca_test_pred)
confusionMatrix( testing$diagnosis,pred2test)

#wtrf
x <- 1:100/10
y <- 1:100
z <- x %o% y
z <- z + .2*z*runif(25) - .1*z

library(rgl)
persp3d(x, y, z, col="skyblue")

# Trees Decision
library(kernlab); library(caret); 
data(iris)
inTrain <- createDataPartition(iris$Species, p=0.7, list=FALSE)
training <- iris[inTrain,]; testing <- iris[-inTrain,]
qplot(Petal.Width, Sepal.Width, color=Species, data=training)

modFit <- train(Species ~., method="rpart", data=training)
print(modFit$finalModel)
plot(modFit$finalModel, uniform =TRUE, main="Classification Error")
text(modFit$finalModel, use.n=TRUE,  all=TRUE, cex = 0.8)
#^dendogram

library(rattle)
fancyRpartPlot(modFit$finalModel)
# ^doesnt work, nvm

test_pred <- predict(modFit, newdata=testing)
confusionMatrix(testing$Species,test_pred)

#Bagging: bootstrap aggregating
#_ resample cases and re-evaluate predictions
#_ average or majority vote

library(ElemStatLearn)
data(ozone, package="ElemStatLearn")
ozone <- ozone[order(ozone$ozone),] #outcome: ozone variable

ll <- matrix(NA,nrow=10,ncol=155)
for(i in 1:10){
     ss <- sample(1:dim(ozone)[1],replace=T)
     ozone0 <- ozone[ss,]; ozone0 <- ozone0[order(ozone0$ozone),]
     loess0 <- loess(temperature ~ ozone,data=ozone0,span=0.2)
     ll[i,] <- predict(loess0,newdata=data.frame(ozone=1:155))
}
plot(ozone$ozone,ozone$temperature,pch=19,cex=0.5)
for(i in 1:10){lines(1:155,ll[i,],col="grey",lwd=2)}
lines(1:155,apply(ll,2,mean),col="red",lwd=2)

predictors = data.frame(ozone=ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictors, temperature, B = 10,
               bagControl = bagControl(fit = ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))

plot(ozone$ozone,temperature,col='lightgrey',pch=19)
points(ozone$ozone,predict(treebag$fits[[1]]$fit,predictors),pch=19,col="red")
points(ozone$ozone,predict(treebag,predictors),pch=19,col="blue")

ctreeBag$fit
ctreeBag$pred
ctreeBag$aggregate
# Bagging is mostly used fo rnonlinear models
#resample data, refit nonlinear model, re-average all of them

library(ISLR); library(ggplot2); library(caret)
Wage <- subset(Wage, select= -c(logwage))
inTrain <- createDataPartition(Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]

modFit <- train(wage ~., method = "gbm", data = training, verbose = FALSE)
print(modFit)

qplot(predict(modFit, testing), wage, data=testing)

### Model Based prediction

data(iris); library(ggplot2)
table(iris$Species)
     
inTrain <- createDataPartition(y = iris$Species, p = 0.7, list=FALSE)
training <- iris[inTrain, ]; testing <- iris[-inTrain, ]

modlda <- train(Species ~., data=training, method="lda")
modnb <- train(Species ~., data=training, method="nb") #nb naive bayes
plda <- predict(modlda, testing); pnb <- predict(modnb, testing)
table(plda, pnb)
#all agree except for one value
equalPredictions <- (plda==pnb)
qplot(Petal.Width, Sepal.Width, color = equalPredictions, data = testing)

#Quiz time
#1
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)
set.seed(125)
training<- subset(segmentationOriginal,Case=="Train")
testing<- subset(segmentationOriginal,Case=="Test")
library(rattle)

modFit <- train(Class ~., method="rpart", data=training)

modFit$finalModel
fancyRpartPlot(modFit$finalModel)

#3
library(pgmm)
data(olive)
olive = olive[,-1]

newdata = as.data.frame(t(colMeans(olive)))

#using classification trees
modelFit2<- train(Area~., data=olive,method="rpart")
result2<- predict(modelFit2,newdata)
result2


#4
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

set.seed(13234)

modFit <- train(chd ~age+alcohol+obesity+tobacco+typea+ldl, data=trainSA, method = "glm", family="binomial")
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
result3_train <-  missClass(trainSA$chd,predict(modFit, trainSA))
result3_test <-  missClass(testSA$chd,predict(modFit,testSA))
result3_train
result3_test

#5
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)
modFit <- train(y ~., data= vowel.train, method = "rf")
result <- predict(modFit,vowel.test)
varImp(modFit)



##################
##################
#Is it project time? I think it's project time#
##################
##################

training <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
testing <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))

#training has classe column, what does it mean?
# A,B,C,D,E correspond to different set of actions

library(kernlab); library(caret); library(rpart); library(gbm)
# to single out the predictor variables to be used as must are incomplete data
str(training)
set.seed(2)

#Now for a fleeseeks box because a lot of cleaning needs to be done
cols_na <- nearZeroVar(training)
training <- training[, -cols_na]
#use any(is.na(cols)) to remove columns with NA inside them 
#use any(x=="") to remove 
#remove non-useful like name etc, to identify -.- (X, timestamp, window)

keep_index <- !sapply(training, function(x) any(is.na(x)))
training <- training[, keep_index]
keep_index <- sapply(colnames(training), function(x) !grepl("X|time|window",x))
training <- training[, keep_index]

#now do the same for test
     

#now do the same for test
keep_index <- !sapply(testing, function(x) any(is.na(x)))
testing <- testing[, keep_index]
keep_index <- sapply(colnames(testing), function(x) !grepl("X|time|window",x))
testing <- testing[, keep_index]
dim(testing)

#Now to move to a Meeseek's box
#Create train and test data from training
index_train <- createDataPartition(training$classe, p = 0.7, list=FALSE)
training1 <- training[index_train, ]
testing1 <- training[-index_train, ]

#random forest, decision trees, naive bayes, lda, gbm, pca

modFit <- train(classe ~., method = "lda", data = training1)
pred_lda <- predict(modFit, testing1)
confusionMatrix(pred_lda, testing1$classe)

modFit_rf <- train(classe ~ ., method = "rf", data=training1, trControl=trainControl(method="cv", 5), ntree=250)
pred_rf <- predict(modFit_rf, testing1)
confusionMatrix(pred_rf, testing1$classe)

modFit_gbm <- train(classe ~ ., method = "gbm", data=training1 )
pred_gbm <- predict(modFit_gbm, testing1)
confusionMatrix(pred_rpart, testing1$classe)

modFit_rpart <- train(classe)

names(getModelInfo())
