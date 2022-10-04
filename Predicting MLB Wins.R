#Filter to 2004 through 2019
library(Lahman)
View(Teams)
Teams.Since.2004 <- Teams[Teams$yearID >= 2004,]
View(Teams.Since.2004)
Teams.df <- Teams.Since.2004[Teams.Since.2004$yearID < 2020,]
tail(Teams.df)

#Prep data for Linear Model
#Includes getting rid of games, losses as intuitive and do not matter
Teams.df.lm <- Teams.df[,-c(1:8, 10:14, 41:48)]
View(Teams.df.lm)

#Partition data
nrow(Teams.df.lm) #480 rows; 75/25 split would mean 360/120 for training data and validation data
RNGkind(sample.kind = "Rounding") 
set.seed(123)  #set seed for reproducing the partition later
train.index <- sample(c(1:dim(Teams.df.lm)[1]), dim(Teams.df.lm)[1]*0.75)  
train.df <- Teams.df.lm[train.index, ]
valid.df <- Teams.df.lm[-train.index, ]
head(train.df)
head(valid.df)

#Build model - try multiple methods
teams.lm <- lm(W ~ ., data = train.df)
summary(teams.lm)

#Exhaustive Search
teams.lm.exhaus <- lm(W ~ ., data = train.df)
summary(teams.lm.exhaus) #summary of the model, Adjusted R-Squared of .9379

library(forecast)

teams.lm.exhaus.pred <- predict(teams.lm.exhaus, valid.df) #predicted values on validation set
accuracy(teams.lm.exhaus.pred, valid.df$W) #prediction accuracy evaluated on validation set
#RMSE = 3.0095, MAPE = 3.1190

#Backward elimination
teams.lm.back <- step(teams.lm, direction = "backward")
summary(teams.lm.back) #Adjusted R-Squared of .9384
teams.lm.back.pred <- predict(teams.lm.back, valid.df)
accuracy(teams.lm.back.pred, valid.df$W)
#RMSE = 2.9940, MAPE = 3.1188

#Forward selection
#Create model with no predictors first
teams.lm.null <- lm(W~1, data = train.df)
teams.lm.forward <- step(teams.lm.null, scope=list(lower=teams.lm.null, upper=teams.lm), direction = "forward")
summary(teams.lm.forward) #Adjusted R-Squared of .9328
teams.lm.forward.pred <- predict(teams.lm.forward, valid.df)
accuracy(teams.lm.forward.pred, valid.df$W)
#RMSE = 3.2571, MAPE = 3.4108

#Stepwise selection
teams.lm.step <- step(teams.lm, direction = "both")
summary(teams.lm.step) #Adjusted R-Squared = .9415
teams.lm.step.pred <- predict(teams.lm.step, valid.df)
accuracy(teams.lm.step.pred, valid.df$W)
#RMSE = 2.9940, MAPE = 3.1188 (same as backwards)

#Backwards/Stepwise have lowest error - go with that model

#Show some of the results
predict(teams.lm.back, valid.df[1:10,])
Teams[c(2448,2452:2454,2470,2472,2474,2475,2477,2480),c(1,41,9)]

predict(teams.lm.back, valid.df[11:20,])
Teams[c(2487,2499,2501,2509,2513,2520:2522,2529,2531),c(1,41,9)]

#Detecting lack of fit
plot(teams.lm.step$fitted.values, teams.lm.step$residuals, xlab="Fitted Values", ylab="Residuals", main="Residual Plot") #looks randomly distributed
#detecting the normality assumption
hist(teams.lm.step$residuals, breaks= 20, main="Histogram of Residuals", xlab="Residual Value") #looks normal
qqnorm(teams.lm.step$residuals)
qqline(teams.lm.step$residuals) #extremes taper off a little bit
library(lmtest)
#detecting residual correlation
dwtest(teams.lm.step) #Might have to investigate further on this (p=.06988) but value is 1.8455

#plot of predicted vs. actual wins
plot(x = predict(teams.lm.back,valid.df), y = valid.df$W, xlab="Predicted Wins", ylab="Actual Wins", main="Fitted vs. Actual Wins")

#Plot Actual wins vs runs scored and runs against
plot(x = Teams.df.lm$R, y = Teams.df.lm$W, xlab="Runs Scored", ylab="Wins", main="Wins versus Runs Scored")
plot(x = Teams.df.lm$RA, y = Teams.df.lm$W, xlab="Runs Against", ylab="Wins", main="Wins versus Runs Against")

#Shapiro-Wilks Test
shapiro.test(teams.lm.back$residuals) # W = 0.99517

#VIF for multicollinearity
library(regclass)
VIF(teams.lm.back)
#Multidollinearity within AB, H, RA, E, FP
#Multicollinearity means that some of many of the predictor variables in our model are corellated

#ACMS 30600 Notebook page 6-4 for normality tests

dim(Teams.df.lm)
