#This is the Capstone Project code of Aditya Prasad Dash for Predicting Weeekly Sales from Store Features in the Choose Your Own Project in the course Data Science: Capstone Project (HarvardX PH125.9x).

if(!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)
if(!require(tidyr)) install.packages("tidyr")
library(tidyr)
if(!require(caret)) install.packages("caret")
library(caret)
if(!require(stringr)) install.packages("stringr")
library(stringr)
if(!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)
if(!require(lubridate)) install.packages("lubridate")
library(lubridate)
if(!require(corrplot)) install.packages("corrplot")
library(corrplot)

# Reading the dataset
#https://www.kaggle.com/datasets/yasserh/walmart-dataset/data by M YASSER H
options(timeout = 120)


walmart_dataset <- read.csv("/Users/aditya/Documents/Coursework/Online_Courses/Machine_Learning_and_Deep_Learning/Data_Science_Edx_Harvard_R/Final_Project/Capstone_Project/Walmart_Dataset/Walmart.csv")
str(walmart_dataset)
head(walmart_dataset,5)
walmart_dataset<-na.omit(walmart_dataset)
range(walmart_dataset$Weekly_Sales)
unique(walmart_dataset$Store)
walmart_dataset%>% ggplot(aes(as.character(Store), Weekly_Sales)) + geom_boxplot()

mean_sales_store<-walmart_dataset%>%group_by(Store)%>%summarise(mean_sales=mean(Weekly_Sales))
mean_sales_store %>% ggplot(aes(as.character(Store), mean_sales)) + geom_point()

walmart_dataset<-walmart_dataset%>% pivot_wider(names_from = Store, names_prefix="Store",values_from = Store, values_fn = length,values_fill = 0)
head(walmart_dataset)
length(unique(walmart_dataset$Day))
walmart_dataset<-walmart_dataset%>%mutate(Day=day(dmy(Date)),Month=month(dmy(Date)), Year=year(dmy(Date)))
head(walmart_dataset%>%select(c("Day","Month","Year")),5)

walmart_dataset%>% ggplot(aes(as.character(Day), Weekly_Sales)) + geom_boxplot()
walmart_dataset%>% ggplot(aes(as.character(Month), Weekly_Sales)) + geom_boxplot()
walmart_dataset%>% ggplot(aes(as.character(Year), Weekly_Sales)) + geom_boxplot()

plot(walmart_dataset%>%group_by(Day)%>%summarise(mean_sales=mean(Weekly_Sales)))
plot(walmart_dataset%>%group_by(Month)%>%summarise(mean_sales=mean(Weekly_Sales)))
plot(walmart_dataset%>%group_by(Year)%>%summarise(mean_sales=mean(Weekly_Sales)))

walmart_dataset%>%group_by(Day)%>%summarise(n=n(),mean_sales=mean(Weekly_Sales))%>%arrange(desc(mean_sales))
walmart_dataset%>%group_by(Month)%>%summarise(n=n(),mean_sales=mean(Weekly_Sales))%>%arrange(desc(mean_sales))
walmart_dataset%>%group_by(Year)%>%summarise(n=n(),mean_sales=mean(Weekly_Sales))%>%arrange(desc(mean_sales))

walmart_dataset<-walmart_dataset%>% pivot_wider(names_from = Month, names_prefix="Month",values_from = Month, values_fn = length,values_fill = 0)
walmart_dataset<-walmart_dataset%>% pivot_wider(names_from = Year, names_prefix="Year",values_from = Year, values_fn = length,values_fill = 0)
colnames(walmart_dataset)

walmart_dataset<-walmart_dataset%>%select(-c("Date")) 
walmart_dataset<-walmart_dataset%>%select(-c("Day")) 

walmart_dataset%>% ggplot(aes(Holiday_Flag, Weekly_Sales, group=Holiday_Flag)) + geom_boxplot()
walmart_dataset%>% ggplot(aes(Temperature, Weekly_Sales)) + geom_point()
walmart_dataset%>% ggplot(aes(Fuel_Price, Weekly_Sales)) + geom_point()
walmart_dataset%>% ggplot(aes(CPI, Weekly_Sales)) + geom_point()
walmart_dataset%>% ggplot(aes(Unemployment, Weekly_Sales)) + geom_point()

#Scaling the dataset
fMinMaxSclaer <- function(x) ( #https://www.geeksforgeeks.org/how-to-normalize-and-standardize-data-in-r/
  (x - min(x)) / (max(x) - min(x))
)


walmart_dataset[-1]<-as.data.frame(lapply(walmart_dataset[-1],fMinMaxSclaer))
head(walmart_dataset)
str(walmart_dataset)

#https://stackoverflow.com/questions/62679940/geom-bar-of-named-number-vector
cor_data<-cor(walmart_dataset$Weekly_Sales,walmart_dataset%>%select(where(is.numeric)))
barplot(cor_data)


#Splitting the dataset into training and testing sets
test_index <- createDataPartition(y = walmart_dataset$Weekly_Sales, times = 1,
                                 p = 0.15, list = FALSE)
testing_set <- walmart_dataset[test_index,]
training_set <- walmart_dataset[-test_index,]


nrow(training_set)
nrow(testing_set)

head(training_set,5)

fit_lm<-train(Weekly_Sales~.,method="lm",data=training_set)
summary(fit_lm)
y_hat<-predict(fit_lm,testing_set)

MAE<-function(true_values,predicted_values){
  mean(abs(predicted_values - true_values))
}
RMSE <- function(true_values, predicted_values){
  sqrt(mean((true_values - predicted_values)^2))}

mae_lm<-MAE(testing_set$Weekly_Sales,y_hat)

rmse_lm<-RMSE(testing_set$Weekly_Sales,y_hat)  
mean_weeklysales_val<-mean(testing_set$Weekly_Sales)
relative_mae_lm<-mae_lm/mean_weeklysales_val
relative_rmse_lm<-rmse_lm/mean_weeklysales_val

print(mae_lm)
print(relative_mae_lm)
print(rmse_lm)
print(relative_rmse_lm)

fit_knnreg<-train(Weekly_Sales~.,method="knn",data=training_set,metric="RMSE",tuneGrid = data.frame(k = seq(1, 30, 2)))
summary(fit_knnreg)
plot(fit_knnreg)
print(fit_knnreg)
fit_knnreg$bestTune
y_hat<-predict(fit_knnreg,testing_set)
mae_knnreg<-MAE(testing_set$Weekly_Sales,y_hat)
relative_mae_knnreg<-mae_knnreg/mean_weeklysales_val
rmse_knnreg<-RMSE(testing_set$Weekly_Sales,y_hat)
relative_rmse_knnreg<-rmse_knnreg/mean_weeklysales_val

print(mae_knnreg)
print(relative_mae_knnreg)
print(rmse_knnreg)
print(relative_rmse_knnreg)

hist(training_set$Weekly_Sales)
range(training_set$Weekly_Sales)
min_weekly_sales<-min(training_set$Weekly_Sales)
range_weekly_sales<-max(training_set$Weekly_Sales)-min(training_set$Weekly_Sales)

##Logistic Regression

training_set<-training_set%>%mutate(Category_Weekly_Sales=floor((Weekly_Sales-min_weekly_sales)/range_weekly_sales*2.9))
testing_set<-testing_set%>%mutate(Category_Weekly_Sales=floor((Weekly_Sales-min_weekly_sales)/range_weekly_sales*2.9))

hist(as.numeric(training_set$Category_Weekly_Sales))
training_set%>%group_by(Category_Weekly_Sales)%>%summarise(n=n(), mean_weekly_sales=mean(Weekly_Sales))

training_set<-training_set%>%mutate(Category_Weekly_Sales=as.factor(Category_Weekly_Sales))
testing_set<-testing_set%>%mutate(Category_Weekly_Sales=as.factor(Category_Weekly_Sales))

fit_knncl<-train(Category_Weekly_Sales~.,method="knn",data=training_set[,-1],tuneGrid = data.frame(k = seq(15,15,2)))
summary(fit_knncl)
fit_knncl$bestTune
y_hat_knncl<-predict(fit_knncl,testing_set[,-1],type="raw")
confusionMatrix(y_hat_knncl, testing_set$Category_Weekly_Sales)

#References
#https://www.kaggle.com/datasets/yasserh/walmart-dataset/data
#Data Science: Capstone, HarvardX PH125.9x provided via the edX platform
#Introduction to Data Science by Rafael A. Irizarry
#https://stackoverflow.com/questions/74706268/how-to-create-dummy-variables-in-r-based-on-multiple-values-within-each-cell-in
# https://www.geeksforgeeks.org/how-to-normalize-and-standardize-data-in-r/
# https://stackoverflow.com/questions/62679940/geom-bar-of-named-number-vector
