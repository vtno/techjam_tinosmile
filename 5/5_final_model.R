
library(data.table)
library(ggplot2)
library(xgboost)
library(caret)
library(dplyr)
library(magrittr)
library(mice)

training = fread("tj_05_training.csv",header = FALSE,colClasses = c("character","integer"),showProgress = TRUE) %>% unique
names(training) <- c("card_no","gender")
training[,flag := "train"]
training = training[!duplicated(training$card_no),]

test = fread("tj_05_test.csv",showProgress = TRUE,colClasses = "character")
names(test) <- c("card_no")
test[,flag := "test"]

label = rbind(training,test,fill=TRUE)

transaction = fread("tj_05_credit_card_transaction.csv",showProgress = TRUE,colClasses = c(txn_amount = "numeric",card_no = "character"))
transaction[,txn_date := as.Date(txn_date)]
transaction[,txn_hour := as.numeric(txn_hour)]


ss = label %>% left_join(transaction,by="card_no") %>% setDT
temp = ss[,list(p = mean(gender,na.rm = TRUE),n=.N),by=list(mer_cat_code)]
temp = temp[n > 100 & (p < 0.45 | p > 0.55)]

transaction = transaction[mer_cat_code %in% temp$mer_cat_code]
transaction = label %>% left_join(transaction,by="card_no") %>% setDT


full_data = transaction
full_data[,count := .N,by=list(card_no,mer_cat_code)]
full_data = full_data[,list(card_no,gender,flag,mer_cat_code,count)] %>% unique
##full_data = dcast(full_data,card_no  + gender + flag ~ mer_cat_code) %>% setDT
##full_data[is.na(full_data)] <- 0


training_data = full_data[flag == "train"]
training_data[,flag := NULL]

test_data = full_data[flag == "test"]
test_data[,flag := NULL]

full_knn = prop.table(table(full_data$card_no,full_data$mer_cat_code),margin = 1) %>% data.table
full_knn[is.nan(N) , N := 0]
full_knn = dcast(full_knn,V1~V2) %>% setDT
setnames(full_knn,old="V1",new ="card_no")



train_knn = full_knn[card_no %in% training$card_no]
train_knn = left_join(train_knn,training,by="card_no")
train_label = train_knn$gender
train_knn$gender = NULL
train_knn$flag = NULL

k = 10

test_knn = full_knn[card_no %in% test$card_no]
knn_pred = knn(train = train_knn,test = test_knn,cl = train_label,k = k)
test_knn$pred = as.numeric(knn_pred)-1
output = test %>% left_join(test_knn[,list(card_no,pred)],by="card_no")
write.csv(output$pred, paste0("5 - k",k,".txt"),row.names = FALSE)

print(summary(output$pred))


