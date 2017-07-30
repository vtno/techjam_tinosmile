library(data.table)
library(ggplot2)
library(xgboost)
library(caret)
library(dplyr)
library(magrittr)
library(mice)

training = fread("tj_01_training.csv",header = FALSE,colClasses = c("character","integer"),showProgress = TRUE)
names(training) <- c("card_no","npl")
training[,flag := "train"]

test = fread("tj_01_test.csv",showProgress = TRUE,colClasses = "character")
names(test) <- c("card_no")
test[,flag := "test"]

label = rbind(training,test,fill=TRUE)

credit_card = fread("tj_01_creditcard_card.csv"
                    ,colClasses = c(cst_id = "character",card_no = "character"),showProgress = TRUE)[
                      ,pos_dt := as.Date(pos_dt)][
                        ,open_dt := as.Date(open_dt)][
                          ,icr_lmt_amt := as.integer(cr_lmt_amt > prev_cr_lmt_amt)  
                          ]

customer = fread("tj_01_creditcard_customer.csv",colClasses = c(cst_id = "character")
                 ,showProgress =TRUE)[
                   ,pos_dt:= as.Date(pos_dt)][,age := as.integer(age)]

transaction = fread("tj_01_creditcard_transaction.csv"
                    ,colClasses = c(card_no = "character",txn_hour ="integer")
                    ,showProgress = TRUE)[,txn_date := as.Date(txn_date)]


full_data = label %>% 
  left_join(credit_card,by="card_no") %>% 
  left_join(customer,by="cst_id",suffix =c(".credit",".customer")) %>% 
  left_join(transaction,by="card_no") %>% setDT


full_data[is.na(age), age := round(mean(customer$age,na.rm = TRUE))]
full_data = full_data[order(card_no,txn_date,txn_hour)]
full_data[,hour_diff := (24*(txn_date-shift(txn_date))+(txn_hour-shift(txn_hour))),by= list(card_no)]
full_data[,hour_diff := as.numeric(hour_diff)]
full_data[is.na(hour_diff),hour_diff := 0]

full_data[,pos_dt.customer := NULL]
full_data[,pos_dt.credit := NULL]
full_data[,open_dt := as.numeric(Sys.Date() - open_dt)]
full_data[,exp_dt := as.numeric(Sys.Date() - exp_dt)]
full_data[,left_dt := as.numeric(exp_dt-open_dt)]
full_data[,mer_id := (mer_id == 0)]

training_data = full_data[flag == "train"]
training_data[,flag := NULL]

test_data = full_data[flag == "test"]
test_data[,flag := NULL]

agg_training_data = training_data[,list(recency = max(txn_date),
                                        frequency = .N,
                                        avg_gap = (mean(hour_diff)),
                                        avg_amt = mean(txn_amount),
                                        mer_id = mean(mer_id)
)
,by = list(cst_id,card_no,age,cr_line_amt,cr_lmt_amt,icr_lmt_amt,npl,bill_cyc,open_dt,exp_dt,left_dt)]
agg_training_data[,cst_n_card := .N,by=list(cst_id)]
agg_training_data[,recency := as.numeric(max(recency)-recency)]

train_feature = agg_training_data[,list(age,cr_line_amt,cr_lmt_amt,icr_lmt_amt,recency,frequency,avg_gap,avg_amt,cst_n_card,bill_cyc,mer_id)]
train_label = agg_training_data[,list(npl)]
#train_label = train_label[!is.na(npl)]
levels(train_label$npl) <- make.names(levels(factor(train_label$npl)))

dtrain <- xgb.DMatrix(
  data = as.matrix(train_feature),
  label= as.matrix(train_label)
)
watchlist <- list(train=dtrain)

# Hyper Parameter
params = list(objective = "binary:logistic",max_depth= 4, eta=0.005, colsample_bytree = .8, min_child_weight = 1, subsample = 0.6,gamma = 1)

#Train model

bst <- xgb.train(data=dtrain,params = params, watchlist=watchlist,verbose = 1,nrounds = 2273,print.every.n = 100,early.stop.round = 500,maximize = FALSE)

agg_test_data = test_data[,list(recency = max(txn_date),
                                frequency = .N,
                                avg_gap = (mean(hour_diff)),
                                avg_amt = mean(txn_amount),
                                mer_id = mean(mer_id)
)
,by = list(cst_id,card_no,age,cr_line_amt,cr_lmt_amt,icr_lmt_amt,npl,bill_cyc,open_dt,exp_dt,left_dt)]
agg_test_data[,cst_n_card := .N,by=list(cst_id)]
agg_test_data[,recency := as.numeric(max(recency)-recency)]

test_feature = agg_test_data[,list(age,cr_line_amt,cr_lmt_amt,icr_lmt_amt,recency,frequency,avg_gap,avg_amt,cst_n_card,bill_cyc,mer_id)]


dtest <- xgb.DMatrix(
  data = as.matrix(test_feature)
)

v = agg_test_data
v$predict = round(predict(bst,dtest))
output = inner_join(test,v[,list(card_no,predict)],by="card_no") %>% setDT

#write output
write.csv(output[,list(predict)],"1.txt",row.names = FALSE,col.names = NA)