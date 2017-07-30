
library(data.table)
library(ggplot2)
library(xgboost)
library(caret)
library(dplyr)
library(magrittr)
library(mice)


training = fread("tj_03_training.csv",header = FALSE,colClasses = c("character","integer"),showProgress = TRUE)
names(training) <- c("account_no","label")
training[,flag := "train"]

test = fread("tj_03_test.csv",showProgress = TRUE,colClasses = "character")
names(test) <- c("account_no")
test[,flag := "test"]

label = rbind(training,test,fill=TRUE)

deposit = fread("tj_03_deposit_txn.csv",showProgress = TRUE,colClasses = c(from_to_account_no = "character",account_no = "character",txn_dt = "date",txn_type = "character"))[,txn_dt := as.Date(txn_dt)]

account = fread("tj_03_account_info.csv",showProgress =TRUE,colClasses = c(account_no = "character",txn_dt ="date",open_date = "date",customer_type = "character",last_active_date = "date",dormant_days = "integer"))
account[,txn_dt := as.Date(txn_dt)][,open_date := as.Date(open_date)][,last_active_date := as.Date(last_active_date)]

account[, c("compound_frq","compound_frq_unit","eff_interest_rate") := NULL]
account = account[order(account_no,customer_type,-txn_dt)] #order by date
account[, txn_dormant := (shift(txn_dt) - txn_dt), by = list(account_no,customer_type)][,txn_dormant := as.numeric(txn_dormant)] 
account[, txn_date := as.Date(txn_dt)][,txn_date := as.numeric(txn_date - min(txn_date)+1)] 
account[, act_date := as.numeric(as.Date(last_active_date))][,act_date := (act_date - min(act_date)+1)]
account[, opn_date := as.numeric(as.Date(open_date))][,opn_date := (opn_date - min(opn_date)+1)]
account[, freq := .N/as.numeric(max(txn_dt)-min(txn_dt)) ,by=list(account_no,customer_type)]
account[,avg_dormant := mean(dormant_days,na.rm = TRUE),by=list(account_no,customer_type)] #find average dormant 
account[,avg_txn_dormant := mean(txn_dormant,na.rm = TRUE),by=list(account_no,customer_type)] # same here

sum_account = account[,list(account_no,customer_type,avg_dormant,avg_txn_dormant,freq)] %>% unique

deposit[,txn_total_amount := sum(txn_amount),by=list(account_no)]
deposit[,n := .N,by=list(account_no)]
deposit[,dr_n := nrow(.SD[txn_type == "DR"])/n,by=list(account_no)]
deposit[,txn_avg_amount := txn_total_amount/n]
deposit = deposit[order(account_no,txn_dt)]
deposit[,gap := ( 24*as.numeric(txn_dt-shift(txn_dt)) + txn_hour - shift(txn_hour)),by=list(account_no)]
deposit[,avg_gap := mean(gap,na.rm = TRUE),by=list(account_no)]
deposit[,freq := .N/as.numeric(max(txn_dt)-min(txn_dt)),by=list(account_no)]
deposit[,dr_to_cr := nrow(.SD[txn_type=="DR"])/nrow(.SD[txn_type=="CR"]),by=list(account_no)]
deposit[,dr_p := length(unique(.SD[txn_type == "DR"]$txn_amount))/nrow(.SD[txn_type=="DR"]),by=list(account_no)]
deposit[,cr_p := length(unique(.SD[txn_type == "CR"]$txn_amount))/nrow(.SD[txn_type=="CR"]),by=list(account_no)]
deposit[,oth_acc := mean(from_to_account_no == "0"),by=list(account_no)]
deposit[,oth_acc_dr := mean(.SD[txn_type == "DR"]$from_to_account_no == "0"),by=list(account_no)]
deposit[,oth_acc_cr := mean(.SD[txn_type == "CR"]$from_to_account_no == "0"),by=list(account_no)]
deposit[,acc_id_p := length(unique(.SD[!from_to_account_no == 0]$from_to_account_no))/nrow(.SD[! from_to_account_no == "0" ]),by=list(account_no)]

deposit = deposit[,list(account_no,n,dr_n,txn_avg_amount,avg_gap,freq,dr_to_cr,dr_p,cr_p,oth_acc,oth_acc_cr,oth_acc_dr,acc_id_p)] %>% unique

full_data = label %>% 
  left_join(sum_account,by="account_no") %>% 
  left_join(deposit,by="account_no",suffix =c(".account",".deposit")) %>% setDT

full_data = full_data[!duplicated(full_data$account_no),]
full_data[is.na(full_data)]<- -1
full_data[,freq.account := ifelse(is.finite(freq.account),freq.account,-1)]
full_data[,freq.deposit := ifelse(is.finite(freq.deposit),freq.deposit,-1)]
full_data[,dr_to_cr := ifelse(is.finite(dr_to_cr),dr_to_cr,-1)]


training_data = full_data[flag == "train"]
training_data[,flag := NULL]

test_data = full_data[flag == "test"]
test_data[,flag := NULL]

train_feature = training_data[,list(customer_type = as.numeric(customer_type),avg_dormant,avg_txn_dormant,freq.account,n ,dr_n,txn_avg_amount,avg_gap,freq.deposit,dr_to_cr,dr_p,cr_p,oth_acc,oth_acc_cr,oth_acc_dr,acc_id_p)]

train_label = training_data[,list(label)]
#train_label = train_label[!is.na(npl)]
#levels(train_label$label) <- make.names(levels(factor(train_label$label)))

dtrain <- xgb.DMatrix(
  data = as.matrix(train_feature),
  label= as.matrix(train_label)
)
watchlist <- list(train=dtrain)

# Hyper Parameter
params = list(objective = "binary:logistic",max_depth= 4, eta=0.004, colsample_bytree = 1, min_child_weight = 1, subsample = 0.6,gamma = 1)

#Train model

bst <- xgb.train(data=dtrain,params = params, watchlist=watchlist,verbose = 1,nrounds = 6768,print_every_n = 100, early_stopping_rounds = 500,maximize = FALSE)

test_feature = test_data[,list(customer_type = as.numeric(customer_type),avg_dormant,avg_txn_dormant,freq.account,n ,dr_n,txn_avg_amount,avg_gap,freq.deposit,dr_to_cr,dr_p,cr_p,oth_acc,oth_acc_cr,oth_acc_dr,acc_id_p)]

dtest <- xgb.DMatrix(
  data = as.matrix(test_feature)
)

v = test_data
v = v[!duplicated(v$account_no),]
v$predict = round(predict(bst,dtest))
output = inner_join(test,v[,list(account_no,predict)],by="account_no") %>% setDT
write.csv(output[,list(predict)],"3.txt",row.names = FALSE,col.names = NA)
