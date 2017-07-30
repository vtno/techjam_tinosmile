library(data.table)
library(ggplot2)
library(xgboost)
library(caret)
library(dplyr)
library(magrittr)
library(mice)


training = fread("tj_04_training.csv",header = FALSE,colClasses = c("character","integer"),showProgress = TRUE)
names(training) <- c("account_no","label")
training[,flag := "train"]

test = fread("tj_04_test.csv",showProgress = TRUE,colClasses = "character")
names(test) <- c("account_no")
test[,flag := "test"]

label = rbind(training,test,fill=TRUE)

account = fread("account_info.csv",colClasses = c(account_no = "character"), showProgress = TRUE)
transaction = fread("account_transaction.csv",colClasses = c(account_no = "character",fm_to_account_no = "character"))

transaction[,txn_dt := as.Date(txn_dt)]
transaction[,txn_tm := NULL]


transaction = transaction[order(account_no,txn_dt)]
transaction[,gap := as.numeric(txn_dt-shift(txn_dt)),by=list(account_no)]
transaction[,avg_gap := mean(gap,na.rm = TRUE),by=list(account_no)]
transaction[,n := .N,by=list(account_no)]
transaction[,freq := .N/as.numeric(max(txn_dt)-min(txn_dt)),by=list(account_no)]
transaction[,dr_to_cr := nrow(.SD[txn_type=="DR"])/nrow(.SD[txn_type=="CR"]),by=list(account_no)]
transaction[,dr_p := length(unique(.SD[txn_type == "DR"]$txn_amt))/nrow(.SD[txn_type=="DR"]),by=list(account_no)]
transaction[,cr_p := length(unique(.SD[txn_type == "CR"]$txn_amt))/nrow(.SD[txn_type=="CR"]),by=list(account_no)]
transaction[,oth_acc := mean(fm_to_account_no == "0"),by=list(account_no)]
transaction[,oth_acc_dr := mean(.SD[txn_type == "DR"]$fm_to_account_no == "0"),by=list(account_no)]
transaction[,oth_acc_cr := mean(.SD[txn_type == "CR"]$fm_to_account_no == "0"),by=list(account_no)]
transaction[,acc_id_p := length(unique(.SD[!fm_to_account_no == 0]$fm_to_account_no))/nrow(.SD[! fm_to_account_no == "0" ]),by=list(account_no)]

full_data = label %>% left_join(account,by="account_no") %>% left_join(transaction,by="account_no") %>% setDT
full_data[,opn_dt := as.Date(opn_dt)]
full_data[,pos_dt := as.Date(pos_dt)]

full_data[is.na(full_data)] <- -1
full_data[!is.finite(freq),freq := -1]

full_data = full_data[,list(account_no,label,freq,n,dr_to_cr,avg_gap,cr_p,dr_p,oth_acc,oth_acc_dr,oth_acc_cr,acc_id_p,flag)] %>% unique

training_data = full_data[flag == "train"]
training_data[,flag := NULL]

test_data = full_data[flag == "test"]
test_data[,flag := NULL]


train_feature = training_data[,list(freq,n,dr_to_cr,avg_gap,cr_p,dr_p,oth_acc,oth_acc_dr,oth_acc_cr,acc_id_p)]
train_label = training_data[,list(label = label == "sa")]
#train_label = train_label[!is.na(npl)]
levels(train_label$label) <- make.names(levels(factor(train_label$label)))

dtrain <- xgb.DMatrix(
  data = as.matrix(train_feature),
  label= as.matrix(train_label)
)
watchlist <- list(train=dtrain)

# Hyper Parameter
params = list(objective = "binary:logistic",max_depth= 4, eta=0.005, colsample_bytree = 1, min_child_weight = 1, subsample = 0.6,gamma = 1)


bst <- xgb.train(data=dtrain,params = params, watchlist=watchlist,verbose = 1,nrounds = 1562,print.every.n = 100,early.stop.round = 500,maximize = FALSE)

test_feature = test_data[,list(freq,n,dr_to_cr,avg_gap,cr_p,dr_p,oth_acc,oth_acc_dr,oth_acc_cr)]

dtest <- xgb.DMatrix(
  data = as.matrix(test_feature)
)

v = test_data
v$predict = round(predict(bst,dtest))
output = inner_join(test,v[,list(account_no,predict)],by="account_no") %>% setDT
output[,predict := ifelse(predict == 1,"sa","ca")]
#write.csv(output[,list(predict)],"4.txt",row.names = FALSE,col.names = NA,quote = FALSE)
write.table(output[,list(predict)],"4.txt",sep = ",",col.names = FALSE,row.names = FALSE, quote = FALSE)
