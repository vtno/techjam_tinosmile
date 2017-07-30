library(plyr)
library(data.table)
library(ggplot2)
library(xgboost)
library(caret)
library(dplyr)
library(magrittr)
library(mice)

training = fread("tj_02_training.csv",header = FALSE,colClasses = c("character","numeric"),showProgress = TRUE)
names(training) <- c("account_no","label")
training[,flag := "train"]

test = fread("tj_02_test.csv",showProgress = TRUE, colClasses = "character")
names(test) <- c("account_no")
test[,flag := "test"]

label = rbind(training,test,fill=TRUE)


account_txn = fread("tj_02_account_transaction.csv", colClasses=c("character", "character", "numeric" ,"character", "numeric", "character"), showProgress = TRUE)
acc_x_card = fread("tj_02_acc_x_card.csv", colClasses='character')

cc_txn = fread("tj_02_creditcard_transaction.csv", colClasses=c("character", "character", "numeric", "numeric", "character", "character"))
wholesale_mcc <- c('2741', '2791', '2842', '5013', '5021', '5039', '5044', '5045', '5046', '5047', '5051', '5065', '5072', '5074', '5085', '5094', '5099','5111', '5122', '5131', '5137', '5139', '5169', '5172', '5192', '5193', '5198', '5199', '7375', '7379', '7829', '8734')

wholesale_txn <- cc_txn[mer_cat_code %in% wholesale_mcc]
acc_features <- account_txn[,
              list(
                uniq_from_to_acc = length(unique(from_to_account_no)),
                uniq_from_to_acc_p = length(unique(from_to_account_no))/length(from_to_account_no),
                DR_count = as.numeric(nrow(.SD[txn_type=="DR"])),
                CR_count = as.numeric(nrow(.SD[txn_type=="CR"]))
              ),
              by=list(account_no)
            ]

cc_wholesale <- cc_txn[,
                 list(
                   is_wholesale = mer_cat_code %in% wholesale_mcc
                 ),
                 by=list(card_no)
               ]

cc_features = cc_wholesale[,
             list(
              count_whole_sale_txn = nrow(.SD[is_wholesale == TRUE])
              ),
             by=list(card_no)
            ]

cc_acc_txn = acc_features %>%
  left_join(acc_x_card, by='account_no') %>%
  left_join(cc_features, by='card_no') %>% setDT

cc_acc_txn[,card_no := NULL]
cc_acc_txn[is.na(count_whole_sale_txn)] <- 0
features <- cc_acc_txn

features_selection_vector <- c('uniq_from_to_acc', 'uniq_from_to_acc_p', 'DR_count', 'CR_count', 'count_whole_sale_txn')

full_data = label %>% left_join(features, by='account_no') %>% setDT

full_data[is.na(full_data)] <- 0
summary(full_data)

training_data = full_data[flag == "train"]
training_data[,flag := NULL]

test_data = full_data[flag == "test"]
test_data[,flag := NULL]

train_feature = training_data[, features_selection_vector, with=FALSE]
train_label = training_data[,list(label)]

dtrain <- xgb.DMatrix(
  data = as.matrix(train_feature),
  label= as.matrix(train_label)
)
watchlist <- list(train=dtrain)
params = list(objective = "binary:logistic",max_depth= 4, eta=0.001, colsample_bytree = 1, min_child_weight = 1, subsample = 0.6, gamma = 1)

#Train model

bst <- xgb.train(data=dtrain,params = params, watchlist=watchlist,verbose = 1,nrounds = 17, print.every.n = 100,early.stop.round = 500,maximize = FALSE)

test_feature = test_data[, features_selection_vector, with = FALSE ]

dtest <- xgb.DMatrix(
  data = as.matrix(test_feature)
)

v = test_data
v$predict = round(predict(bst,dtest))
output = inner_join(test,v[,list(account_no,predict)],by="account_no") %>% setDT
write.csv(output[,list(predict)],"2.txt",row.names = FALSE,col.names = NA)

```
