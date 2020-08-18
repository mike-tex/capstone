# Abalone Dataset
# Predicting the age of abalone 
# from physical measurements

# Context
# The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. Further information, such as weather patterns and location (hence food availability) may be required to solve the problem.
# 
# Original Dataset
# The original dataset can be acessed at https://archive.ics.uci.edu/ml/datasets/abalone.


# Name		        Data        Type	  Meas.	Description
# ----		        ---------	  -----	  -----------
# Sex		          nominal			M, F, and I (infant)
# Length		      continuous	mm	    Longest shell measurement
# Diameter	      continuous	mm	    perpendicular to length
# Height		      continuous	mm	    with meat in shell
# Whole weight	  continuous	grams	  whole abalone
# Shucked weight	continuous	grams	  weight of meat
# Viscera weight	continuous	grams	  gut weight (after bleeding)
# Shell weight	  continuous	grams	  after being dried
# Rings		        integer			        +1.5 gives the age in years


# https://www.kaggle.com/rodolfomendes/abalone-dataset
# https://www.kaggle.com/rodolfomendes/abalone-dataset/download
# https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
# https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names


if(!require(tidyverse)){install.packages("tidyverse")}
if(!require(caret)){install.packages("caret")}
if(!require(PivotalR)){install.packages("PivotalR")}
if(!require(Metrics)){install.packages("Metrics")}
if(!require(gbm)){install.packages("gbm")}
if(!require(gam)){install.packages("gam")}
if(!require(splines)){install.packages("splines")}
if(!require(foreach)){install.packages("foreach")}
if(!require(gridExtra)){install.packages("gridExtra")}
if(!require(corrplot)){install.packages("corrplot")}
if(!require(doMC)){
  install.packages("doMC", 
                   repos="http://R-Forge.R-project.org")}

doMC::registerDoMC(cores=4)




##################################################
##
##   load abalone dataset from PivotalR library 
##
##################################################

if(exists("abalone")){rm(list = ls())}
data(abalone)

# remove id column since not needed
abalone <- abalone %>% select(-id)

##
## data prep for regression
##

#create numeric sex column 
abalone_r <- abalone %>%
  mutate(sex_num =
           case_when(
             sex == "I" ~ 0,
             sex == "F" ~ 1,
             sex == "M" ~ 2
           )) 

# drop sex 
abalone_r <- abalone_r %>% select(-sex)

# reorder so sex_num is at top where sex was
abalone_r <- cbind(abalone_r[9], abalone_r[1:8])

# # # factor rings
# abalone_r$rings <- factor(abalone_r$rings)

save(abalone_r, file = "abalone_r.Rdata")


##
## data prep for classification
##

# based on rings breakdown, histogram and summary,
# classify age groups using ring quartiles:
# junior, adult, senior

abalone_c <- abalone_r %>%
  mutate(rgrp = case_when(
    between(rings, 1, 7) ~ "juvenile",
    between(rings, 8, 11) ~ "adult",
    between(rings, 12, 29) ~ "senior"
  ))

# remove rings 
abalone_c <- abalone_c %>% select(-rings)



#######################################
##                                   ##
##         ANALYZE THE DATA          ##
##                                   ##
#######################################

str(abalone)

summary(abalone_r)

## coorelation (w/o sex column)
cor(abalone_r$rings, abalone_r[-9])

##
## plot geom boxplot, histogram and density
##

# function to create list of plots by geom
plot_data <- 
  function(x_data, x_col, x_geom, x_parm = "") {
  fx <- "qplot(x_data[,x_col], 
               geom = x_geom, 
               xlab = x_col)"
  fx_parm <- paste0(fx, x_parm)
  eval(parse(text = fx_parm))
}

# density plot for each attribute
x_geom <- "density"
plots <- lapply(colnames(abalone[-1]),
                plot_data, 
                x_data = abalone[-1], 
                x_geom = x_geom, 
                x_parm = "")
grid.arrange(grobs = plots, ncol = 3, top = x_geom)


# histograms each attribute
x_geom <- "histogram"
plots <- lapply(colnames(abalone[-1]),
                plot_data, 
                x_data = abalone[-1], 
                x_geom = x_geom, 
                x_parm = "")
suppressMessages(message(
  grid.arrange(grobs = plots, 
               ncol = 3, 
               top = x_geom)))



# boxplots
x_geom <- "boxplot"
plots <- lapply(colnames(abalone[-1]),
                plot_data, 
                x_data = abalone[-1], 
                x_geom = x_geom, 
                x_parm = " + coord_flip()")
grid.arrange(grobs = plots, ncol = 3, top = x_geom)

# pairs matrix scatterplot
pairs(abalone_r)

# correlation plot color heat map
correlations <- cor(abalone[-1])
corrplot(correlations, 
       method="circle",
       title = "\n\n\n\nAbalone correlation plot") 



##############################################
##
##  split_data() into test and train - 90% train
##
##############################################

set.seed(3)
idx <-
  createDataPartition(
    y = abalone$rings,
    times = 1,
    p = 0.1,
    list = FALSE
  )

train_set_r <- abalone_r[-idx,]
test_set_r <- abalone_r[idx,]



## Apply multiple models

# initialize variables
models <- c("lm",
            "glm",
            "knn",
            "svmLinear",
            "rpart",
            "gamLoess",
            "treebag",
            "gbm",
            "rf")
training_data <- train_set_r
control <- trainControl(
  method="repeatedcv", 
  number=10, 
  repeats=3)
metric <- "RMSE"
seed <- 3

if(!file.exists("regress_fits.Rdata")){
regress_fits <- lapply(models, function(model) {
  print(model)
  set.seed(seed)
  train(
    rings ~ .,
    data = training_data,
    method = model,
    metric = metric,
    trControl = control
  )
})
names(regress_fits) <- models
save(regress_fits, file = "regress_fits.Rdata")
} else {
  load(file = "regress_fits.Rdata")
}

if(!exists("regress_fits")){
  load(file = "regress_fits.Rdata")}

results <- map_df(regress_fits, function(model) {
  method <- model$method
  RMSE <- model$results$RMSE %>% min()
  return(list("method" = method, "RMSE" = RMSE))
})

results %>% 
  as_tibble() %>%   
  # as_data_frame() %>%   
  arrange(RMSE)

best_regress_model <- 
  results$method[which.min(results$RMSE)] 

pred_regress <-
  predict.train(
    object = regress_fits[[best_regress_model]],
    newdata = test_set_r,
    type = "raw")

pred_regress <- round(pred_regress)

## Accuracy
accuracy(pred_regress, test_set_r$rings)

# [1] 0.4929375 # using abalone for train and test
## Accuracy = 0.4929375
# 
# using train_set for models 
# and test_set for predict
# [1] 0.249522





## How to improve?  
# Classification grouping rings by age:
## juvenile, adult, & senior


###########################
##                       ##
##    RINGS GROUPING     ##
##                       ##
###########################

##
## split the data after adding rgrp
# using index created earlier

train_set_c <- abalone_c[-idx,]
test_set_c<- abalone_c[idx,]

if(!file.exists("class_fits.Rdata")) {
  models <- c(
    "knn",
    "lda",
    "qda",
    "naive_bayes",
    "svmLinear",
    "svmRadial",
    "gamLoess",
    "multinom",
    "rf"
  )
  
  control <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3)
  metric <- "accuracy"
  seed <- 2020
  
  class_fits <- lapply(models, function(model) {
    print(model)
    set.seed(seed)
    train(
      rgrp ~ .,
      data = train_set_r,
      method = model,
      metric = metric,
      trControl = control
    )
  })
  
  names(class_fits) <- models
  
  save(class_fits, file = "class_fits.Rdata")
} else  {
  load(file = "class_fits.Rdata")
}

results <- map_df(class_fits, function(model) {
  method <- model$method
  Accuracy <- model$results$Accuracy %>% min()
  return(list("method" = method, "Accuracy" = Accuracy))
})

results %>%
  as_tibble() %>%
  # as_data_frame() %>%
  arrange(desc(Accuracy))

best_class <- results$method[which.max(results$Accuracy)] 

pred_class <-
  predict.train(
    object = class_fits[[best_class]],
    newdata = test_set_c)

test_set_c$rgrp <- factor(test_set_c$rgrp)

confusionMatrix(pred_class, 
  test_set_c$rgrp)$overall[["Accuracy"]]












