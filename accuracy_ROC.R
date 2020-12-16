accuracy_ROC <- function(model,
                         data, 
                         target_variable = "opinia",
                         predicted_class = "1") {
  
  # generate forecasted probabilities
  # for the predicted class (default = "Yes")
  
  forecasts_p <- predict(model, data,
                         type = "prob")[, predicted_class]
  
  # and also the predicted class
  forecasts_c <- predict(model, data)
  
  # real values - the pull () function will replace
  # tibble object in vector
  
  real <- (data[, target_variable]) %>% pull
  
  # we calculate the area under the ROC chart
  AUC <- roc(predictor = forecasts_p,
             # forecasted probabilities of success
             response = real) # real values
  
  # save classification table as the table object
  
  table <- confusionMatrix(forecasts_c, # forecasted class
                           real, # real
                           predicted_class) 
  # at the end  return the result
  result <- c(table$overall[1], # Accuracy 
              table$byClass[c(1:2, 11)], # sens, spec, balanced
              ROC = AUC$auc)
  
  return(result)
  
}
