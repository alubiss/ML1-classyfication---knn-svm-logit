

# Dane potrzebne do przeprowadzenia badania pochodzą m.in. z serwisu goodreads.com oraz steamboatbooks.com. Zgromadzona baza zawiera 51 000 recenzji dotyczących 363 różnych książek. Wśród nich jest 20 gatunków książek (thriller, science fiction itp.). Na podstawie opisów i tytułów książek wyszczególniono 9 różnych, najczęściej występujących tematyk (miłość, wojna, śmierć itp.). Dane dotyczące liczby sprzedanych egzemplarzy książki według serwisu amazon.com pochodzą z bazy danych serwisu kaggle.com. 
# Zmiennymi objaśniajacymi są również: płeć autora książki, płeć recenzenta, płeć głównego bahatera książki, gatunek, cena, liczba stron, średnia liczba przyznanych gwiazdek, rok wydania książki, liczba zgromadzonych przez książkę nagród, format wydania książki.
# Zmienną objaśnianą jest  "opinia recenzenta"  (pozytywna/ negatywna opinia).

# Wczytywanie danych
library(readxl)
model2 <- read_excel("~/Desktop/model2.xlsx")
#View(model2)

library(dplyr)
library(readr)
library(AER)
library(lmtest) # lrtest(), waldtest()
library(nnet)
#library(caret)
library(verification)
library(tidyr)
library(foreign)
library(janitor) # tabyl()
library(class)
require(caret)
library(pROC)
library(DMwR) 
library(ROSE)
dane= model2
summary(dane)
glimpse(dane)

# Przygotowanie danych
# Usunięcie brakow danych, zadeklarowanie zmiennych jakościowych:

dane = na.omit(dane)

dane$formaty <- as.factor(dane$formaty)
dane$opinia <- as.factor(dane$opinia)
dane$plec_autora <- as.factor(dane$plec_autora)
dane$plec_recenzenta <- as.factor(dane$plec_recenzenta)
dane$rodzaj <- as.factor(dane$rodzaj)
dane$tematyka <- as.factor(dane$tematyka)

levels(dane$opinia)
levels(dane$plec_autora)
levels(dane$plec_recenzenta)
levels(dane$rodzaj)
levels(dane$tematyka)

# Ogólna postać modelu

modelformula <- opinia ~  
  cena+ formaty + lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + rok + stars + tematyka + sprzedaz

# Podział danych na treningowe i testowe 

set.seed(123456789)

which_train <- createDataPartition(dane$opinia, 
                                   p = 0.7, 
                                   list = FALSE) 
dane.train <- dane[which_train,]
dane.test <- dane[-which_train,]

# Model I - logit

ctrl_nocv <- trainControl(method = "none")
set.seed(123456789)

dane.logit.train <- 
  train(opinia ~  
          cena+ formaty + lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + rok + stars +                    tematyka + sprzedaz,
        data =dane.train, # training sample
        method = "glm",
        family = "binomial",
        trControl = ctrl_nocv)

summary(dane.logit.train)

# Wartość AIC modelu równa  13848.

# Znalezienie odpowiedniej postaci modelu

# Po przeprowadzniu metody od ogółu do szczegołu :

summary(dane.logit.train)

dane2.train=dane.train

# pominięcie poziomu tematyka 1
levels(dane.train$tematyka)[levels(dane.train$tematyka)==1]="0"
levels(dane.train$tematyka)
# pominięcie zmiennej cena

# pominięcie poziomu rodzaj 8
levels(dane.train$rodzaj)[levels(dane.train$rodzaj)==8]="1"
levels(dane.train$rodzaj)

# pominięcie zmiennej formaty

# pominięcie poziomu tematyka 3
levels(dane.train$tematyka)[levels(dane.train$tematyka)==3]="0"
levels(dane.train$tematyka)

# pominięcie zmiennej sprzedaz

# pominięcie poziomu rodzaj 2
levels(dane.train$rodzaj)[levels(dane.train$rodzaj)==2]="1"
levels(dane.train$rodzaj)

# pominięcie poziomu tematyka 4
levels(dane.train$tematyka)[levels(dane.train$tematyka)==4]="0"
levels(dane.train$tematyka)

# pominięcie zmiennej lstron

# pominięcie poziomu rodzaj 14
levels(dane.train$rodzaj)[levels(dane.train$rodzaj)==14]="1"
levels(dane.train$rodzaj)

# pominięcie poziomu rodzaj 9
levels(dane.train$rodzaj)[levels(dane.train$rodzaj)==9]="1"
levels(dane.train$rodzaj)

# pominięcie poziomu rodzaj 7
levels(dane.train$rodzaj)[levels(dane.train$rodzaj)==7]="1"
levels(dane.train$rodzaj)

# pominięcie poziomu rodzaj 5
levels(dane.train$rodzaj)[levels(dane.train$rodzaj)==5]="1"
levels(dane.train$rodzaj)

# pominięcie zmiennej lnagrody

# pominięcie poziomu rodzaj 16
levels(dane.train$rodzaj)[levels(dane.train$rodzaj)==16]="1"
levels(dane.train$rodzaj)

# pominięcie poziomu rodzaj 16
levels(dane.train$rodzaj)[levels(dane.train$rodzaj)==16]="1"
levels(dane.train$rodzaj)

# pominięcie poziomu tematyka 7
levels(dane.train$tematyka)[levels(dane.train$tematyka)==7]="0"
levels(dane.train$tematyka)

# pominięcie zmiennej stars

# pominięcie poziomu tematyka 6
levels(dane.train$tematyka)[levels(dane.train$tematyka)==6]="0"
levels(dane.train$tematyka)

set.seed(123456789)
dane.logit20.train <- 
  train(opinia ~  
          plec_autora +plec_recenzenta + rodzaj + tematyka,
        data =dane.train, # training sample
        method = "glm",
        family = "binomial",
        trControl = ctrl_nocv)
summary(dane.logit20.train)

# Wartość AIC modelu 20 wynosi 13831. 

# Model II - logit
# Porównainie oszacowań modeli:

summary(dane.logit.train)
summary(dane.logit20.train)

# Wartość kryterium informacyjnego spadła z 13848 do 13831, a więc na tej podstawie wybieramy model drugi, w którym wszystkie zmienne są istotne.

modelformula <- opinia ~  
  lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka

# Dostosowanie danych testowych do formuły modelu, którą będziemy się posługiwać:
dane2.test=dane.test

# pominięcie poziomu tematyka 1
levels(dane.test$tematyka)[levels(dane.test$tematyka)==1]="0"

# pominięcie poziomu rodzaj 8
levels(dane.test$rodzaj)[levels(dane.test$rodzaj)==8]="1"

# pominięcie poziomu tematyka 3
levels(dane.test$tematyka)[levels(dane.test$tematyka)==3]="0"

# pominięcie poziomu rodzaj 2
levels(dane.test$rodzaj)[levels(dane.test$rodzaj)==2]="1"

# pominięcie poziomu tematyka 4
levels(dane.test$tematyka)[levels(dane.test$tematyka)==4]="0"

# pominięcie poziomu rodzaj 14
levels(dane.test$rodzaj)[levels(dane.test$rodzaj)==14]="1"

# pominięcie poziomu rodzaj 9
levels(dane.test$rodzaj)[levels(dane.test$rodzaj)==9]="1"

# pominięcie poziomu rodzaj 7
levels(dane.test$rodzaj)[levels(dane.test$rodzaj)==7]="1"

# pominięcie poziomu rodzaj 5
levels(dane.test$rodzaj)[levels(dane.test$rodzaj)==5]="1"

# pominięcie poziomu rodzaj 16
levels(dane.test$rodzaj)[levels(dane.test$rodzaj)==16]="1"

# pominięcie poziomu rodzaj 16
levels(dane.test$rodzaj)[levels(dane.test$rodzaj)==16]="1"

# pominięcie poziomu tematyka 7
levels(dane.test$tematyka)[levels(dane.test$tematyka)==7]="0"

# pominięcie poziomu tematyka 6
levels(dane.test$tematyka)[levels(dane.test$tematyka)==6]="0"



# Model III - KNN

# Kolejnym etapem pracy jest zastosowanie metody K-najbliższych sąsiadów:

set.seed(123456789)

different_k <- data.frame(k = 1:55)
ctrl_cv5 <- trainControl(method = "cv",
                         number = 5)
set.seed(123456789)

dane.knn_cv.train <- 
  train(opinia ~  
          lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
        data = dane.train, 
        method = "knn",
        trControl = ctrl_cv5,
        tuneGrid = different_k,
        # data transformation,
        preProcess = c("range"))

plot(dane.knn_cv.train)
#dane.knn_cv.train

# Accuracy
# 0.9127290

dane.knn_cv.train$finalModel$k

# Na podstwie wyników wybieramy k=55.

set.seed(123456789)

dane.knn_cv.train <- 
  train(opinia ~  
          lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
        data = dane.train, 
        method = "knn",
        trControl = ctrl_cv5,
        tuneGrid = data.frame(k = 55),
        # data transformation,
        preProcess = c("range"))


# Model knn dla pierwszej postaci modelu (z uwzględnieniem nieistotnych zmiennych):

set.seed(123456789)

different_k <- data.frame(k = 1:55)
ctrl_cv5 <- trainControl(method = "cv",
                         number = 5)
set.seed(123456789)

dane2.knn_cv.train <- 
  train(opinia ~  
          cena+ formaty + lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + rok + stars +                    tematyka + sprzedaz,
        data = dane2.train, 
        method = "knn",
        trControl = ctrl_cv5,
        tuneGrid = different_k,
        # data transformation,
        preProcess = c("range"))

plot(dane2.knn_cv.train)
#dane2.knn_cv.train

dane2.knn_cv.train$finalModel$k

# Na podstwie wyników wybieramy k=55.

set.seed(123456789)

dane2.knn_cv.train <- 
  train(opinia ~  
          cena+ formaty + lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + rok + stars +                    tematyka + sprzedaz,
        data = dane2.train, 
        method = "knn",
        trControl = ctrl_cv5,
        tuneGrid = data.frame(k = 55),
        # data transformation,
        preProcess = c("range"))


# Model IV - SVM (support vector machines)

# Kolejnym zastosowanym modelem jest model SVM:

set.seed(123456789)

parametersC_sigma <- 
  expand.grid(C = c(1, 5, 10, 25, 50, 100),
              sigma = c(0.001, 0.01, 0.1, 1, 10, 100, 1000))

set.seed(123456789)

dane.svm_Radial <- train(opinia ~  
                           lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
                         data = dane.train,
                         method = "svmRadial",
                         tuneGrid = parametersC_sigma,
                         trControl = ctrl_cv5)

dane.svm_Radial

# sigma = 100 and C = 1

set.seed(123456789)

parametersC_sigma <- 
  expand.grid(C = 1,
              sigma = 100)

set.seed(123456789)

dane.svm <- train(opinia ~  
                    lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
                  data = dane.train,
                  method = "svmRadial",
                  tuneGrid = parametersC_sigma,
                  trControl = ctrl_cv5)

dane.svm

# Accuracy  Kappa
# 0.912729  0    


# Model SVM dla wszystkich zmiennych modelu:

set.seed(123456789)

parametersC_sigma <- 
  expand.grid(C = c(1, 5, 10, 25, 50, 100),
              sigma = c(0.001, 0.01, 0.1, 1, 10, 100, 1000))

set.seed(123456789)

dane2.svm_Radial <- train(opinia ~  
                            cena+ formaty + lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + rok + stars +                    tematyka + sprzedaz,
                          data = dane2.train,
                          method = "svmRadial",
                          tuneGrid = parametersC_sigma,
                          trControl = ctrl_cv5)

dane2.svm_Radial

# sigma = 1 and C = 1


set.seed(123456789)

parametersC_sigma <- 
  expand.grid(C = 1,
              sigma = 1)

set.seed(123456789)

dane2.svm <- train(opinia ~  
                     cena+ formaty + lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + rok + stars +                    tematyka + sprzedaz,
                   data = dane2.train,
                   method = "svmRadial",
                   tuneGrid = parametersC_sigma,
                   trControl = ctrl_cv5)

dane2.svm

# Accuracy  Kappa
# 0.912729  0

# Dane są w pełni separowalne, dlatego nie ma potrzeby zastosowywać modelu SVM z dodatkowymi zmiennymi.

logit1 <- confusionMatrix(predict(dane.logit.train, 
                                  newdata = dane2.test), 
                          dane2.test$opinia,
                          positive = "1")

logit2 <- confusionMatrix(predict(dane.logit2.train, 
                                  newdata = dane.test), 
                          dane.test$opinia,
                          positive = "1") 

knn <- confusionMatrix(predict(dane.knn_cv.train, 
                               newdata = dane.test), 
                       dane.test$opinia,
                       positive = "1")

knn2 <- confusionMatrix(predict(dane2.knn_cv.train, 
                                newdata = dane2.test), 
                        dane2.test$opinia,
                        positive = "1")

svm <- confusionMatrix(predict(dane.svm_Radial, 
                               newdata = dane.test), 
                       dane.test$opinia,
                       positive = "1")

svm2 <- confusionMatrix(predict(dane2.svm_Radial, 
                                newdata = dane2.test), 
                        dane2.test$opinia,
                        positive = "1")

# Porównanie modeli

names <- c("logit1", "logit2", "knn", "knn2", "svm", "svm2")
scores <- cbind(names,rbind(logit1$overall, logit2$overall, knn$overall, knn2$overall, svm$overall, svm2$overall))
scores<-as.data.frame(scores)
scores <- scores %>% arrange(desc(Accuracy)) 

scores

# Z wyników widzimy, że wszytskie przedstawione wyżej modele cechuje taki sam poziom precyzji przy klasyfikacji opinii recenzentów. Wskaźnik jest bardzo wysoki i wynosi 91%. Wynika to z tego, że baza danych zawiera w większości opinie pozytywne. Oznacza to, że dane nie są zrówoważone, klasy nie są w przybliżeniu równo liczne. Modele KNN i SVM są bardziej odporne na niezróważone dane, zatem sprawdzimy model logitowy korzystając z metod re-sampling or re-weighting, kótre pomogą rozwiązać problem przetrenowania i niezrównoważenia danych.

# Pierwsza postać modelu:

set.seed(123456789)

fiveStats <- function(...) c(twoClassSummary(...), 
                             defaultSummary(...))

ctrl_cv <- trainControl(method = "cv",
                        summaryFunction = fiveStats)
set.seed(123456789)

dane.logit17.train <- 
  train(opinia ~  
          lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
        data =dane.train, # training sample
        method = "glm",
        family = "binomial",
        trControl = ctrl_nocv)
summary(dane.logit17.train)

dane.logit.train_ost= dane.logit17.train 

source("accuracy_ROC.R")

dane.logit17.train %>%
  accuracy_ROC(data = dane.test)

# Model A:
  
# weighing observations
  
# Nakładamy wagi:

set.seed(123456789)

(freqs <- table(dane.train$opinia))

myWeights <- ifelse(dane.train$opinia == "1",
                    0.5/freqs[1], 
                    0.5/freqs[2])

set.seed(123456789)

dane.logit.train_weighted <- 
  train(opinia ~  
          lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
        data = dane.train, 
        method = "glm",
        family = "binomial",
        trControl = ctrl_nocv,
        weights = myWeights)
dane.logit.train_weighted

dane.logit.train_weighted %>%
  accuracy_ROC(data = dane.test)

# Możemy zaobserwować niewielki wzrost ROC

# Model B:
  
# down-sampling

set.seed(123456789)

ctrl_cv$sampling <- "down"

set.seed(123456789)

dane.logit.train_down <- 
  train(opinia ~  
          lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
        data = dane.train,
        method = "glm",
        family = "binomial",
        trControl = ctrl_nocv)
dane.logit.train_down

dane.logit.train_down %>%
  accuracy_ROC(data = dane.test)

# Model C:
  
# up-sampling

set.seed(123456789)

ctrl_cv$sampling <- "up"

set.seed(123456789)

dane.logit.train_up <- 
  train(opinia ~  
          lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
        data = dane.train, 
        method = "glm",
        family = "binomial",
        trControl = ctrl_nocv)
dane.logit.train_up

dane.logit.train_up %>%
  accuracy_ROC(data = dane.test)


# Model D:
  
# SMOTE model

set.seed(123456789)

ctrl_cv$sampling <- "smote"

set.seed(123456789)

# pomniejszenie próbki treningowej, przetwarzająca tylko trudne obszary
dane.logit.train_smote <- 
  train(opinia ~  
          lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + tematyka,
        data = dane.train %>% sample_n(10000), 
        method = "glm",
        family = "binomial",
        trControl = ctrl_nocv)
dane.logit.train_smote

dane.logit.train_smote %>%
  accuracy_ROC(data = dane.test)


# Porównanie oszacowań modeli:

model_train_down = dane.logit.train_down
model_logit = dane.logit.train_ost
model_train_smote = dane.logit.train_smote
model_train_up = dane.logit.train_up
model_train_weighted = dane.logit.train_weighted

models_all <- ls(pattern = "model_")

sapply(models_all,
       function(x) accuracy_ROC(get(x), 
                                data = dane.test)) %>% 
  t()

# Wszystkie modele mają taką samą precyzje predykcji. Najwyższą wartość ROC ma model model_train_weighted, dzięki temu że zastosowano wagi dla zrównoważenia liczby pozytywnych i negatywnych opinii recenzentów.
