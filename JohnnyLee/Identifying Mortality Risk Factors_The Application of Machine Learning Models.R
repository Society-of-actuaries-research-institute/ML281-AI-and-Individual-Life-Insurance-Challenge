#setting
options(scipen = 999)
set.seed(1234)

#packages
library(dplyr)
library(lubridate)
library(readr)
library(tidyr)
library(stringi)
library(stringr)
library(caret)
library(ranger)
library(mlr)
library(kernelshap)
library(shapviz)
library(patchwork)
library(datawizard)

#import data
file = "ILEC_2012_19 - 20240429.txt"
check = read.table(file, header = TRUE, sep = "\t", nrows = 5)
col_list = colnames(check)

df = list()
for (i in c(1:5)) {
  df[[i]] = read.table(file, header = FALSE, sep = "\t", skip = (i-1)*10000000+1, nrows = 10000000, col.names = col_list)
  df[[i]] = df[[i]] %>%
            filter(Insurance_Plan != "Other",
                   !SOA_Antp_Lvl_TP %in% c("U","Unknown"),
                   !SOA_Guar_Lvl_TP %in% c("U","Unknown"),
                   SOA_Post_Lvl_Ind != "ULT",
                   Preferred_Indicator != "U",
                   is.na(Number_of_Pfd_Classes) | Number_of_Pfd_Classes !="U",
                   is.na(Preferred_Class) | Preferred_Class != "U") %>%
            mutate(Issue_Age_Grp = case_when(between(Issue_Age, 0, 17) ~ "00-17",
                                             between(Issue_Age, 18, 29) ~ "18-29",
                                             between(Issue_Age, 30, 39) ~ "30-39",
                                             between(Issue_Age, 40, 49) ~ "40-49",
                                             between(Issue_Age, 50, 59) ~ "50-59",
                                             between(Issue_Age, 60, 69) ~ "60-69",
                                             .default = "70+"),
                   Duration_Grp = case_when(between(Duration, 1, 5) ~ "01-05",
                                            between(Duration, 6, 10) ~ "06-10",
                                            between(Duration, 11, 15) ~ "11-15",
                                            between(Duration, 16, 20) ~ "16-20",
                                            between(Duration, 21, 25) ~ "21-25",
                                            .default = "26+"),
                   Face_Amount_Band = as.character(substr(Face_Amount_Band, 1, 2)),
                   Attained_Age_Grp = case_when(between(Attained_Age, 0, 17) ~ "00-17",
                                                between(Attained_Age, 18, 29) ~ "18-29",
                                                between(Attained_Age, 30, 39) ~ "30-39",
                                                between(Attained_Age, 40, 49) ~ "40-49",
                                                between(Attained_Age, 50, 59) ~ "50-59",
                                                between(Attained_Age, 60, 69) ~ "60-69",
                                                between(Attained_Age, 70, 79) ~ "70-79",
                                                between(Attained_Age, 80, 89) ~ "80-89",
                                                .default = "90+"),
                   SOA_Antp_Lvl_TP = case_when(SOA_Antp_Lvl_TP == "5 yr anticipated" ~ "05",
                                               SOA_Antp_Lvl_TP == "10 yr anticipated" ~ "10",
                                               SOA_Antp_Lvl_TP == "15 yr anticipated" ~ "15",
                                               SOA_Antp_Lvl_TP == "20 yr anticipated" ~ "20",
                                               SOA_Antp_Lvl_TP == "25 yr anticipated" ~ "25",
                                               SOA_Antp_Lvl_TP == "30 yr anticipated" ~ "30",
                                               SOA_Antp_Lvl_TP == "40 yr anticipated" ~ "40",
                                               .default = "NLT"),
                   SOA_Guar_Lvl_TP = case_when(SOA_Guar_Lvl_TP == "5 yr guaranteed" ~ "05",
                                               SOA_Guar_Lvl_TP == "10 yr guaranteed" ~ "10",
                                               SOA_Guar_Lvl_TP == "15 yr guaranteed" ~ "15",
                                               SOA_Guar_Lvl_TP == "20 yr guaranteed" ~ "20",
                                               SOA_Guar_Lvl_TP == "25 yr guaranteed" ~ "25",
                                               SOA_Guar_Lvl_TP == "30 yr guaranteed" ~ "30",
                                               SOA_Guar_Lvl_TP == "40 yr guaranteed" ~ "40",
                                               .default = "NLT"),
                   SOA_Post_Lvl_Ind = case_when(SOA_Post_Lvl_Ind %in% c("N/A","NLT") ~ "NLT",
                                                .default = SOA_Post_Lvl_Ind),
                   Preferred_Indicator = as.character(Preferred_Indicator),
                   Number_of_Pfd_Classes = case_when(as.character(Number_of_Pfd_Classes) == "NA" | is.na(Number_of_Pfd_Classes) ~ "NA",
                                                     .default = as.character(Number_of_Pfd_Classes)),
                   Preferred_Class = case_when(as.character(Preferred_Class) == "NA" | is.na(Preferred_Class) ~ "NA",
                                               .default = as.character(Preferred_Class))) %>%
            select(-c(Issue_Age, Attained_Age, Duration)) %>%
            group_by(Observation_Year, Age_Ind, Sex, Smoker_Status, Insurance_Plan, Issue_Age_Grp, Duration_Grp, 
                     Face_Amount_Band, Issue_Year, Attained_Age_Grp, SOA_Antp_Lvl_TP, SOA_Guar_Lvl_TP, SOA_Post_Lvl_Ind, 
                     Slct_Ult_Ind, Preferred_Indicator, Number_of_Pfd_Classes, Preferred_Class) %>%
            summarise(Amount_Exposed = sum(Amount_Exposed),
                      Policies_Exposed = sum(Policies_Exposed),
                      Death_Claim_Amount = sum(Death_Claim_Amount),
                      Death_Count = sum(Death_Count),
                      ExpDth_VBT2015_Cnt = sum(ExpDth_VBT2015_Cnt),
                      ExpDth_VBT2015_Amt = sum(ExpDth_VBT2015_Amt),
                      ExpDth_VBT2015wMI_Cnt = sum(ExpDth_VBT2015wMI_Cnt),
                      ExpDth_VBT2015wMI_Amt = sum(ExpDth_VBT2015wMI_Amt),
                      Cen2MomP1wMI_Amt = sum(Cen2MomP1wMI_Amt),
                      Cen2MomP2wMI_Amt = sum(Cen2MomP2wMI_Amt),
                      Cen3MomP1wMI_Amt = sum(Cen3MomP1wMI_Amt),
                      Cen3MomP2wMI_Amt = sum(Cen3MomP2wMI_Amt),
                      Cen3MomP3wMI_Amt = sum(Cen3MomP3wMI_Amt))
  Sys.sleep(5)
  gc()
}
df = bind_rows(df)

#data manipulation
df_model = df %>%
           ungroup() %>%
           select(Observation_Year, Sex, Smoker_Status, Insurance_Plan, Issue_Age_Grp, Duration_Grp, Face_Amount_Band, Issue_Year, 
                  Attained_Age_Grp, SOA_Antp_Lvl_TP, SOA_Guar_Lvl_TP, SOA_Post_Lvl_Ind, Slct_Ult_Ind, 
                  Number_of_Pfd_Classes, Preferred_Class, Policies_Exposed, Death_Count, Amount_Exposed, Death_Claim_Amount) %>%
           mutate(Preferred_Class = case_when(Number_of_Pfd_Classes == "4" & Preferred_Class == "1" ~ "1",
                                              Number_of_Pfd_Classes == "3" & Preferred_Class == "1" ~ "2",
                                              Number_of_Pfd_Classes == "2" & Preferred_Class == "1" ~ "3",
                                              Number_of_Pfd_Classes == "4" & Preferred_Class == "2" ~ "4",
                                              Number_of_Pfd_Classes == "3" & Preferred_Class == "2" ~ "5",
                                              Number_of_Pfd_Classes == "4" & Preferred_Class == "3" ~ "6",
                                              Number_of_Pfd_Classes == "2" & Preferred_Class == "2" ~ "7",
                                              Number_of_Pfd_Classes == "3" & Preferred_Class == "3" ~ "8",
                                              Number_of_Pfd_Classes == "4" & Preferred_Class == "4" ~ "9",
                                              Number_of_Pfd_Classes == "NA" | Preferred_Class == "NA" ~ "10",
                                              .default = NA_character_)) %>%
           group_by(Observation_Year, Sex, Smoker_Status, Insurance_Plan, Issue_Age_Grp, Duration_Grp, Face_Amount_Band, Issue_Year, 
                    Attained_Age_Grp, SOA_Antp_Lvl_TP, SOA_Guar_Lvl_TP, SOA_Post_Lvl_Ind, Slct_Ult_Ind, Preferred_Class) %>%
           summarise(Policies_Exposed = sum(Policies_Exposed),
                     Death_Count = sum(Death_Count),
                     Amount_Exposed = sum(Amount_Exposed), 
                     Death_Claim_Amount = sum(Death_Claim_Amount)) %>%
           ungroup() %>%
           mutate(Death_Flg = ifelse(Death_Count > 0, 1, 0)) %>%
           select(-c(Amount_Exposed, Death_Count, Death_Claim_Amount)) %>%
           distinct()

df_model = df_model %>%
           mutate(Sex = factor(case_when(Sex == "M" ~ "1",
                                         Sex == "F" ~ "2",
                                         .default = NA_character_), 
                               levels = c("1","2"), 
                               labels = c("Male","Female")),
                  Smoker_Status = factor(case_when(Smoker_Status == "NS" ~ "1",
                                                   Smoker_Status == "S" ~ "2",
                                                   Smoker_Status == "U" ~ "3",
                                                   .default = NA_character_), 
                                         levels = c("1","2","3"), 
                                         labels = c("Non Smoker","Smoker","Uni-Smoke")),
                  Insurance_Plan = factor(case_when(Insurance_Plan == "Perm" ~ "1",
                                                    Insurance_Plan == "Term" ~ "2",
                                                    Insurance_Plan == "UL" ~ "3",
                                                    Insurance_Plan == "ULSG" ~ "4",
                                                    Insurance_Plan == "VL" ~ "5",
                                                    Insurance_Plan == "VLSG" ~ "6",
                                                    .default = NA_character_),
                                          levels = c("1","2","3","4","5","6"), 
                                          labels = c("Whole Life","Term Life","Universal Life","Universal Life with Secondary Guarantees",
                                                     "Variable Universal Life","Variable Universal Life with Secondary Guarantees")),
                  Issue_Age_Grp = factor(case_when(Issue_Age_Grp == "00-17" ~ "1",
                                                   Issue_Age_Grp == "18-29" ~ "2",
                                                   Issue_Age_Grp == "30-39" ~ "3",
                                                   Issue_Age_Grp == "40-49" ~ "4",
                                                   Issue_Age_Grp == "50-59" ~ "5",
                                                   Issue_Age_Grp == "60-69" ~ "6",
                                                   Issue_Age_Grp == "70+" ~ "7",
                                                   .default = NA_character_),
                                         levels = c("1","2","3","4","5","6","7"),
                                         labels = c("00-17","18-29","30-39","40-49","50-59","60-69","70+")),
                  Duration_Grp = factor(case_when(Duration_Grp == "01-05" ~ "1",
                                                  Duration_Grp == "06-10" ~ "2",
                                                  Duration_Grp == "11-15" ~ "3",
                                                  Duration_Grp == "16-20" ~ "4",
                                                  Duration_Grp == "21-25" ~ "5",
                                                  Duration_Grp == "26+" ~ "6",
                                                  .default = NA_character_),
                                        levels = c("1","2","3","4","5","6"), 
                                        labels = c("01-05","06-10","11-15","16-20","21-25","26+")),
                  Face_Amount_Band = factor(Face_Amount_Band, 
                                            levels = c("01","02","03","04","05","06","07","08","09","10","11"),
                                            labels = c("0-9999","10000-24999","25000-49999","50000-99999","100000-249000","250000-499999",
                                                      "500000-999999","1000000-2499999","2500000-4999999","5000000-9999999","10000000+")),
                  Attained_Age_Grp = factor(case_when(Attained_Age_Grp == "00-17" ~ "1",
                                                      Attained_Age_Grp == "18-29" ~ "2",
                                                      Attained_Age_Grp == "30-39" ~ "3",
                                                      Attained_Age_Grp == "40-49" ~ "4",
                                                      Attained_Age_Grp == "50-59" ~ "5",
                                                      Attained_Age_Grp == "60-69" ~ "6",
                                                      Attained_Age_Grp == "70-79" ~ "7",
                                                      Attained_Age_Grp == "80-89" ~ "8",
                                                      Attained_Age_Grp == "90+" ~ "9",
                                                      .default = NA_character_),
                                            levels = c("1","2","3","4","5","6","7","8","9"), 
                                            labels = c("00-17","18-29","30-39","40-49","50-59","60-69","70-79","80-89","90+")),
                  SOA_Antp_Lvl_TP = factor(case_when(SOA_Antp_Lvl_TP == "05" ~ "1",
                                                     SOA_Antp_Lvl_TP == "10" ~ "2",
                                                     SOA_Antp_Lvl_TP == "15" ~ "3",
                                                     SOA_Antp_Lvl_TP == "20" ~ "4",
                                                     SOA_Antp_Lvl_TP == "25" ~ "5",
                                                     SOA_Antp_Lvl_TP == "30" ~ "6",
                                                     SOA_Antp_Lvl_TP == "40" ~ "7",
                                                     SOA_Antp_Lvl_TP == "NLT" ~ "8",
                                                     .default = NA_character_),
                                           levels = c("1","2","3","4","5","6","7","8"),
                                           labels = c("05","10","15","20","25","30","40","Non Level Term")),
                  SOA_Guar_Lvl_TP = factor(case_when(SOA_Guar_Lvl_TP == "05" ~ "1",
                                                     SOA_Guar_Lvl_TP == "10" ~ "2",
                                                     SOA_Guar_Lvl_TP == "15" ~ "3",
                                                     SOA_Guar_Lvl_TP == "20" ~ "4",
                                                     SOA_Guar_Lvl_TP == "25" ~ "5",
                                                     SOA_Guar_Lvl_TP == "30" ~ "6",
                                                     SOA_Guar_Lvl_TP == "40" ~ "7",
                                                     SOA_Guar_Lvl_TP == "NLT" ~ "8",
                                                     .default = NA_character_),
                                           levels = c("1","2","3","4","5","6","7","8"),
                                           labels = c("05","10","15","20","25","30","40","Non Level Term")),
                  SOA_Post_Lvl_Ind = factor(case_when(SOA_Post_Lvl_Ind == "WLT" ~ "1",
                                                      SOA_Post_Lvl_Ind == "PLT" ~ "2",
                                                      SOA_Post_Lvl_Ind == "NLT" ~ "3",
                                                      .default = NA_character_),
                                            levels = c("1","2","3"),
                                            labels = c("Within Level Term Period","Post Level Term Period","Non Level Term")),
                  Slct_Ult_Ind = factor(case_when(Slct_Ult_Ind == "S" ~ "1",
                                                  Slct_Ult_Ind == "U" ~ "2",
                                                  .default = NA_character_),
                                        levels = c("1","2"),
                                        labels = c("Select","Ultimate")),
                  Preferred_Class = factor(Preferred_Class,
                                           levels = c("1","2","3","4","5","6","7","8","9","10"),
                                           labels = c("4-1","3-1","2-1","4-2","3-2","4-3","2-2","3-3","4-4","NA")),
                  Death_Flg = factor(Death_Flg,
                                     levels = c("0","1"),
                                     labels = c("Alive","Dead")))
label1 = df_model$Death_Flg
feature1 = df_model[,c(1:14)]

#under-sampling
df_model1 = downSample(feature1, label1, yname = "Death_Flg")

#train test split
ind1 = sample(2, nrow(df_model1), replace = TRUE, prob = c(0.8, 0.2))
train1 = df_model1[ind1 == 1,]
test1 = df_model1[ind1 == 2,]

#EDA
#numerical features
par(mfrow = c(1, 2))
boxplot(train1$Observation_Year)
title(expression(underline("Observation Year")))
boxplot(train1$Issue_Year)
title(expression(underline("Issue Year")))

#categorical features
p1 = ggplot(data = train1, aes(x = Sex, fill = Sex)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Sex"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p2 = ggplot(data = train1, aes(x = Smoker_Status, fill = Smoker_Status)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Smoker_Status"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p3 = ggplot(data = train1, aes(x = Insurance_Plan, fill = Insurance_Plan)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Insurance_Plan"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p4 = ggplot(data = train1, aes(x = Issue_Age_Grp, fill = Issue_Age_Grp)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Issue_Age_Grp"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p5 = ggplot(data = train1, aes(x = Duration_Grp, fill = Duration_Grp)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Duration_Grp"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p6 = ggplot(data = train1, aes(x = Face_Amount_Band, fill = Face_Amount_Band)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Face_Amount_Band"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = c("01","02","03","04","05","06","07","08","09","10","11")) +
      theme_classic(base_size = 12) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5))
p7 = ggplot(data = train1, aes(x = Attained_Age_Grp, fill = Attained_Age_Grp)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Attained_Age_Grp"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p8 = ggplot(data = train1, aes(x = SOA_Antp_Lvl_TP, fill = SOA_Antp_Lvl_TP)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("SOA_Antp_Lvl_TP"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) + +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p9 = ggplot(data = train1, aes(x = SOA_Guar_Lvl_TP, fill = SOA_Guar_Lvl_TP)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("SOA_Guar_Lvl_TP"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p10 = ggplot(data = train1, aes(x = SOA_Post_Lvl_Ind, fill = SOA_Post_Lvl_Ind)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("SOA_Post_Lvl_Ind"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p11 = ggplot(data = train1, aes(x = Slct_Ult_Ind, fill = Slct_Ult_Ind)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Slct_Ult_Ind"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p12 = ggplot(data = train1, aes(x = Preferred_Class, fill = Preferred_Class)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Preferred_Class"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))

(p1|p2)/(p3|p4)
(p5|p6)/(p7|p8)
(p9|p10)/(p11|p12)

#label
p13 = ggplot(data = df_model, aes(x = Death_Flg, fill = Death_Flg)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Death_Flg: Before Under-sampling"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))
p14 = ggplot(data = train1, aes(x = Death_Flg, fill = Death_Flg)) +
      geom_bar(width = 0.5, colour = "black") +
      labs(title = expression(underline("Death_Flg: After Under-sampling"))) +
      xlab("") +
      ylab("Count") +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
      theme_classic(base_size = 12) +
      theme(legend.position="none",
            plot.title = element_text(face = "bold", hjust = 0.5))

p13|p14

#optimize num.trees
x = matrix(nrow = 10, ncol = 2)
for (i in c(1:10)) {
  rf_model = ranger(Death_Flg ~ ., data = train1, num.trees = i * 50)
  x[i,1] = i * 50
  x[i,2] = rf_model$prediction.error
}
oob_error_x = data.frame(x)
colnames(oob_error_x) = c("Num_Trees","OOB_Prediction_Error")

y = matrix(nrow = 5, ncol = 2)
for (i in c(1:5)) {
  rf_model = ranger(Death_Flg ~ ., data = train1, num.trees = 500 + i * 100)
  y[i,1] = 500 + i * 100
  y[i,2] = rf_model$prediction.error
}
oob_error_y = data.frame(y)
colnames(oob_error_y) = c("Num_Trees","OOB_Prediction_Error")

z = matrix(nrow = 2, ncol = 2)
for (i in c(1:2)) {
  rf_model = ranger(Death_Flg ~ ., data = train1, num.trees = 1000 + i * 500)
  z[i,1] = 1000 + i * 500
  z[i,2] = rf_model$prediction.error
}
oob_error_z = data.frame(z)
colnames(oob_error_z) = c("Num_Trees","OOB_Prediction_Error")

oob_error = bind_rows(oob_error_x, oob_error_y, oob_error_z)
colnames(oob_error) = c("Num_Trees","OOB_Prediction_Error")
ggplot(data = oob_error, aes(x = Num_Trees, y = OOB_Prediction_Error)) +
  geom_line(linewidth = 1) +
  xlab("Number of Trees(num.trees)") +
  ylab("OOB Prediction Error") +
  theme_classic(base_size = 12)

#hyper-parameter tuning
grid = expand.grid(.mtry = seq(5,15,5),
                   .splitrule = "gini",
                   .min.node.size = c(1,5,10))
tune_model = caret::train(Death_Flg ~ ., 
                          data = train1,
                          method = "ranger",
                          trControl = trainControl(method="cv", number = 5),
                          tuneGrid = grid,
                          num.trees = 1000)

rf_model1_tuned = ranger(Death_Flg ~ ., 
                          data = train1, 
                          num.trees = 1000,
                          mtry = 14,
                          min.node.size = 10,
                          importance = "permutation",
                          splitrule = "gini")

p1_tuned = predict(rf_model1_tuned, test1)
confusionMatrix(p1_tuned$predictions, test1$Death_Flg)
vi1_tuned = data.frame(rf_model1_tuned$variable.importance) %>%
            mutate(Feature = rownames(data.frame(rf_model1_tuned$variable.importance)))
colnames(vi1_tuned)[1] = "Importance"
rownames(vi1_tuned) = NULL
vi1_tuned = vi1_tuned %>%
            select(Feature, Importance) %>% 
            arrange(desc(Importance)) %>%
            mutate(Feature = factor(Feature, levels = Feature))
ggplot(vi1_tuned, aes(x = Feature, y = Importance, fill = Feature)) +
  geom_bar(stat = "identity", width = 0.5, colour = "black") +
  geom_text(aes(label=Feature), vjust=-0.5, position = position_dodge(width = 2), size = 3.5) +
  scale_x_discrete(labels = NULL, breaks = NULL) + 
  labs(x = NULL) +
  theme_classic(base_size = 12) +
  theme(legend.position="none")

#SHAP value
shap_ind = sample(nrow(train1), 200)
shap_sample1 = train1[shap_ind,]
shap_ind2 = sample(nrow(shap_sample1), 20)
shap_sample2 = shap_sample1[shap_ind2,]

rf_model_shap1 = ranger(Death_Flg ~ Face_Amount_Band + Preferred_Class + Insurance_Plan + Issue_Year + Attained_Age_Grp + Issue_Age_Grp + Observation_Year, 
                        data = train1,
                        num.trees = 1000,
                        mtry = 7,
                        min.node.size = 10,
                        importance = "permutation",
                        splitrule = "gini",
                        probability = TRUE)

s1 = kernelshap(rf_model_shap, shap_sample1, bg_X = shap_sample2,
                feature_names = c("Face_Amount_Band","Preferred_Class","Insurance_Plan","Issue_Year","Attained_Age_Grp","Issue_Age_Grp","Observation_Year"))
sv1 = shapviz(s1)
sv_importance(sv1$Dead, kind = "bee") +
  theme_classic(base_size = 16)

sv_dependence(sv1$Dead, v = "Face_Amount_Band", color_var = "Face_Amount_Band") +
  scale_x_discrete(labels = c("01","02","03","04","05","06","07","08","09","10","11")) +
  theme_classic(base_size = 16)

sv_dependence(sv1$Dead, v = "Insurance_Plan", color_var = NULL) +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme_classic(base_size = 16)

sv_dependence(sv1$Dead, v = "Insurance_Plan", color_var = "Face_Amount_Band") +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme_classic(base_size = 16)

shap_ind = sample(nrow(train1), 500)
shap_sample1 = train1[shap_ind,]
shap_ind2 = sample(nrow(shap_sample1), 50)
shap_sample2 = shap_sample1[shap_ind2,]

rf_model_shap2 = ranger(Death_Flg ~ Preferred_Class + SOA_Guar_Lvl_TP + Duration_Grp + Smoker_Status + Sex, 
                        data = train1,
                        num.trees = 1000,
                        mtry = 5,
                        min.node.size = 10,
                        importance = "permutation",
                        splitrule = "gini",
                        probability = TRUE)

s2 = kernelshap(rf_model_shap2, shap_sample1, bg_X = shap_sample2,
                feature_names = c("Preferred_Class","SOA_Guar_Lvl_TP","Duration_Grp","Smoker_Status","Sex"))
sv2 = shapviz(s2)
sv_importance(sv2$Dead, kind = "bee") +
  theme_classic(base_size = 16)

sv_dependence(sv2$Dead, v = "SOA_Guar_Lvl_TP", color_var = "Duration_Grp") +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme_classic(base_size = 16)

sv_dependence(sv2$Dead, v = "Preferred_Class", color_var = "Smoker_Status") +
  scale_x_discrete(labels = function(x) str_wrap(x, width = 10)) +
  theme_classic(base_size = 16)