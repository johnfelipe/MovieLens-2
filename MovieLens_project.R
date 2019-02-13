# libraries ####
library(caret)
library(anytime)
library(tidyverse)
library(lubridate)

# Load the data ####
#Create test and validation sets

# Create edx set, validation set, and submission file

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Formats ####
str(edx)

edx$userId <- as.factor(edx$userId) # Convert `userId` to `factor`.
edx$movieId <- as.factor(edx$movieId) # Convert `movieId` to `factor`.
edx$genres <- as.factor(edx$genres) # Convert `genres` to `factor`.
edx$timestamp <- as.POSIXct(edx$timestamp, origin = "1970-01-01") # Convert `timestamp to `POSIXct`.

edx <- edx %>% # It extracts the release year of the movie and creates `year` column.
  mutate(title = str_trim(title)) %>%
  extract(title, c("title_tmp", "year"),
          regex = "^(.*) \\(([0-9 \\-]*)\\)$",
          remove = F) %>%
  mutate(year = if_else(str_length(year) > 4,
                        as.integer(str_split(year, "-",
                                             simplify = T)[1]),
                        as.integer(year))) %>%
  mutate(title = if_else(is.na(title_tmp), title, title_tmp)) %>%
  select(-title_tmp)  %>%
  mutate(genres = if_else(genres == "(no genres listed)",
                          `is.na<-`(genres), genres))

edx <- edx %>% mutate(year_rate = year(timestamp))
# It extracts the year that the rate was given by the user.

edx <- edx %>% select(-title, -timestamp) # Drop `title` & `timestamp` columns.
edx$genres <- as.factor(edx$genres)

# Exploratory ####
summary(edx$rating)

edx %>% ggplot(aes(rating)) +
  geom_histogram(fill = "darkgreen") +
  labs(title = "Rating distribution",
       x = "Rate",
       y = "Frequency")

# Unique users
edx %>%
  summarize(Unique_Users = n_distinct(userId),
            Unique_Movies = n_distinct(movieId),
            Unique_Genres = n_distinct(genres))

# Year
edx %>% ggplot(aes(year)) +
  geom_histogram(fill = "darkblue") +
  labs(title = "Year distribution",
       subtitle = "Rates by year",
       x = "Year",
       y = "Freq")

# Genres rating ####
g <- edx %>%
  select(genres, rating) %>%
  group_by(genres) %>%
  summarize(mean = mean(rating), median = median(rating), n = n()) %>%
  arrange(desc(mean)) %>%
  head(20)

print(g)

# Genres rating appearences ####
g[2:nrow(g), ] %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  head(15) %>%
  group_by(genres) %>%
  summarise(appearances = sum(n)) %>%
  arrange(desc(appearances))

# Genres appearences ####
edx %>%
  select(genres, rating) %>%
  group_by(genres) %>%
  summarize(mean = mean(rating), median = median(rating), n = n()) %>%
  arrange(desc(mean)) %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  head(20) %>%
  group_by(genres) %>%
  summarise(appearances = sum(n)) %>%
  arrange(desc(appearances)) %>%
  head(10)

# Train & Test set ####
edx <- edx %>% select(userId, movieId, rating)

test_index <- createDataPartition(edx$rating, times = 1, p = .2, list = F)
# Create the index

train <- edx[-test_index, ] # Create Train set
test <- edx[test_index, ] # Create Test set
test <- test %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

dim(train)
dim(test)

# Baseline ####
mu_hat_baseline <- mean(train$rating) # Mean accross all movies.
RMSE_baseline <- RMSE(test$rating, mu_hat_baseline)
RMSE_baseline

# Table 1 ####
rmse_table <- data_frame(Method = "Baseline", RMSE = RMSE_baseline)
rmse_table %>% knitr::kable(caption = "RMSEs")

# User and Movie effect Model ####
mu <- mean(train$rating)

movie_avgs <- train %>%
  group_by(movieId) %>%
  summarize(m_i = mean(rating - mu))

user_avgs <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(u_i = mean(rating - mu - m_i))

predicted_ratings <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + m_i + u_i) %>% .$pred

model_RMSE <- RMSE(predicted_ratings, test$rating)
model_RMSE 

# Table 2 ####
rmse_table <- rbind(rmse_table,
                    data_frame(Method = "User & Movie Effect", RMSE = model_RMSE))

rmse_table %>% knitr::kable(caption = "RMSEs")

# User and Movie effect Model on "validation" data ####
validation <- validation %>% select(userId, movieId, rating)

validation$userId <- as.factor(validation$userId)
validation$movieId <- as.factor(validation$movieId)

validation <- validation[complete.cases(validation), ]

# Predictions on "validation"
mu <- mean(train$rating)

movie_avgs <- train %>%
  group_by(movieId) %>%
  summarize(m_i = mean(rating - mu))

user_avgs <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(u_i = mean(rating - mu - m_i))

predicted_ratings <- test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + m_i + u_i) %>% .$pred

predicted_val <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + m_i + u_i) %>% .$pred

val_RMSE <- RMSE(predicted_val, validation$rating, na.rm = T)
val_RMSE

# Table 3 ####
rmse_table_val <- data_frame(Method = "User & Movie Effect on validation", RMSE = val_RMSE)
rmse_table_val %>% knitr::kable(caption = "RMSEs on validation data set")

# Regularization ####
lambda_values <- seq(0, 7, .2)

RMSE_function_reg <- sapply(lambda_values, function(l){
  
  mu <- mean(train$rating)
  
  m_i <- train %>%
    group_by(movieId) %>%
    summarize(m_i = sum(rating - mu)/(n()+l))
  
  u_i <- train %>%
    left_join(m_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(u_i = sum(rating - m_i - mu)/(n()+l))
  
  predicted_ratings <- test %>%
    left_join(m_i, by = "movieId") %>% 
    left_join(u_i, by = "userId") %>%
    mutate(pred = mu + m_i + u_i) %>% .$pred
  
  return(RMSE(predicted_ratings, test$rating))
})

qplot(lambda_values, RMSE_function_reg,
      main = "Regularisation",
      xlab = "RMSE", ylab = "Lambda") # lambda vs RMSE

lambda_opt <- lambda_values[which.min(RMSE_function_reg)]
lambda_opt # Lambda which minimizes RMSE

# Table 4 ####
rmse_table <- rbind(rmse_table,
                    data_frame(Method = "User & Movie Effect Regularisation",
                               RMSE = min(RMSE_function_reg)))

rmse_table %>% knitr::kable(caption = "RMSEs")

# Regularisation on "validation" data set ####
RMSE_function_val_reg <- sapply(lambda_values, function(l){
  
  mu <- mean(train$rating)
  
  m_i <- train %>%
    group_by(movieId) %>%
    summarize(m_i = sum(rating - mu)/(n()+l))
  
  u_i <- train %>%
    left_join(m_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(u_i = sum(rating - m_i - mu)/(n()+l))
  
  predicted_val_reg <- validation %>%
    left_join(m_i, by = "movieId") %>% 
    left_join(u_i, by = "userId") %>%
    mutate(pred = mu + m_i + u_i) %>% .$pred
  
  return(RMSE(predicted_val_reg, validation$rating, na.rm = T))
})

qplot(lambda_values, RMSE_function_val_reg,
      main = "Regularisation on validation data set",
      xlab = "Lambda", ylab = "RMSE")

lambda_opt_reg <- lambda_values[which.min(RMSE_function_val_reg)]
lambda_opt_reg # Lambda which minimizes RMSE

min_rmse <- min(RMSE_function_val_reg) # Best RMSE
min_rmse

# Table 5 ####
rmse_table_val <- rbind(rmse_table_val,
                        data_frame(Method = "User & Movie Effect Reg. on validation",
                                   RMSE = min(RMSE_function_val_reg)))
rmse_table_val %>% knitr::kable(caption = "RMSEs on validation data set") 

# RMSE Summary Table ####
rbind(rmse_table, rmse_table_val) %>%
  knitr::kable(caption = "RMSEs Summary", "latex", booktabs = T) %>%
  row_spec(5, bold = T, color = "white", background = "green")
