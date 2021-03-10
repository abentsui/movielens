library(tidyverse)
library(caret)
library(data.table)

#Save the data into rda
save(edx, file = "edx.rda")
save(validation, file = "validation.rda")

#Load data
load("edx.rda")
load("validation.rda")


#Part 1: Some basic definition
#Define the calculation of RMSE function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#Part 2: Prepare the train and test set using edx
#Separate train and test set
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, 
                                  p = 0.2, list = FALSE)
test_set <- edx[test_index,]
train_set <- edx[-test_index,]

#Ensure the users and movies which are NOT in the training set removed in test set
test_set <- test_set %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "movieId")

save(train_set, file = "train_set.rda")


#Part 3: Study of the data

#Histogram of the no. of ratings per user
train_set %>%
  group_by(userId) %>%
  summarize(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_continuous(trans="log10") +
  ggtitle("Users")

#Histogram of the no. of ratings per movie
train_set %>%
  group_by(movieId) %>%
  summarize(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_continuous(trans="log10") +
  ggtitle("Movies")

#Filter out user with no. of ratings less than 100
user_filter <- train_set %>%
  group_by(userId) %>%
  summarize(n = n()) %>%
  filter(n >= 100) %>%
  pull(userId)

#Filter out movie with no. of ratings less than 400
movie_filter <- train_set %>%
  group_by(movieId) %>%
  summarize(n = n()) %>%
  filter(n >= 400) %>%
  pull(movieId)


#Part 4: Principal Component Analysis (PCA) of the movielens data

#Prepare the matrix for PCA
sel_y <- train_set %>%
  select(userId, movieId, rating) %>%
  filter(movieId %in% movie_filter) %>%
  filter(userId %in% user_filter) %>%
  spread(movieId, rating) %>%
  as.matrix()

#Modify sel_y with setting row names
rownames(sel_y) <- sel_y[,1]
sel_y <-sel_y[,-1]

#Subtract the rowMeans, colMeans for each entry
sel_y <- sweep(sel_y, 1, rowMeans(sel_y, na.rm = TRUE))
sel_y <- sweep(sel_y, 2, colMeans(sel_y, na.rm = TRUE))

#We make all NAs as 0
sel_y[is.na(sel_y)] <- 0
sel_y <- sweep(sel_y, 1, rowMeans(sel_y))

#Perform PCA
temp_pca <- prcomp(sel_y, rank = 400)

#Check the variance explained by the principal components
plot(temp_pca$sdev)

var_explained <- cumsum(temp_pca$sdev^2/sum(temp_pca$sdev^2))

jpeg('temp_pca_plot.jpg')
plot(var_explained)
dev.off()

min(which(var_explained > 0.5))

#We multiply the matrix in PCA to get the residual explained by the principal components
pca_result <- temp_pca$x %*% t(temp_pca$rotation)

#We turn the matrix into a tibble for prediction
pca_df <- pca_result %>%
  as_tibble() %>%
  mutate(userId = rownames(pca_result)) %>%
  gather(movieId, res, -userId) %>%
  mutate(movieId = as.numeric(movieId)) %>%
  mutate(userId = as.numeric(userId))


#Part 5: Run the prediction with regularized user and movie effect
#Tuning for lambda
lambdas <- seq(0, 10, 0.25)

#Perform prediction and see which lambda achieves best RMSE for regularized user and movie effect
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(l + n()))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(l + n()))
  
  #Prediction using test set
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

min(rmses)


#Part 6: Run the prediction with regularized user and movie effect + PCA effect
#Use the best lambda found in the previous section
l <- lambdas[which.min(rmses)]
mu <- mean(train_set$rating)

b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(l + n()))

b_u <- train_set %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(l + n()))

#Use the test set to see the RMSE
#We left join the pca_df to include the PCA effect
predicted_ratings <- test_set %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(pca_df, by = c("movieId", "userId")) %>% #Left join the PCA result
  mutate_at("res", ~replace(., is.na(.), 0)) %>% #Replace the NA values by 0
  mutate(pred = mu + b_i + b_u + res) %>%
  pull(pred)

RMSE(test_set$rating, predicted_ratings)


#Part 7: Refit the model to the entire edx data set
movie_filter <- edx %>%
  group_by(movieId) %>%
  summarize(n = n()) %>%
  filter(n >= 400) %>%
  pull(movieId)

user_filter <- edx %>%
  group_by(userId) %>%
  summarize(n = n()) %>%
  filter(n >= 100) %>%
  pull(userId)

sel_y <- edx %>%
  select(userId, movieId, rating) %>%
  filter(movieId %in% movie_filter) %>%
  filter(userId %in% user_filter) %>%
  spread(movieId, rating) %>%
  as.matrix()

#Modify sel_y with setting row names
rownames(sel_y) <- sel_y[,1]
sel_y <-sel_y[,-1]

#Subtract the rowMeans, colMeans for each entry
sel_y <- sweep(sel_y, 1, rowMeans(sel_y, na.rm = TRUE))
sel_y <- sweep(sel_y, 2, colMeans(sel_y, na.rm = TRUE))

#We make all NAs as 0
sel_y[is.na(sel_y)] <- 0
sel_y <- sweep(sel_y, 1, rowMeans(sel_y))

#Perform PCA
temp_pca <- prcomp(sel_y, rank = 400)

#We multiply the matrix in PCA to get the residual explained by the principal components
pca_result <- temp_pca$x %*% t(temp_pca$rotation)

#We turn the matrix into a tibble for prediction
pca_df <- pca_result %>%
  as_tibble() %>%
  mutate(userId = rownames(pca_result)) %>%
  gather(movieId, res, -userId) %>%
  mutate(movieId = as.numeric(movieId)) %>%
  mutate(userId = as.numeric(userId))


#Get statistics using edx data set
mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(l + n()))

b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(l + n()))


#Part 8: Predict wth the validation set
#Predict validation set using edx data set results(b_i, b_u, pca_df)
#Regularized movie and user effect + PCA effect
predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(pca_df, by = c("movieId", "userId")) %>%
  mutate_at("res", ~replace(., is.na(.), 0)) %>%
  mutate(pred = mu + b_i + b_u + res) %>%
  pull(pred)

RMSE(validation$rating, predicted_ratings)


#Predict validation set using edx data set results without using PCA results
predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

RMSE(validation$rating, predicted_ratings)


#Predict validation set with using average to guess
RMSE(validation$rating, mu)
