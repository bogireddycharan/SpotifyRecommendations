# Activate the packages

library("caret")
library("ggplot2")
library("tidyr")
library("tidyverse")
library("dplyr")
#if(!require(devtools)) install.packages("devtools")
#devtools::install_github("kassambara/factoextra")
library("factoextra")
library("reshape2")
library('cluster')
library('glmnet')
library('randomForest')
library('e1071')
library('MASS')

# Importing the dataset

songs <- read.csv("SpotifyData.csv")

dim(songs)

colnames(songs)


# Data Cleaning

# Removing the index column

songs <- songs %>% select(-X)


# Check for missing values

is.null(songs)

# There ar no missing values in the data.

# Duplicate values

songs <- songs %>% distinct(song_title, artist, .keep_all = TRUE)

# 35 duplicate rows were omitted based on song_title, artist

# Check the datatypes of columns

str(songs)

# Exploratory Data Analysis

# Analyzing the data distribution of different variables using histogram

hist(songs$acousticness)
hist(songs$danceability)
hist(songs$duration_ms)
hist(songs$energy)
hist(songs$instrumentalness)
hist(songs$key)
hist(songs$liveness)
hist(songs$loudness)
hist(songs$speechiness)
hist(songs$tempo)
hist(songs$time_signature)
hist(songs$valence)

# The top 10 artists with more songs in the spotify data

popular_artists <- songs %>%
  group_by(artist) %>%
  summarize(Freq=n()) %>%
  arrange(desc(Freq)) %>%
  head(10) %>% as.data.frame()

ggplot(popular_artists, aes(x = reorder(artist, -Freq), y = Freq)) + geom_bar(stat = 'identity') + 
  xlab("Artist") +
  ylab("Songs by artist") +
  ggtitle("Songs by Top-10 artists")

# Drawing the Correlation heat map

#calculate correlation between each pairwise combination of variables
cor_songs <- round(cor(songs[1:14]), 2)

#melt the data frame
melted_cormat <- melt(cor_songs)

#create correlation heatmap

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() +
  geom_text(aes(Var2, Var1, label = value), size = 5) +
  scale_fill_gradient2(low = "blue", high = "red",
                       limit = c(-1,1), name="Correlation") +
  theme(axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.background = element_blank())

# From the correlation plots, we can see that loudness & energy have higher correlation.
# Moreover, acousticness is highly negatively correlated with energy and loudness.

# Drawing the scatterplots between highly correlated variables

# Scatter plot between energy and loudness

ggplot(songs, aes(x = loudness, y = energy)) + 
  geom_point()

# Scatter plot between acousticness and energy

ggplot(songs, aes(x = acousticness, y = energy)) + 
  geom_point()

# Scatter plot between acousticness and loudness

ggplot(songs, aes(x = acousticness, y = loudness)) + 
  geom_point()

# convert the duration_ms to duration_min for feasibility

songs$duration_ms <- round((songs$duration_ms/(1000 * 60)),2)

# Change the column name to duration_min

colnames(songs)[colnames(songs) == "duration_ms"] ="duration_min"

# Explore the distribution of songs based on song duration_min

hist(songs$duration_min, breaks = 15, col = "cyan4")

# The songs with duration between 3 to 4 min are higher in number.


# Data Pre-Processing

# Finding the outliers using box plots & density plots

boxplot(songs$acousticness, pch = 4, main = 'Box plot of Acousticness')
densityplot(songs$acousticness, pch = 6)

boxplot(songs$danceability, pch = 4, main = 'Box plot of Danceability')
densityplot(songs$danceability, pch = 6)

boxplot(songs$duration_min, pch = 4, main = 'Box plot of duration_min')
densityplot(songs$duration_min, pch = 6)

boxplot(songs$energy, pch = 4, main = 'Box plot of energy')
densityplot(songs$energy, pch = 6)

boxplot(songs$instrumentalness, pch = 4, main = 'instrumentalness')
densityplot(songs$instrumentalness, pch = 6)

boxplot(songs$key, pch = 4, main = 'key')
densityplot(songs$key, pch = 6)

boxplot(songs$liveness, pch = 4, main = 'liveness')
densityplot(songs$liveness, pch = 6)

boxplot(songs$loudness, pch = 4, main = 'loudness')
densityplot(songs$loudness, pch = 6)

boxplot(songs$speechiness, pch = 4, main = 'speechiness')
densityplot(songs$speechiness, pch = 6)

boxplot(songs$tempo, pch = 4, main = 'tempo')
densityplot(songs$tempo, pch = 6)

boxplot(songs$valence, pch = 4, main = 'valence')
densityplot(songs$valence, pch = 6)

# The less number of outliers observed in the density plots are formed as part of the data distribution. Hence, we want to retain the outliers as they have less influence on the remaining data points.

# Feature Reduction

# Low Variance Filter

# Checking for variance of all variables to identify low variance variables

songs_variances <- songs %>% select(-c('song_title', 'artist')) %>% 
  summarize(across(everything(), ~ var(scale(., center = FALSE), na.rm = TRUE))) %>% 
  pivot_longer(everything(), names_to = "feature", values_to = "variance") %>% 
  arrange(desc(variance))

songs_variances

# The variable time_signature has very low variance (0.00418) compared to all.

# Hence, we are dropping the time_signature variable for further analysis.

songs <- songs %>% select(-time_signature)

# High Correlation Filter

# From the correlation heat map, we have seen that there is a high correlation (0.76) between loudness and energy.

cor(songs$loudness, songs$energy)

# In general, an absolute correlation coefficient of >0.7 among two or more predictors indicates the presence of multicollinearity.

# Hence, we wanted to keep either one of loudness or energy variable. Hence, we are dropping loudness feature.

songs <- songs %>% select(-loudness)

# Feature Reduction

# Applying PCA

#calculate principal components
results <- prcomp(songs[1:12], scale = TRUE)

#reverse the signs
results$rotation <- -1*results$rotation

#display principal components
results$rotation

#calculate total variance explained by each principal component
results$sdev^2 / sum(results$sdev^2)

# From the above PCA analysis and variances explained, we can see that

# To explain 88% of variance, we need to consider 9 PCA components.

components <- 9
selected_components <- results$rotation[,1:components]
selected_components

# The principle component analysis is not allowing feature reduction significantly. Hence, we are going with the actual data for clustering analysis.

# Standardize the songs data

songs[1:12] <- scale(songs[1:12])

# Remove the data rows with outliers above and below 3 standard deviations

songs <- subset(songs, songs$duration_min < 3)
songs <- subset(songs, songs$speechiness < 3)
songs <- subset(songs, songs$liveness < 3)
songs <- subset(songs, songs$tempo < 3)
songs <- subset(songs, songs$energy > -3)

# Remove the columns with song_title and artist

songs <- songs %>% select(-c("song_title", "artist"))

# final dataset for model building

head(songs) #sample of data


# Clustering

# K-means clustering

# Using the K-Means procedure, create clusters with k=2,3,4,5,6,7.

# Measuring the clustering distance.
distance <- get_dist(songs)
#distance

#Visualize the distance matrix
#fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# K-means clustering with variable centers

k2 <- kmeans(songs, centers = 2, nstart = 25,iter.max = 10)
k3 <- kmeans(songs, centers = 3, nstart = 25,iter.max = 10)
k4 <- kmeans(songs, centers = 4, nstart = 25, iter.max = 10)
k5 <- kmeans(songs, centers = 5, nstart = 25, iter.max = 10)
k6 <- kmeans(songs, centers = 6, nstart = 25, iter.max = 10)
k7 <- kmeans(songs, centers = 7, nstart = 25, iter.max = 10)

#Check outputs and understand the meaning of outputs. 
k2
str(k2)

# Plot K2

plot(songs$acousticness, songs$danceability,
     col=k2$cluster, pch=as.numeric(k2$cluster))
centers <- data.frame(cluster=factor(1:2), k2$centers) #k=2
points(centers$acousticness, centers$danceability,
       col=centers$cluster, pch=as.numeric(centers$cluster),
       cex=3, lwd=3)

# Plot K3

plot(songs$acousticness, songs$danceability,
     col=k3$cluster, pch=as.numeric(k3$cluster))
centers <- data.frame(cluster=factor(1:3), k3$centers) #k=3
points(centers$acousticness, centers$danceability,
       col=centers$cluster, pch=as.numeric(centers$cluster),
       cex=3, lwd=3)

# Plot K4

plot(songs$acousticness, songs$danceability,
     col=k4$cluster, pch=as.numeric(k4$cluster))
centers <- data.frame(cluster=factor(1:4), k4$centers) #k=4
points(centers$acousticness, centers$danceability,
       col=centers$cluster, pch=as.numeric(centers$cluster),
       cex=3, lwd=3)

# Plot K5

plot(songs$acousticness, songs$danceability,
     col=k5$cluster, pch=as.numeric(k5$cluster))
centers <- data.frame(cluster=factor(1:5), k5$centers) #k=5
points(centers$acousticness, centers$danceability,
       col=centers$cluster, pch=as.numeric(centers$cluster),
       cex=3, lwd=3)

# Plot k6

plot(songs$acousticness, songs$danceability,
     col=k6$cluster, pch=as.numeric(k6$cluster))
centers <- data.frame(cluster=factor(1:6), k6$centers) #k=6
points(centers$acousticness, centers$danceability,
       col=centers$cluster, pch=as.numeric(centers$cluster),
       cex=3, lwd=3)

# Plot k7

plot(songs$acousticness, songs$danceability,
     col=k7$cluster, pch=as.numeric(k7$cluster))
centers <- data.frame(cluster=factor(1:7), k7$centers) #k=7
points(centers$acousticness, centers$danceability,
       col=centers$cluster, pch=as.numeric(centers$cluster),
       cex=3, lwd=3)

# 2.3.2 Create the WSS plots as demonstrated in class and select a suitable k value based on the “elbow”. [Use the code that we discuss in the class.]

#function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(songs, k, nstart = 25)$tot.withinss
}
k.values <-1:15

# extract wss for 2-15 clusters
wss_values <- sapply(k.values, wss)

par(mfrow = c(1, 1))
plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     main="Elbow Chart for Clusters",
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

# K=5 is the optimum number of clusters

# Alternative method of plotting "wss".
fviz_nbclust(songs, FUN = kmeans, method = "wss")

# K=5 is the suitable number of clusters.

# Plot K5

plot(songs$acousticness, songs$danceability,
     col=k5$cluster, pch=as.numeric(k5$cluster))
centers <- data.frame(cluster=factor(1:5), k5$centers) #k=5
points(centers$acousticness, centers$danceability,
       col=centers$cluster, pch=as.numeric(centers$cluster),
       cex=3, lwd=3)

# Print the cluster centers
print(k5$centers)

# Print the cluster assignments for each data point
print(k5$cluster)

k5
str(k5)


# Hierarchical Clustering

# Calculate Euclidean distances between data points
distances <- dist(songs)

# Perform hierarchical clustering using complete linkage
hclust_model <- hclust(distances, method = "complete")
hclust_model

# Plot the dendrogram to visualize the hierarchical clustering
plot(hclust_model, hang = -1, main = "Hierarchical Clustering Dendrogram", xlab = "Songs")


# Finding the optimum number of clusters

# Calculate the within-cluster sum of squares (WSS)
wss <- numeric(length = 20)  # To store WSS values for 10 clusters

for (k in 1:20) {
  # Cut the dendrogram to get clusters
  clusters <- cutree(hclust_model, k)
  
  # Calculate the WSS for the current cluster configuration
  current_wss <- 0
  for (i in 1:k) {
    cluster_points <- songs[clusters == i, ]
    cluster_center <- colMeans(cluster_points)
    cluster_size <- sum(clusters == i)
    current_wss <- current_wss + sum((cluster_points - cluster_center)^2)
  }
  
  wss[k] <- current_wss
}

# Draw the elbow plot
plot(1:20, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of Clusters", ylab = "Within-Cluster Sum of Squares",
     main = "Elbow Plot for Hierarchical Clustering")

# This code calculates the WSS for the first 10 clusters by iteratively cutting the dendrogram and computing the sum of squares for each cluster configuration. The elbow plot helps you visualize the point where the decrease in WSS starts to slow down, indicating a potential optimal number of clusters.


# Identify the "elbow point" visually from the plot

# The elbow point is at 5 clusters

num_clusters <- 5
clusters <- cutree(hclust_model, k = num_clusters)
# clusters

# Print the cluster assignments for each data point
print(clusters)

plot(clusters, main = 'Hierarchical clustering Dendogram with 5 clusters', xlab = 'songs')

# Performance evaluation

# Initialize variables for WCSS and BCSS
total_wcss <- 0
total_bcss <- 0

# Calculate the centroid of the entire dataset

total_centroid <- colMeans(songs)

# Loop through each cluster

for (i in 1:5) {
  cluster_points <- songs[clusters == i, ]
  cluster_center <- colMeans(cluster_points)
  cluster_size <- nrow(cluster_points)
  
  # Calculate WCSS for the cluster
  total_wcss <- total_wcss + sum((cluster_points - cluster_center)^2)
  
  # Calculate BCSS for the cluster
  bcss_contribution <- sum(((cluster_center - total_centroid)^2) * cluster_size)
  total_bcss <- total_bcss + bcss_contribution
}

# Print the results
print(paste("Total WCSS:", total_wcss))
print(paste("Total BCSS:", total_bcss))

# Calculate individual wcss and bcss

# Calculate WCSS and BCSS for each cluster
cluster_wcss <- numeric(5)
cluster_bcss <- numeric(5)

for (i in 1:5) {
  cluster_points <- songs[clusters == i, ]
  cluster_center <- colMeans(cluster_points)
  cluster_wcss[i] <- sum((cluster_points - cluster_center)^2)
  cluster_bcss[i] <- sum((cluster_center - colMeans(songs))^2) * nrow(cluster_points)
}

print("Individual Cluster WCSS:")
print(cluster_wcss)

print("Individual Cluster BCSS:")
print(cluster_bcss)


# Agglomerative clustering Method


# Perform hierarchical clustering using agnes function
agnes_model <- agnes(distances, method = "complete")
agnes_model

# Plot the dendrogram to visualize the hierarchical clustering
plot(agnes_model, main = "Agglomerative Clustering Dendrogram", xlab = "Songs")



# Finding the optimum number of clusters

# Calculate the within-cluster sum of squares (WSS)
wss <- numeric(length = 20)  # To store WSS values for 10 clusters

for (k in 1:20) {
  # Cut the dendrogram to get clusters
  clusters_agnes <- cutree(agnes_model, k)
  
  # Calculate the WSS for the current cluster configuration
  current_wss <- 0
  for (i in 1:k) {
    cluster_points <- songs[clusters_agnes == i, ]
    cluster_center <- colMeans(cluster_points)
    cluster_size <- sum(clusters_agnes == i)
    current_wss <- current_wss + sum((cluster_points - cluster_center)^2)
  }
  
  wss[k] <- current_wss
}

# Draw the elbow plot
plot(1:20, wss, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of Clusters", ylab = "Within-Cluster Sum of Squares",
     main = "Elbow Plot for Agglomerative Clustering")

# This code calculates the WSS for the first 20 clusters by iteratively cutting the dendrogram and computing the sum of squares for each cluster configuration. The elbow plot helps you visualize the point where the decrease in WSS starts to slow down, indicating a potential optimal number of clusters.

# Identify the "elbow point" visually from the plot

# The elbow point is at 5 clusters

num_clusters <- 5
clusters_agnes <- cutree(agnes_model, k = num_clusters)

# Print the cluster assignments for each data point

print(clusters_agnes)

plot(clusters_agnes, main = 'Agglomerative clustering Dendogram with 5 clusters', xlab = 'songs')


# Performance evaluation

# Initialize variables for WCSS and BCSS
total_agnes_wcss <- 0
total_agnes_bcss <- 0

# Calculate the centroid of the entire dataset

total_centroid <- colMeans(songs)

# Loop through each cluster

for (i in 1:5) {
  cluster_agnes_points <- songs[clusters_agnes == i, ]
  cluster_agnes_center <- colMeans(cluster_agnes_points)
  cluster_agnes_size <- nrow(cluster_agnes_points)
  
  # Calculate WCSS for the cluster
  total_agnes_wcss <- total_agnes_wcss + sum((cluster_agnes_points - cluster_agnes_center)^2)
  
  # Calculate BCSS for the cluster
  bcss_agnes_contribution <- sum(((cluster_agnes_center - total_centroid)^2) * cluster_agnes_size)
  total_agnes_bcss <- total_agnes_bcss + bcss_agnes_contribution
}

# Print the results
print(paste("Total WCSS:", total_agnes_wcss))
print(paste("Total BCSS:", total_agnes_bcss))

# Calculate individual wcss and bcss

# Calculate WCSS and BCSS for each cluster
cluster_agnes_wcss <- numeric(5)
cluster_agnes_bcss <- numeric(5)

for (i in 1:5) {
  cluster_agnes_points <- songs[clusters_agnes == i, ]
  cluster_agnes_center <- colMeans(cluster_agnes_points)
  cluster_agnes_wcss[i] <- sum((cluster_agnes_points - cluster_agnes_center)^2)
  cluster_agnes_bcss[i] <- sum((cluster_agnes_center - colMeans(songs))^2) * nrow(cluster_agnes_points)
}

print("Individual Cluster WCSS:")
print(cluster_agnes_wcss)

print("Individual Cluster BCSS:")
print(cluster_agnes_bcss)

# Calculate the BCSS to tot. SS ratio

bcss_to_totSS = (total_agnes_bcss/(total_agnes_wcss+total_agnes_bcss))*100

print("The BCSS to total. SS of the agnes model is:\n")
print(bcss_to_totSS)


#### Classification Model Building

# Define feature matrix and Target Vector

X = songs[1:11]
y <- ifelse(songs$target < 0, 0, 1)
y

# Split the data into train and test sets

# Set the seed for reproducibility
set.seed(6840)

# Split the data into 80% training and 20% testing
splitIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[splitIndex, ]
X_test <- X[-splitIndex, ]
y_train <- y[splitIndex]
y_test <- y[-splitIndex]

# Checking for class imbalance

sum(y_train == 1)
sum(y_train == 0)

#### Classification Model Building

# Logistic Regression Model Building

# Combine the response variable and predictor matrix into a single data frame
data_train <- cbind(y_train, X_train)

# Create a logistic regression model using stepwise selection
stepwise_model <- stepAIC(glm(y_train ~ ., data = data_train, family = binomial), direction = "both")

# Print the summary of the selected model
summary(stepwise_model)

# Predict the values for X_test
data_test <- data.frame(X_test)  # Create a data frame for X_test
y_log_pred <- predict(stepwise_model, newdata = data_test, type = "response")

# Convert probabilities to class labels (0 or 1)
y_log_pred_class <- ifelse(y_log_pred > 0.5, 1, 0)

# Print predicted classes
print(y_log_pred_class)


# Create a confusion matrix
confusion_mat <- confusionMatrix(factor(y_log_pred_class), factor(y_test))
confusion_mat

# Calculate accuracy
accuracy <- confusion_mat$overall["Accuracy"]

# Calculate precision, recall, and F1-score
precision <- confusion_mat$byClass["Pos Pred Value"]
recall <- confusion_mat$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the metrics
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))
print("Confusion Matrix:")
print(confusion_mat$table)



#### Random Forest Model Building

# Random Forests inherently perform feature selection 
# by considering subsets of features during the construction of 
# individual decision trees and then combining their predictions.

# Create a Random Forest model

rf_model <- randomForest(as.factor(y_train) ~ ., data = as.data.frame(X_train), ntree = 500)

# Predict the values for X_test
y_rf_pred <- predict(rf_model, newdata = as.data.frame(X_test))

# Create a confusion matrix
confusion_mat <- confusionMatrix(factor(y_rf_pred), factor(y_test))

# Calculate accuracy
accuracy <- confusion_mat$overall["Accuracy"]

# Calculate precision, recall, and F1-score
precision <- confusion_mat$byClass["Pos Pred Value"]
recall <- confusion_mat$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the metrics
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))
print("Confusion Matrix:")
print(confusion_mat$table)
