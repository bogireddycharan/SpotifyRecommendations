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
