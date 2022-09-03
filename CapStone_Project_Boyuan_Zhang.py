# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 23:37:24 2022

@author: Boyuan Zhang
"""

# 0 Init
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from scipy.special import expit # this is the logistic sigmoid function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# 1 Load data
data= np.genfromtxt('movieReplicationSet_Capstone.csv',delimiter = ',')
data = data[1:,:]
df = pd.read_csv('movieReplicationSet_Capstone.csv',delimiter = ',')

data_movie = data[:,:400]
data_sensation = data[:,400:420]
data_personality = data[:,420:464]
data_experience = data[:,464:474]
data_gender = data[:,474]
data_child = data[:,475]
data_social = data[:,476]


#################################################################################################################################
####### 1) What is the relationship between sensation seeking and movie experience?

# Clean nans
### Q1
data_Q1 = np.column_stack((data_sensation,data_experience))
data_Q1 = data_Q1[~np.isnan(data_Q1).any(axis=1)]
data_sensation_c = data_Q1[:,0:20]
data_experience_c = data_Q1[:,20:30]

### display Corr
corrMatrix_sensation = np.corrcoef(data_sensation_c,rowvar=False) #Compute the correlation matrix

# Plot the data:
plt.imshow(corrMatrix_sensation) 
plt.xlabel('Question')
plt.ylabel('Question')
plt.colorbar()
plt.show()


##### 2 PCA

### PCA sensation
zscored_sensation = stats.zscore(data_sensation_c)
pca_sensation = PCA().fit(zscored_sensation)

eigVals_sensation = pca_sensation.explained_variance_
loadings_sensation = pca_sensation.components_*-1
rotatedData_sensation = pca_sensation.fit_transform(zscored_sensation)*-1
varExplained_sensation = eigVals_sensation/sum(eigVals_sensation)*100
### display
for ii in range(len(varExplained_sensation)):
    print(varExplained_sensation[ii].round(3))
    

### scree plot    
numFactors= len(eigVals_sensation)
x = np.linspace(1,numFactors,numFactors)
plt.bar(x, eigVals_sensation, color='gray')
plt.plot([0,numFactors],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component sensation')
plt.ylabel('Eigenvalue')
plt.show()

### check number of factors
kaiserThreshold = 1
threshold = 90
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals_sensation > kaiserThreshold))
print('Number of factors selected by elbow criterion: 3') #Due to visual inspection by primate
eigSum = np.cumsum(varExplained_sensation)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)
# Choose 6 factors by Kaiser criterion

### visualize the factors 
whichPrincipalComponent = 0 # Select and look at one factor at a time 
plt.bar(x,loadings_sensation[whichPrincipalComponent,:]) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show() # Show bar plot

### Store our transformed data
x_sensation = np.column_stack((rotatedData_sensation[:,0],rotatedData_sensation[:,1],rotatedData_sensation[:,2],rotatedData_sensation[:,3],rotatedData_sensation[:,4],rotatedData_sensation[:,5]))


### PCA experience
zscored_experience = stats.zscore(data_experience_c)
pca_experience = PCA().fit(zscored_experience)

eigVals_experience = pca_experience.explained_variance_
loadings_experience = pca_experience.components_*-1
rotatedData_experience = pca_experience.fit_transform(zscored_experience)*-1
varExplained_experience = eigVals_experience/sum(eigVals_experience)*100
# display
for ii in range(len(varExplained_experience)):
    print(varExplained_experience[ii].round(3))


### scree plot    
numFactors= len(eigVals_experience)
x = np.linspace(1,numFactors,numFactors)
plt.bar(x, eigVals_experience, color='gray')
plt.plot([0,numFactors],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component experience')
plt.ylabel('Eigenvalue')
plt.show()

### check number of factors
kaiserThreshold = 1
threshold = 90
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals_experience > kaiserThreshold))
print('Number of factors selected by elbow criterion: 2') #Due to visual inspection by primate
eigSum = np.cumsum(varExplained_experience)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)
# Choose 2 factors by both Kaiser criterion and elbow criterion

### visualize the factors 
whichPrincipalComponent = 1 # Select and look at one factor at a time 
plt.bar(x,loadings_experience[whichPrincipalComponent,:]) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show() # Show bar plot

### Store our transformed data
x_experience = np.column_stack((rotatedData_experience[:,0],rotatedData_experience[:,1]))

###### Correlation between sensation and experience
#correlation_Q1 = pd.Series(rotatedData_sensation[:,0]).corr(pd.Series(rotatedData_experience[:,0]), method = 'pearson')
varCorrs_Q1 = np.corrcoef(np.transpose(x_sensation),np.transpose(x_experience))

### draw the correlation map
plt.figure(figsize=(10, 10)) # set the size of the correaltion map
plt.imshow(varCorrs_Q1)
plt.title('Sensation vs Experience Correlation Correlation Map',fontdict={'fontsize':15}, pad=20) 
plt.colorbar()
plt.show()

# From the correlation map, we can visualize that there is no strong correlation between the 6 original data new coordinates 
# in sensation we accounted with the 2 original data new coordinates in the experience we accounted.
# Thus, we conclude that there is no strong relationship between sensation seeking and movie experience.



#################################################################################################################################
####### 2) Is there evidence of personality types based on the data of these research participants? 
#######    If so, characterize these types both quantitatively and narratively. 

### Clean data
data_personality_c = data_personality[~np.isnan(data_personality).any(axis=1)]

### display Corr
corrMatrix_personality = np.corrcoef(data_personality_c,rowvar=False) #Compute the correlation matrix

# Plot the data:
plt.imshow(corrMatrix_personality) 
plt.xlabel('Personality')
plt.ylabel('Personality')
plt.colorbar()
plt.show()

### PCA
zscored_personality = stats.zscore(data_personality_c)
pca_personality = PCA().fit(zscored_personality)

eigVals_personality = pca_personality.explained_variance_
loadings_personality = pca_personality.components_*-1
rotatedData_personality = pca_personality.fit_transform(zscored_personality)*-1
varExplained_personality = eigVals_personality/sum(eigVals_personality)*100

### display
for ii in range(len(varExplained_personality)):
    print(varExplained_personality[ii].round(3))
    
### scree plot    
numFactors= len(eigVals_personality)
x = np.linspace(1,numFactors,numFactors)
plt.bar(x, eigVals_personality, color='gray')
plt.plot([0,numFactors],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component personality')
plt.ylabel('Eigenvalue')
plt.show()

### visualize the factors 
whichPrincipalComponent = 0 # Select and look at one factor at a time 
plt.bar(x,loadings_personality[whichPrincipalComponent,:]) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Personality')
plt.ylabel('Loading')
plt.show() # Show bar plot

### check number of factors
kaiserThreshold = 1
threshold = 90
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals_personality > kaiserThreshold))
print('Number of factors selected by elbow criterion: 6') #Due to visual inspection by primate
eigSum = np.cumsum(varExplained_personality)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)
# Choose 6 factors by elbow criterion

### Store our transformed data
x_personality = rotatedData_personality[:,:6]

### Silhouette: Find number of cluster

numCluster = 9
sSum = np.empty([numCluster,1])*np.NaN # init container to store sums
# Compute kMeans for each k:
for ii in range(2, numCluster+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii)).fit(x_personality) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x_personality,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot 

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,numCluster,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()
# Number of clusters best fit 2


###### kMeans:
numPersonality_fit = 2
kMeans = KMeans(n_clusters = numPersonality_fit).fit(x_personality) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 

# Plot the color-coded data:
for ii in range(numPersonality_fit):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x_personality[plotIndex,0],x_personality[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('Extroversion Level')
    plt.ylabel('Spirit Strength')
    
    

#################################################################################################################################
####### 3) Are movies that are more popular rated higher than movies that are less popular?

### Operationalize the popularity of a movie by how many ratings it has received
numRatings = np.empty([400,1])
for ii in range(400):
    data_movie_ii = data_movie[:,ii]
    data_movie_ii_c = data_movie_ii[~np.isnan(data_movie_ii)]
    numRatings[ii] = len(data_movie_ii_c)

### Gather the mean of ratings of each movie
movieRatings = np.empty([400,1])
for ii in range(400):
    data_movie_ii = data_movie[:,ii]
    data_movie_ii_c = data_movie_ii[~np.isnan(data_movie_ii)]
    movieRatings[ii] = np.mean(data_movie_ii_c)
    
### Plot a graph to visualize the relationship between the popularity of movies and the mean ratings of movies
plt.plot(numRatings, movieRatings,'bo')
plt.xlabel('Movie Popularity')
plt.ylabel('Movie Rating')
plt.title('Popularity vs Rating')


#################################################################################################################################
####### 4) Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?

### Load and Clean data
data_shrek = df['Shrek (2001)'].to_numpy()
data_Q4 = np.stack((data_shrek,data_gender), axis = 1)
data_Q4_c = data_Q4[~np.isnan(data_Q4).any(axis = 1)]
data_Q4_female = data_Q4_c[np.where(data_Q4_c[:,1] == 1)]
data_Q4_male = data_Q4_c[np.where(data_Q4_c[:,1] == 2)]

### Mann-Whitney U test
u_Q4, p_Q4 = stats.mannwhitneyu(data_Q4_female[:,0],data_Q4_male[:,0])
print(p_Q4)



#################################################################################################################################
####### 5) Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings? 

### Load and Clean data
data_lion = df['The Lion King (1994)'].to_numpy()
data_Q5 = np.stack((data_lion,data_child), axis = 1)
data_Q5_c = data_Q5[~np.isnan(data_Q5).any(axis = 1)]
data_Q5_only = data_Q5_c[np.where(data_Q5_c[:,1]==1)]
data_Q5_sibling = data_Q5_c[np.where(data_Q5_c[:,1]==0)]

### Mann-Whitney U test
u_Q5, p_Q5 = stats.mannwhitneyu(data_Q5_only[:,0],data_Q5_sibling[:,0])
print(p_Q5)


    
#################################################################################################################################
####### 6) Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who prefer to watch them alone? 
  
### Load and Clean data
data_wallStreet = df['The Wolf of Wall Street (2013)'].to_numpy()
data_Q6 = np.stack((data_wallStreet,data_social), axis = 1)
data_Q6_c = data_Q6[~np.isnan(data_Q6).any(axis = 1)]
data_Q6_alone = data_Q6_c[np.where(data_Q6_c[:,1]==1)]
data_Q6_notalone = data_Q6_c[np.where(data_Q6_c[:,1]==0)]

### Mann-Whitney U test
u_Q6, p_Q6 = stats.mannwhitneyu(data_Q6_alone[:,0],data_Q6_notalone[:,0])
print(p_Q6)


    
#################################################################################################################################
####### 7) There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, 
#######    ‘Indiana Jones’, ‘Jurassic Park’, ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this 
#######    dataset. How many of these are of inconsistent quality, as experienced by viewers?

### Star Wars

### Load and Clean data
df_Starwars = df.iloc[:,df.columns.str.contains('Star Wars')]
df_Starwars_c = df_Starwars.dropna()
data_Starwars = df_Starwars_c.to_numpy()

# Kruskal-Wallis test: Nonparametric test for more than 2 groups

h_Starwars,pK_Starwars = stats.kruskal(data_Starwars[:,0],data_Starwars[:,1],data_Starwars[:,2],data_Starwars[:,3],data_Starwars[:,4],data_Starwars[:,5])
print(pK_Starwars)

### Harry Potter

### Load and Clean data
df_Harry = df.iloc[:,df.columns.str.contains('Harry Potter')]
df_Harry_c = df_Harry.dropna()
data_Harry = df_Harry_c.to_numpy()

# Kruskal-Wallis test: Nonparametric test for more than 2 groups

h_Harry,pK_Harry = stats.kruskal(data_Harry[:,0],data_Harry[:,1],data_Harry[:,2],data_Harry[:,3])
print(pK_Harry)

### The Matrix

### Load and Clean data
df_Matrix = df.iloc[:,df.columns.str.contains('The Matrix')]
df_Matrix_c = df_Matrix.dropna()
data_Matrix = df_Matrix_c.to_numpy()

# Kruskal-Wallis test: Nonparametric test for more than 2 groups

h_Matrix,pK_Matrix = stats.kruskal(data_Matrix[:,0],data_Matrix[:,1],data_Matrix[:,2])
print(pK_Matrix)

### Indiana Jones

### Load and Clean data
df_Indiana = df.iloc[:,df.columns.str.contains('Indiana Jones')]
df_Indiana_c = df_Indiana.dropna()
data_Indiana = df_Indiana_c.to_numpy()

# Kruskal-Wallis test: Nonparametric test for more than 2 groups

h_Indiana,pK_Indiana = stats.kruskal(data_Indiana[:,0],data_Indiana[:,1],data_Indiana[:,2],data_Indiana[:,3])
print(pK_Indiana)

### Jurassic Park

### Load and Clean data
df_Jurassic = df.iloc[:,df.columns.str.contains('Jurassic Par')]
df_Jurassic_c = df_Jurassic.dropna()
data_Jurassic = df_Jurassic_c.to_numpy()

# Kruskal-Wallis test: Nonparametric test for more than 2 groups

h_Jurassic,pK_Jurassic = stats.kruskal(data_Jurassic[:,0],data_Jurassic[:,1],data_Jurassic[:,2])
print(pK_Jurassic)

### Pirates of the Caribbean

### Load and Clean data
df_Pirates = df.iloc[:,df.columns.str.contains('Pirates of the Caribbean')]
df_Pirates_c = df_Pirates.dropna()
data_Pirates = df_Pirates_c.to_numpy()

# Kruskal-Wallis test: Nonparametric test for more than 2 groups

h_Pirates,pK_Pirates = stats.kruskal(data_Pirates[:,0],data_Pirates[:,1],data_Pirates[:,2])
print(pK_Pirates)

### Toy Story

### Load and Clean data
df_Toy = df.iloc[:,df.columns.str.contains('Toy Story')]
df_Toy_c = df_Toy.dropna()
data_Toy = df_Toy_c.to_numpy()

# Kruskal-Wallis test: Nonparametric test for more than 2 groups

h_Toy,pK_Toy= stats.kruskal(data_Toy[:,0],data_Toy[:,1],data_Toy[:,2])
print(pK_Toy)

### Batman

### Load and Clean data
df_Batman = df.iloc[:,df.columns.str.contains('Batman')]
df_Batman_c = df_Batman.dropna()
data_Batman = df_Batman_c.to_numpy()

# Kruskal-Wallis test: Nonparametric test for more than 2 groups

h_Batman,pK_Batman = stats.kruskal(data_Batman[:,0],data_Batman[:,1],data_Batman[:,2])
print(pK_Batman)
























#################################################################################################################################
####### 8) Build a prediction model of your choice (regression or supervised learning) to predict movie 
####### ratings (for all 400 movies) from personality factors only. Make sure to use cross-validation 
####### methods to avoid overfitting and characterize the accuracy of your model.

### Load and clean data

data_Q8_personality = data_personality # From Q2
# fill nan with mean value
for ii in range(44):
    data_Q8_personality[:,ii] = np.nan_to_num(data_Q8_personality[:,ii], nan = np.nanmean(data_Q8_personality[:,ii]))
    
data_Q8_movie = data_movie # from early loading
# fill nan with mean value
for ii in range(400):
    data_Q8_movie[:,ii] = np.nan_to_num(data_Q8_movie[:,ii], nan = np.nanmean(data_Q8_movie[:,ii]))

### PCA Personality 
# Already PCA Personality at Question 2
# Copy code from question 2
zscored_personality = stats.zscore(data_Q8_personality)
pca_personality = PCA().fit(zscored_personality)

eigVals_personality = pca_personality.explained_variance_
loadings_personality = pca_personality.components_*-1
rotatedData_personality = pca_personality.fit_transform(zscored_personality)*-1
varExplained_personality = eigVals_personality/sum(eigVals_personality)*100

### display
for ii in range(len(varExplained_personality)):
    print(varExplained_personality[ii].round(3))
    
### scree plot    
numFactors= len(eigVals_personality)
x = np.linspace(1,numFactors,numFactors)
plt.bar(x, eigVals_personality, color='gray')
plt.plot([0,numFactors],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component personality')
plt.ylabel('Eigenvalue')
plt.show()

### visualize the factors 
whichPrincipalComponent = 0 # Select and look at one factor at a time 
plt.bar(x,loadings_personality[whichPrincipalComponent,:]) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Personality')
plt.ylabel('Loading')
plt.show() # Show bar plot

### check number of factors
kaiserThreshold = 1
threshold = 90
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals_personality > kaiserThreshold))
print('Number of factors selected by elbow criterion: 6') #Due to visual inspection by primate
eigSum = np.cumsum(varExplained_personality)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)
# Choose 6 factors by elbow criterion

### Store our transformed data
x_personality = rotatedData_personality[:,:6]

# Split data into training set and testing set to do cross validation
x_train_Q8,x_test_Q8,y_train_Q8,y_test_Q8 = train_test_split(x_personality,data_Q8_movie,test_size = 0.25)
### Linear Regression
Model_Q8 = LinearRegression().fit(x_train_Q8,y_train_Q8)
yHat_Q8 = Model_Q8.predict(x_test_Q8)

# Assess model accuracy:
modelAccuracy_Q8 = metrics.r2_score(y_test_Q8, yHat_Q8)
modelMSE_Q8 = metrics.mean_squared_error(y_test_Q8, yHat_Q8)
modelRMSE_Q8 = np.sqrt(modelMSE_Q8)
print('Linear Regession model accuracy(R squared COD):', modelAccuracy_Q8)
print('Linear Regession model RMSE:', modelRMSE_Q8)



'''
### Random Forest

# Split data into training set and testing set to do cross validation
x_train_Q8,x_test_Q8,y_train_Q8,y_test_Q8 = train_test_split(x_personality,data_Q8_movie,test_size = 0.25)
#%% Actually doing the random forest
numTrees = 500
clf = RandomForestClassifier(n_estimators=numTrees).fit(x_train_Q8,y_train_Q8) #bagging numTrees trees

# Use model to make predictions:
predictions = clf.predict(x_test_Q8) 

# Assess model accuracy:
modelAccuracy = accuracy_score(y_test_Q8,predictions)
print('Random forest model accuracy:',modelAccuracy)
'''

#################################################################################################################################
####### 9) Build a prediction model of your choice (regression or supervised learning) to predict movie ratings 
#######   (for all 400 movies) from gender identity, sibship status and social viewing preferences (columns 475-477) only. 
#######    Make sure to use cross-validation methods to avoid overfitting and characterize the accuracy of your model. 

### Load and Clean data
data_Q9_movie = data_Q8_movie # Already cleaned movie data in Question 8
data_Q9_gender = np.nan_to_num(data_gender, nan = np.nanmean(data_gender))
data_Q9_sibship = data_child
data_Q9_social = data_social

### stack gender, sibship, social togethor
data_Q9_X = np.stack((data_Q9_gender,data_Q9_sibship,data_Q9_social), axis = 1)

### PCA data_Q9_X
zscored_Q9 = stats.zscore(data_Q9_X)
pca_Q9 = PCA().fit(zscored_Q9)

eigVals_Q9 = pca_Q9.explained_variance_
loadings_Q9 = pca_Q9.components_*-1
rotatedData_Q9 = pca_Q9.fit_transform(zscored_Q9)*-1
varExplained_Q9 = eigVals_Q9/sum(eigVals_Q9)*100

### display
for ii in range(len(varExplained_Q9)):
    print(varExplained_Q9[ii].round(3))
    
### scree plot    
numFactors= len(eigVals_Q9)
x = np.linspace(1,numFactors,numFactors)
plt.bar(x, eigVals_Q9, color='gray')
plt.plot([0,numFactors],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component Q9')
plt.ylabel('Eigenvalue')
plt.show()

### visualize the factors 
whichPrincipalComponent = 0 # Select and look at one factor at a time 
plt.bar(x,loadings_Q9[whichPrincipalComponent,:]) # note: eigVecs multiplied by -1 because the direction is arbitrary
plt.xlabel('Q9')
plt.ylabel('Loading')
plt.show() # Show bar plot

### check number of factors
kaiserThreshold = 1
threshold = 90
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals_Q9 > kaiserThreshold))
print('Number of factors selected by elbow criterion: 3') #Due to visual inspection by primate
eigSum = np.cumsum(varExplained_Q9)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)
# Choose 2 factors by Kaiser

### Store our transformed data
x_Q9 = rotatedData_Q9[:,:2]

# Split data into training set and testing set to do cross validation
x_train_Q9,x_test_Q9,y_train_Q9,y_test_Q9 = train_test_split(x_Q9,data_Q9_movie,test_size = 0.25)
### Linear Regression
Model_Q9 = LinearRegression().fit(x_train_Q9,y_train_Q9)
yHat_Q9 = Model_Q9.predict(x_test_Q9)

# Assess model accuracy:
modelAccuracy_Q9 = metrics.r2_score(y_test_Q9, yHat_Q9)
modelMSE_Q9 = metrics.mean_squared_error(y_test_Q9, yHat_Q9)
modelRMSE_Q9 = np.sqrt(modelMSE_Q9)
print('Linear Regession model accuracy(R squared COD):', modelAccuracy_Q9)
print('Linear Regession model RMSE:', modelRMSE_Q9)




#################################################################################################################################
####### 10) Build a prediction model of your choice (regression or supervised learning) to predict movie ratings 
#######     (for all 400 movies) from all available factors that are not movie ratings (columns 401- 477). 
#######     Make sure to use cross-validation methods to avoid overfitting and characterize the accuracy of your model.

### Load and Clean data
data_Q10_movie = data_Q8_movie # Already cleaned movie data in Question 8

data_Q10_factors = data[:,400:477]
for ii in range(77):
    data_Q10_factors[:,ii] = np.nan_to_num(data_Q10_factors[:,ii], nan = np.nanmean(data_Q10_factors[:,ii]))

### PCA Q10 factors
zscored_Q10 = stats.zscore(data_Q10_factors)
pca_Q10 = PCA().fit(zscored_Q10)

eigVals_Q10 = pca_Q10.explained_variance_
loadings_Q10 = pca_Q10.components_*-1
rotatedData_Q10 = pca_Q10.fit_transform(zscored_Q10)*-1
varExplained_Q10 = eigVals_Q10/sum(eigVals_Q10)*100

### display
for ii in range(len(varExplained_Q10)):
    print(varExplained_Q10[ii].round(3))
    
### scree plot    
numFactors= len(eigVals_Q10)
x = np.linspace(1,numFactors,numFactors)
plt.bar(x, eigVals_Q10, color='gray')
plt.plot([0,numFactors],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component Q10')
plt.ylabel('Eigenvalue')
plt.show()

### visualize the factors 
whichPrincipalComponent = 1 # Select and look at one factor at a time 
plt.bar(x,loadings_Q10[whichPrincipalComponent,:]) 
plt.xlabel('Q10')
plt.ylabel('Loading')
plt.show() # Show bar plot

### check number of factors
kaiserThreshold = 1
threshold = 90
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals_Q10 > kaiserThreshold))
print('Number of factors selected by elbow criterion: 8') #Due to visual inspection by primate
eigSum = np.cumsum(varExplained_Q10)
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)
# Choose 8 factors by Elbow

### Store our transformed data
x_Q10 = rotatedData_Q10[:,:8]

# Split data into training set and testing set to do cross validation
x_train_Q10,x_test_Q10,y_train_Q10,y_test_Q10 = train_test_split(x_Q10,data_Q10_movie,test_size = 0.25)
### Linear Regression
Model_Q10 = LinearRegression().fit(x_train_Q10,y_train_Q10)
yHat_Q10 = Model_Q10.predict(x_test_Q10)

# Assess model accuracy:
modelAccuracy_Q10 = metrics.r2_score(y_test_Q10, yHat_Q10)
modelMSE_Q10 = metrics.mean_squared_error(y_test_Q10, yHat_Q10)
modelRMSE_Q10 = np.sqrt(modelMSE_Q10)
print('Linear Regession model accuracy(R squared COD):', modelAccuracy_Q10)
print('Linear Regession model RMSE:', modelRMSE_Q10)



#################################################################################################################################
######## Extra Credit) Whether the participant with no response in either of the last two columns (Introvert? Shy?) rate movies differently
######## compared to the participants who responded?

### Load and Clean data
data_extra_476_477 = data[:,475:477]
data_extra_movie = data_Q8_movie # from question 8
mean_ratings_extra = np.empty([1097,1]) # mean ratings of every participant on all 400 movies
for ii in range(1097):
    mean_ratings_extra[ii,0] = np.mean(data_extra_movie[ii,:])
# stack them together and split to different groups
data_extra = np.column_stack((mean_ratings_extra,data_extra_476_477))

# participant who at responed to both questions
data_extra_responsed = data_extra[np.where(data_extra[:,1]!=-1) and np.where(data_extra[:,2]!=-1)]
# participant who did not respond to at least one of the two question
data_extra_notresposned = data_extra[np.where(data_extra[:,1]==-1) or np.where(data_extra[:,2]==-1)]
# we visualized that particpants who did not response, usually did not response to both questions

# Two factors, no need to PCA, do a Mann-Whitney U test
u_extra, p_extra = stats.mannwhitneyu(data_extra_responsed[:,0],data_extra_notresposned[:,0])
print(p_extra)




















