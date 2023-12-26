
# Identifying Fundraising Project Profiles through Clustering

In this descriptive analytics project, I've leveraged kprototypes - an algorithm that incorporates both numerical and cateogrical data - to form clusters representing different profiles of projects. 

_Note: This project can be run with just the Clustering-Model.py file and the Kickstarter.xlsx file specified via the path therein._

## Context
[Kickstarter](https://www.kickstarter.com/about) is a crowdfunding platform with a focus on bringing creative projects to life.

As part of a data mining course, I was tasked with defining meaningful profiles from a dataset of over 15,000 projects from 2009 and 2016.


## Process
At a high level, these are the steps I took in developing this model. This repo currently has the code for my final model. I hope to upload some of the behind-the-scenes code while developing that model once I've gotten that cleaned up. 

#### Model Selection
Noting that much of my dataset was comprised of categorical features as well as the fact that many of the characterstics that would be useful in a product strategy context were categorical - I took this as an opportunity to try the versalite kprototypes algorithm

#### Preprocessing
* Manually selected a subset of variables based on my assesmment of their potential in defining meaningful clusters
* Excluded records which had outliers in terms of their fundraising goals
* Binned categorical variables with a high number of unique values based on domain knowledge and/or frequencies observed during EDA
* Standardized numerical variables

#### Hyperparameter Tuning
* Fit the model with various 'k' values and determined via elbow plot that the optimal number of clusters was 7 
* Tested varying gamma values but stuck to default value of 0.5 for balanced consideration of numerical and categorical data for insights, despite heavier weighting on numerical values yielding a lower cost


## Profiles Discovered

#### The Homegrown All-stars (Cluster 3)
Cluster 3, comprising only 0.2% of the dataset, contained exclusively successful, spotlighted projects with high backers and funds raised, representing the platform's all-stars. 

#### The Global Stars (Cluster 5)
Cluster 5, also small in size, mirrored Cluster 3 in success and spotlight but differed with greater regional diversity, including about 15% non-US projects

#### The Dreamers (Cluster 0)
Cluster 0, about 10% of the dataset, was characterized by high fundraising goals but average success rates and backers, indicating projects with ambitions perhaps too high for the platform. 

#### The Marathoners (Cluster 1)
Cluster 1 was notable for longer campaign durations, managing near-average success rates despite lower staff-pick and spotlight rates, suggesting that extended campaigns might have aided in these projects achieving goals.

#### The Microprojects (Cluster 6)
Cluster 6, about 15% of the dataset, was marked by shorter descriptive blurbs and lower goals but near-average success rates, indicating a potential focus on microprojects. 

#### The Everymen (Cluster 2)
Cluster 2, representing about 26% of the data, was balanced across categories with a high share of Arts projects. It showed near-average success, staff-pick, and spotlight rates, reflecting a well-rounded “every-man” cluster.

#### The Everymen+ (Cluster 4)
Cluster 4, holding about 30% of the projects, was similar to Cluster 2 but with higher funding targets and more spotlighted and successful projects, positioning it as a kind of “Every Man+” cluster


## Appendix

#### Elbow Plot

![Elbow Plot](https://github.com/aoluwolerotimi/Kickstarter-Project-Clustering/blob/main/Images/Elbow%20Plot.png)


#### Numeric Centroids
![Numeric Centroids](https://github.com/aoluwolerotimi/Kickstarter-Project-Clustering/blob/main/Images/Numeric%20Centroids.png)


#### Categorical Frequency Counts
![Categorical Frequency Counts](https://github.com/aoluwolerotimi/Kickstarter-Project-Clustering/blob/main/Images/Categorical%20Frequency%20Counts.png)








