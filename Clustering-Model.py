## IMPORTING LIBRARIES ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

### PREPROCESSING FULL DATASET ###

# Importing the dataset
dataset = pd.read_excel('Kickstarter.xlsx') # Change this to your local path
df = dataset.copy()

# Do not consider projects in interim state
df = df[df['state'].isin(['failed', 'successful'])]

# compute new variable goal_usd by applying static_usd_rate to goal in local currency
df['goal_usd'] = df['goal'] * df['static_usd_rate']
# round goal_usd to nearest integer
df['goal_usd'] = df['goal_usd'].round(0)
# drop goal and static_usd_rate from original dataframe
df.drop(['goal', 'static_usd_rate'], axis=1, inplace=True)
# round usd_pledged to nearest integer
df['usd_pledged'] = df['usd_pledged'].round(0)
# impute "Uncategorized" for missing values in category
df['category'].fillna('Uncategorized', inplace=True)
# drop highly correlated variables (name_len, blurb_len, and launched_at_yr) from original dataframe as determined during EDA
df.drop(['name_len', 'blurb_len', 'launched_at_yr'], axis=1, inplace=True)
# drop ID variables from dataframe
df.drop(['id', 'name'], axis=1, inplace=True)

# Use modified IQR method to identify extreme outliers in goal_usd
iqr_value = iqr(df['goal_usd'])
# calculate upper and lower bounds
lower = df['goal_usd'].quantile(0.25) - (2.5 * iqr_value)
upper = df['goal_usd'].quantile(0.75) + (2.5 * iqr_value)
# df[(df['goal_usd'] < lower) | (df['goal_usd'] > upper)].shape[0] / df.shape[0] # uncomment to check percentage of outliers 
# store outliers in df called ol_goal_usd, retaining their orignal index values
ol_goal_usd = df[(df['goal_usd'] < lower) | (df['goal_usd'] > upper)]
# drop outliers
df.drop(ol_goal_usd.index, axis=0, inplace=True)


# create data-mapping dictionary for categories based on domain knowledge
u_categories = {
    'Gadgets': 'Tech_Hardware',
    'Uncategorized': 'Other',
    'Experimental': 'Arts',
    'Plays': 'Arts',
    'Spaces': 'Other',
    'Web': 'Tech_Software',
    'Apps': 'Tech_Software',
    'Wearables': 'Tech_Hardware',
    'Software': 'Tech_Software',
    'Festivals': 'Arts',
    'Hardware': 'Tech_Hardware',
    'Robots': 'Tech_Hardware',
    'Makerspaces': 'Arts',
    'Musical': 'Arts',
    'Immersive': 'Arts',
    'Flight': 'Tech_Hardware',
    'Sound': 'Tech_Hardware',
    'Academic': 'Other',
    'Places': 'Other',
    'Thrillers': 'Arts',
    'Webseries': 'Arts',
    'Blues': 'Arts',
    'Shorts': 'Arts'
}

# create data-mapping dictionary for regions based on observed freuqencies during prior EDA
regions = {
    'DE': 'Minor_Region',
    'FR': 'Minor_Region',
    'BE': 'Minor_Region',
    'IT': 'Minor_Region',
    'SE': 'Minor_Region',
    'IE': 'Minor_Region',
    'DK': 'Minor_Region',
    'ES': 'Minor_Region',
    'NL': 'Minor_Region',
    'CH': 'Minor_Region',
    'AT': 'Minor_Region',
    'LU': 'Minor_Region',
    'NO': 'Minor_Region',
    'NZ': 'Minor_Region',
    'CA': 'Major_NonUS',
    'GB': 'Major_NonUS',
    'AU': 'Major_NonUS',
    'US': 'US'
}

### PREPROCESSING MODEL SUBSET ###

features = df.copy()

# features worth considering for clustering determined during prior EDA
# select usd_goal, state, country, staff_pick, backers_count, usd_pledged, category, spotlight, name_len_clean, blurb_len_clean, launch_to_deadline_days, launch_to_state_change_days 
features = features[['goal_usd', 'state', 'country', 'staff_pick', 'backers_count', 'usd_pledged', 'category', 'spotlight', 'name_len_clean', 'blurb_len_clean', 'launch_to_deadline_days', 'launch_to_state_change_days']]

# mapping
features['category'] = features['category'].map(u_categories)
features['region'] = features['country'].map(regions)

# drop country
features.drop('country', axis=1, inplace=True)

# convert spotlight and staff_pick to object data type
features['spotlight'] = features['spotlight'].astype('object')
features['staff_pick'] = features['staff_pick'].astype('object')

# move position of columns in num_cols to the front of the dataframe
num_cols = ['goal_usd', 'backers_count', 'usd_pledged', 'name_len_clean', 'blurb_len_clean','launch_to_deadline_days','launch_to_state_change_days' ]
cat_cols = [col for col in features.columns if col not in num_cols]

# Combine num_cols and other_cols to create the new column order
new_col_order = num_cols + cat_cols

# Reorder the columns in the DataFrame
features= features[new_col_order]

# Standardize only numerical features
f_numerical = features.iloc[:,:7]
f_categorical = features.iloc[:,7:]
scaler = StandardScaler()
f_num_std = scaler.fit_transform(f_numerical)

# Convert the standardized features into a DataFrame then merge according to original index
f_num_std_df = pd.DataFrame(data = f_num_std, columns = f_numerical.columns, index = f_numerical.index)
f_std = pd.merge(f_num_std_df, f_categorical, left_index=True, right_index=True)


### BUILDING THE MODEL ###

# # Uncomment for hyperparemeter tuning for optimal number of clusters
# costs = []
# for i in range(4,11):
#     kmixed = KPrototypes(n_clusters=i, random_state = 0)
#     kmixed.fit_predict(f_std, categorical=[7, 8, 9, 10, 11])
#     print(f"Cost for {i} clusters: {kmixed.cost_}")
#     costs.append(kmixed.cost_)


# plt.plot(range(4,11), costs, marker='o')  
# plt.xlabel('Number of clusters')
# plt.ylabel('Cost')
# plt.title('Cost vs Number of Clusters')  
# plt.grid(True)  
# plt.xticks(range(4, 11)) # Ensure x-axis ticks are for each cluster number
# plt.show()

kmixed_final = KPrototypes(n_clusters=7, random_state = 0) # 7 clusters chosen based on elbow method
cluster = kmixed_final.fit_predict(f_std, categorical=[7, 8, 9, 10, 11])
# kmixed_final.gamma # uncomment to display gamma value
# kmixed_final.cost_ #  uncomment to display cost value
final_centroids = pd.DataFrame(kmixed_final.cluster_centroids_, columns=f_std.columns)
# final_centroids.to_excel("final_centroids2.xlsx") # Uncomment to export final centroids to excel for later interpretation



### DERIVING INSIGHTS FROM THE MODEL ###
f_clustered = features.copy()
f_clustered['cluster'] = cluster # creating a column with cluster labels
cluster_dfs = {}  # Dictionary to store each cluster's DataFrame

# Get the unique cluster labels
unique_clusters = f_clustered['cluster'].unique()

# Create a DataFrame for each unique cluster label
for cluster_label in unique_clusters:
    cluster_dfs[f'cluster_{cluster_label}'] = f_clustered[f_clustered['cluster'] == cluster_label]


# # Uncomment to export descriptive statistics to Excel files for each cluster
# for cluster_label, cluster_df in cluster_dfs.items():
#     # Get the descriptive statistics for the numeric columns
#     stats = cluster_df[num_cols].describe()
    
#     # Define the Excel file path
#     file_path = f"cluster_{cluster_label}_descriptive_stats.xlsx"
    
#     # Write the statistics to an Excel file
#     with pd.ExcelWriter(file_path) as writer:
#         stats.to_excel(writer, sheet_name=f"Cluster_{cluster_label}_Stats")

# Cross-tabulation for Categoricals 
crosstab_dfs = {}
for col in cat_cols:
    # Create a crosstab: counts of each unique value in 'col', broken down by cluster
    crosstab_df = pd.crosstab(f_clustered[col], f_clustered['cluster'], margins = True)
    
    # Store in a dictionary
    crosstab_dfs[col] = crosstab_df


# # Uncomment to create a Pandas Excel writer for the crosstab DataFrames
# with pd.ExcelWriter('crosstabs.xlsx', engine='openpyxl') as writer:
#     for col, crosstab_df in crosstab_dfs.items():
#         # Write each DataFrame to a different worksheet
#         crosstab_df.to_excel(writer, sheet_name=col)
