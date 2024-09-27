import pandas as pd
players20 = pd.read_excel('players_final_22_1.xlsx')
#players201 = pd.read_excel('players_final_21_1.xlsx')
corr20 = players20.corr(numeric_only=True)
#corr20_scaled = scaled_features.corr(numeric_only=True)
players20 = players20[players20['ACTUAL_MINUTES']>500]

players20.columns

avg_features = [ 'ACTUAL_MINUTES','DISPLAY_FI_LAST','GP','FG_PCT','FG3_PCT',  'FT_PCT', 'AVG_TOT_REB',
         'AVG_AST', 'AVG_STL',
       'AVG_TURNOVERS', 'AVG_BLK', 'AVG_PTS',  
       
       'percentagePointsMidrange2pt',
       'percentagePointsFastBreak',
       'percentagePointsFreeThrow', 'percentagePointsOffTurnovers',
       'percentagePointsPaint', 'percentageAssisted2pt',
       'percentageUnassisted2pt', 'percentageAssisted3pt',
       'percentageUnassisted3pt',  
       'playerPoints',  
        'matchupFieldGoalPercentage','PIE', 'PER'] #,'OWS','DWS','OBPM', 'DBPM']
corr20 = players20[avg_features].corr(numeric_only=True)


corr20 = players20[avg_features].corr(numeric_only=True)
players20 = players20[avg_features]
###############
#import seaborn as sns
#sns.kdeplot(data=players20, x='GP')
#sns.kdeplot(data=players20, x='ACTUAL_MINUTES')
##sns.kdeplot(data=players20, x='DISPLAY_FI_LAST')
#sns.kdeplot(data=players20, x='FG3_PCT')
#sns.kdeplot(data=players20, x='FG_PCT')
#sns.kdeplot(data=players20, x='FT_PCT')
#sns.kdeplot(data=players20, x='AVG_TOT_REB')
#sns.kdeplot(data=players20, x='AVG_AST')
#sns.kdeplot(data=players20, x='AVG_STL')
#sns.kdeplot(data=players20, x='AVG_TURNOVERS')
#sns.kdeplot(data=players20, x='AVG_BLK')
#sns.kdeplot(data=players20, x='percentagePointsMidrange2pt')
#sns.kdeplot(data=players20, x='percentagePointsFastBreak')
#sns.kdeplot(data=players20, x='percentagePointsFreeThrow')
#sns.kdeplot(data=players20, x='percentagePointsOffTurnovers')
#sns.kdeplot(data=players20, x='percentagePointsPaint')
#sns.kdeplot(data=players20, x='percentageAssisted2pt')
#sns.kdeplot(data=players20, x='percentageUnassisted2pt')
#sns.kdeplot(data=players20, x='percentageAssisted3pt')
#sns.kdeplot(data=players20, x='percentageUnassisted3pt')
#sns.kdeplot(data=players20, x='percentageAssisted2pt')
#sns.kdeplot(data=players20, x='playerPoints')
#sns.kdeplot(data=players20, x='matchupFieldGoalPercentage')
#sns.kdeplot(data=players20, x='PIE')
#sns.kdeplot(data=players20, x='PER')
##########

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'players20' is your DataFrame

# Specify the list of columns you want to plot
columns_to_plot = ['GP', 'ACTUAL_MINUTES',  'FG3_PCT', 'FG_PCT',
                   'FT_PCT', 'AVG_TOT_REB', 'AVG_AST', 'AVG_STL', 'AVG_TURNOVERS',
                   'AVG_BLK', 'AVG_PTS', 'percentagePointsMidrange2pt',
                   'percentagePointsFastBreak', 'percentagePointsFreeThrow',
                   'percentagePointsOffTurnovers', 'percentagePointsPaint',
                   'percentageAssisted2pt', 'percentageUnassisted2pt',
                   'percentageAssisted3pt', 'percentageUnassisted3pt',
                   'playerPoints', 'matchupFieldGoalPercentage', 'PIE', 'PER']


# Calculate the number of rows and columns for subplots
num_rows = 4
num_cols = 6

# Create a figure and axes for subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))

# Flatten axes if num_rows > 1
axes = axes.flatten()

# Iterate through each column in your list
for i, column in enumerate(columns_to_plot):
    # Create a KDE plot for each variable using kdeplot
    sns.kdeplot(data=players20, x=column, ax=axes[i])
    axes[i].set_title(f'KDE Plot of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Density')

# Remove any unused subplots
for j in range(len(columns_to_plot), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()




players20 = players20[players20['GP']>25]

# Perform standard scaling on the selected features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features = [ 'ACTUAL_MINUTES','FG_PCT','FG3_PCT',  'FT_PCT', 'AVG_TOT_REB',
         'AVG_AST', 'AVG_STL',
       'AVG_TURNOVERS', 'AVG_BLK', 'AVG_PTS',  
       
       'percentagePointsMidrange2pt',
       'percentagePointsFastBreak',
       'percentagePointsFreeThrow', 'percentagePointsOffTurnovers',
       'percentagePointsPaint', 'percentageAssisted2pt',
       'percentageUnassisted2pt', 'percentageAssisted3pt',
       'percentageUnassisted3pt',  
       'playerPoints', 
        'matchupFieldGoalPercentage','PIE', 'PER']#,'OWS','DWS','OBPM', 'DBPM']
players20[features] = scaler.fit_transform(players20[features])

scaled_features = players20[features]


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score

silhouette_scores = []
k_values = range(2, 15)  # You can adjust the range as needed

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, init="random")
    cluster_labels = kmeans.fit_predict(scaled_features)
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()



#elbow methof

inertia_values = []
k_values = range(1, 15)  # You can adjust the range as needed

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia_values.append(kmeans.inertia_)

plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()





k =12



# Perform KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=1,init='random')
players20['cluster'] = kmeans.fit_predict(scaled_features)




cluster_centers = kmeans.cluster_centers_
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
print(cluster_centers_df)

import matplotlib.pyplot as plt
import seaborn as sns

# Define a custom color palette with distinct colors for each cluster
# You can choose any color scheme you prefer
custom_palette = sns.color_palette('hsv', n_colors=k)

# Plot clusters using two selected features (you can choose different features)
sns.scatterplot(x='AVG_TOT_REB', y='AVG_PTS', hue='cluster', data=players20, palette=custom_palette)
plt.title('KMeans Clustering2022')
plt.xlabel('Field Goal Percentage')
plt.ylabel('Points')
plt.show()

cluster_players = players20[['DISPLAY_FI_LAST', 'cluster']]


