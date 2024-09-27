# Cluster_NBA
Clustering NBA players

A big part of my thesis project was clustering NBA Players to identify the main player archetypes in the current state of the game. 

In my project i clustered based on the whole player dataset for season 2022-23 and i proceeded to redo the process by splitting the players in 3 categories based on the team's record that season. 
The split was made to research if the archetypes differ for worse teams. (They did actually)

For the project, I handled and created databases based on extractions I did on NBA_API.

Clustering technique was Kmeans and for the evaluation of the final K, I used elbow method (not really effective), silhouette score (a lot helpful) and my 10 plus years of watching NBA nonstop (pretty effective).

Some of the main conclusions:
-Teams love 3nD players (rightfully so they are integral to an effective playstyle)
-Forward is the most fluid position, since they cannot really be clustered, but rather they fit in clusters with more guard-like players like Khris Middleton and more center/forward positions like Zion.
-Old-style centers are still (Jarrett Allen, Rudy Gobert) really important today on teams that thrive in the Regular season.  
