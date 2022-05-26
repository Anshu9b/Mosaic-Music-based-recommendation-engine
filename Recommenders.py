import numpy as np
import pandas

#Class for Popularity based Recommender System model
class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        #Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
    
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, user_id):    
        user_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations
    

#Class for Item similarity based Recommender System model
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Get unique users for a given item (song)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_songs, all_songs):
            
        ####################################
        #Get users for all songs in user_songs.
        ####################################
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
            
        ###############################################
        #Initialize the item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        #############################################################
        #Calculate similarity between user songs and all unique songs
        #in the training data
        #############################################################
        for i in range(0,len(all_songs)):
            #Calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                    
                #Get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]
                    
                #Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix

    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pandas.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, user):
        
        ########################################
        #A. Get all unique songs for this user
        ########################################
        user_songs = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    #Get similar items to given items
    def get_similar_items(self, item_list):
        
        user_songs = item_list
        
        ######################################################
        #B. Get all unique items (songs) in the training data
        ######################################################
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations
    # Class for KNN Similarity Based Recommender Model
class knn_recommender():
    
    def __init__(self):
        self.train_songs_metadata = None
        self.train_data = None
        self.train_songs_list = None
        self.train_s2u = None
        self.train_u2s = None
        self.train_s2t = None
        self.train_s2i = None
        self.n_neighbors = None
        self.distances = None
        self.indices = None
        
    # Create the KNN based recommender system model
    def create(self, train_songs_metadata, train_util_dict, n_neighbors=10):
        self.train_songs_metadata = train_songs_metadata
        self.train_data = train_util_dict["dataset"]
        self.train_songs_list = train_util_dict["songs"]
        self.train_s2u = train_util_dict["s2u"]
        self.train_u2s = train_util_dict["u2s"]
        self.train_s2t = train_util_dict["s2t"]
        self.train_s2i = train_util_dict["s2i"]
        self.n_neighbors = n_neighbors
        # Apply KNN to get Indices and Distances of n_neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(train_songs_metadata)
        self.distances, self.indices = nbrs.kneighbors(train_songs_metadata)
        
    # Get Similar songs to a given song
    def get_similar_items(self, item, return_type="df"):
    
        # Get song index and then list of indices of similar songs
        item_index = self.train_s2i[item]
        # Drop index of the song itself since it's most similar to itself
        # so not worth recommending / similarity comparison
        similar_songs_indices = self.indices[item_index][1:]
        
        if (return_type == "list"):
        
            # Create Ordered list of Similar Songs
            similar_songs = [self.train_songs_list[i] for i in similar_songs_indices]
            return similar_songs
            
        # By Default Return DataFrame
            
        elif (return_type == "df"):
        
            # Create DataFrame of Similar Songs with their title and scores
            similar_songs = [self.train_songs_list[i] for i in similar_songs_indices]
            similar_songs_titles = [self.train_s2t[song] for song in similar_songs]
            score = self.distances[item_index][1:]
            return pd.DataFrame({"song_id":similar_songs, "title":similar_songs_titles, "score":score})
            
        elif (return_type == "ordered_dict"):
        
            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            similar_songs = [self.train_songs_list[i] for i in similar_songs_indices]
            score = self.distances[item_index][1:]
            recommendations_ordered_dict = OrderedDict(zip(similar_songs,score))
            return recommendations_ordered_dict
        
    # Use KNN Recommender to recommend Songs to user
    def recommend(self, user, tau=500, return_type="df"):
        
        # Get list of unique songs listened by user
        user_songs = self.train_u2s[user]
        
        recommendations_list = []
        recommend_dict = {}
        
        # Loop through the songs listened by user
        for item in user_songs:
            
            # Get similar songs similar to the listened song
            similar_dict = self.get_similar_items(item, return_type="ordered_dict")
            
            # Assign each similar song a score = -ve of it's distance
            for recommend in similar_dict.keys():
                score = -similar_dict[recommend]
                if recommend in recommend_dict:
                    recommend_dict[recommend] += score
                else:
                    recommend_dict[recommend] = score
        
        # Sort the songs in decreasing order of their score
        recommendations_list = sorted(recommend_dict.keys(), key=lambda s:recommend_dict[s], reverse=True)
        
        # Recommend only top tau songs
        recommendations_list = recommendations_list[:tau]
          
        if (return_type == "list"):
            return recommendations_list
            
        # By Default Return DataFrame
            
        elif (return_type == "df"):
        
            # Create a DataFrame of Recommended (Song, Title, Score) Triplets

            recommendations_songs_titles = [self.train_s2t[song] for song in recommendations_list]
            score = [recommend_dict[song] for song in recommendations_list]
            return pd.DataFrame({"song_id":recommendations_list, "title":recommendations_songs_titles, "score":score})
            
        elif (return_type == "ordered_dict"):
        
            # Create an Ordered Dictionary of Recommended (Song, Score) Pairs
            score = [recommend_dict[song] for song in recommendations_list]
            return OrderedDict(zip(recommendations_list,score))

