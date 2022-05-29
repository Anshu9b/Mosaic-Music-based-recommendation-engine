# Mosaic - Music based recommndation Engine

A Music Recommendation web based application  which focuses on Algorithms- Cosine Similarity, Tfidf and Eucledian distance and basis on that gives the user freedom to choose from- what is
- Trending 
- Discover less popular Song <br/><br/>

User can choose on the basis 6 modes of functionalities and System can recommend max 20 Songs.
The main purpose of choosing Cosine similarity model was it comibes both user based and content based features which are
## Features
### Functionalities
- 2 modes of recommendations between which the user can toggle anytime:
  - Keep up with what's trending (suggests more popular songs)
  - Hidden gems (suggests less popular songs)
- A total of 6 types of recommendations:
  - by same artist
  - lyrically similar (not seen in most popular music streaming services)
  - similar energy
  - similar mood
  - released around the same time
  - random
  - On the basis of this system can recommend 20 Songs.
  
### UX related features
- Ability to show or hide the lyrics for each song
- Youtube video linked for each song
- Viewing a recommended song by clicking on the listen button, will then lead to recommendations based on that song.

## Technologies Used

 -Python
  ### Library Used
     - Pandas
     - Numpy
     - Scikit Learn(for Tfidf, eucleian distance)
- Jupyter Notebook( For recommendation.ipynb)       
- Streamlit (for frontend)
- Gitpod(for Manging workspace)
- StreamLit Cloud(for deployment)

## Folder Organization

    ├──  test.yml                           # CI pipeline to automatically test code
    ├── .streamlit/config.toml              # For UI Design 
    ├── pickles                             # For Data Mapping
    │   ├── data.pkl                     
    │   ├── energy_similarity_mapping.pkl 
    │   ├── lyric_similarity_mapping.pkl
    │   └── mood_similarity_mapping.pkl
    ├── .gitignore 
    ├── recommendation.ipynb                  # Main recommendation algo file
    ├── app.py                                # Frontend using streamlit
    ├── preprocessing.py                      # Generate pickles
    ├── recommender.py                        # code for core recommendation system
    ├── requirements.txt                      
    ├── run_recommender.py                    # testing the recommendation file
    └── Spotify_songs.csv                     # Initial DataSet
    
## Local Set Up

After cloning the repository and firing up a virtual environment, run the following commands:
```
pip install -r requirements.txt    # installs dependencies
streamlit run app.py               # runs the frontend locally in browser
```
For running tests, ensure you have pytest installed via pip, then simply run the command `pytest`

## Code Deployment
The streamlit frontend is hosted at 





 







