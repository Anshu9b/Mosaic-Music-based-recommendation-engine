# 🎧 Mosaic - Music based recommndation Engine

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

## Hosted Application
The streamlit frontend is hosted at https://share.streamlit.io/anshu9b/mosaic-music-based-recommendation-engine/main/app.py

## My Journey
- Initial days were spent in collecting dataset for my project , I have gone through different data's at last downloaded Kaggle- Spotify dataset by Imuhamad.
- After that I worked on algorithms being a beignner in this my 9-10 days were spented in configuring them according to my dataset.
- After that I configured my jupyter notebook for initial testing of algorithms using recommendation.ipynb file and then broken down into preprocessing.py and run_recommendor.py for pickling and recommendations.
- After that I was able to integrate my jupyter notebooks with streamlit features to land with my app.py file which has main frontend part.
- I then experimented and worked with different themes for UI and was able to land with custom themes and sidebars and menubars.<br> </br>

As per requirements I have made a [google slides presentation] https://docs.google.com/presentation/d/1ClJW7FRqo6ycregjw_KIvj48rf6azVoo/edit?usp=sharing&ouid=106169485861165829208&rtpof=true&sd=true  and a [video demo ]  https://www.youtube.com/watch?v=yZMcLTcoGGs


## !! Thankyou for visting my project





 







