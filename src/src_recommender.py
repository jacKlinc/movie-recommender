from fastbook import *
from fastai.collab import *
from fastai.tabular.all import *
from fastai import *

from dot_product_bias import DotProductBias

import streamlit as st


'# Movie Recommender'
"Here's the [GitHub](https://github.com/jacKlinc/movie_recommender) repo"

### Type favourite
fav = st.text_input("Type a movie and hit Enter...")

### Load model  
learn = load_learner('./data/mdl_movie-recommend.pkl')

### Load dataset
path = untar_data(URLs.ML_100k)

ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])

movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)

# Join ratings and movies
ratings = ratings.merge(movies)

# Add to DataLoaders object
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)


### Make embedding prediction
movie_factors = learn.model.movie_factors
idxs = dls.classes['title'].o2i[fav]
# Finds distances from chosen title
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idxs][None])
# gets top 4
idx = distances.argsort(descending=True)[1:4]


# Recommend
if fav:
    '### You like: '
    fav
    '### You should watch: ', [dls.classes['title'][i] for i in idx]