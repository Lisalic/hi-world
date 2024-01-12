# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

#page config
st.title('Book Recommendation App')
st.set_page_config(
    page_title="Book Recommendation",
    page_icon="📚",
)

#loading data
df = pd.read_csv(hello.csv)
df['Title'] = df['Title'].fillna('')
df['Genre'] = df['Genre'].fillna('')


corpus = df['Title'] + ' ' + df['Genre']
unique_words = set(' '.join(corpus).split())
word_to_index = {word: i for i, word in enumerate(unique_words)}

def vectorize(text):
    vector = np.zeros(len(unique_words))
    for word in text.split():
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    return vector / np.linalg.norm(vector)

book_vectors = np.array([vectorize(text) for text in corpus])


user_input = st.text_input('Enter a book title or genre:', '')

if st.button('Submit'):
    if user_input:
        input_vector = vectorize(user_input)

        # similarity checking
        sim_scores = np.dot(book_vectors, input_vector)

        # selecting random 5 from top 25 
        top_indices = sim_scores.argsort()[::-1][:25]
        weights = np.arange(1, 26)  
        sampled_indices = np.random.choice(top_indices, 5, replace=False, p=weights / weights.sum())

        # Display 
        if len(sampled_indices) > 0:
            st.subheader('Recommended Books:')
            for i, idx in enumerate(sampled_indices):
                st.write(f"{i + 1}. {df.iloc[idx]['Title']} by {df.iloc[idx]['Author']} ({df.iloc[idx]['Genre']})")
        else:
            st.write("No recommendations found.")
