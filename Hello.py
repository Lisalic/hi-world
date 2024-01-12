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

st.set_page_config(
    page_title="Book Recommendation",
    page_icon="📚",
)

# Load the dataset
file_path = 'hello.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"File not found: {file_path}")
    st.stop()

# Preprocess the data
df['Title'] = df['Title'].fillna('')
df['Genre'] = df['Genre'].fillna('')

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Title'] + ' ' + df['Genre'])

# Streamlit app
st.title('Book Recommendation App')

# User input
user_input = st.text_input('Enter a book title or genre:', '')

# Submit button
if st.button('Submit'):
    # Recommend five books based on user input with randomness and preference to higher-ranked items
    if user_input:
        # Transform the user input using the TF-IDF vectorizer
        input_vector = tfidf_vectorizer.transform([user_input])

        # Compute the cosine similarity between the user input and all books
        sim_scores = linear_kernel(input_vector, tfidf_matrix).flatten()

        # Select the top 25 recommendations
        top_indices = sim_scores.argsort()[::-1][:25]

        # Introduce randomness with a preference for higher-ranked items
        weights = np.arange(1, 26)  # Higher weights for higher-ranked items
        sampled_indices = np.random.choice(top_indices, 5, replace=False, p=weights / weights.sum())

        # Display the recommended books
        if len(sampled_indices) > 0:
            st.subheader('Recommended Books:')
            for i, idx in enumerate(sampled_indices):
                st.write(f"{i + 1}. {df.iloc[idx]['Title']} by {df.iloc[idx]['Author']} ({df.iloc[idx]['Genre']})")
        else:
            st.write("No recommendations found.")
