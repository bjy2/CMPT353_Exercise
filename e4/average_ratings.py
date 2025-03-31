import sys
import pandas as pd
import difflib


def match_title(title, movie_list):
    matches = difflib.get_close_matches(title, movie_list, cutoff=0.6)
    return matches[0] if matches else None


def process_ratings(movielist_file, ratings_file, output_file):
    movie_list = pd.read_csv(movielist_file, header=None)[0].tolist()
    ratings_df = pd.read_csv(ratings_file)

    ratings_df['matched_title'] = ratings_df['title'].map(lambda x: match_title(x, movie_list))
    ratings_df = ratings_df.dropna(subset=['matched_title'])

    avg_ratings = ratings_df.groupby('matched_title')['rating'].mean().round(2).reset_index()
    avg_ratings.columns = ['title', 'rating']
    avg_ratings.sort_values(by='title').to_csv(output_file, index=False)


if __name__ == "__main__":
    movielist_file = sys.argv[1]
    ratings_file = sys.argv[2]
    output_file = sys.argv[3]
    process_ratings(movielist_file, ratings_file, output_file)
