import json
import pandas as pd
import re
import matplotlib.pyplot as plt

def main():
    data = load_data('/home/dimitrijem/Documents/dipl/project/src/ad_group.json')
    all_characters = data['name'].apply(extract_characters)
    all_characters = [char for sublist in all_characters for char in sublist]
    unique_characters = set(all_characters)
    data['words'] = data['name'].apply(lambda x: extract_words(x))
    data['word_count'] = data['words'].apply(lambda x: len(x))
    data['word_count'].value_counts().plot(kind='bar')
    plt.xlabel('Word count')
    plt.ylabel('Ad count')
    plt.show()

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
        
    json_data = data['data']
    data = pd.json_normalize(json_data)
    return data

def extract_characters(text):
    return re.findall(r'[^\s\w+]', text)

def extract_words(text):
    return re.findall(r'[\w+]+', text)


if __name__ == "__main__":
    main()
