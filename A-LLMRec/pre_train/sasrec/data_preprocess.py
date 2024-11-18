import os
import gzip
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

def parse(path):
    """
    Parses a gzip file and yields JSON objects line by line.
    """
    g = gzip.open(path, 'rb')
    for l in tqdm(g, desc="Parsing file"):
        yield json.loads(l)

def preprocess(fname):
    """
    Preprocesses the dataset for the recommendation system by:
    - Counting user-item interactions.
    - Mapping users and items to unique IDs.
    - Extracting metadata for items.
    - Creating user-item interaction sequences.
    """
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    file_path = f'../../data/amazon/{fname}.json.gz'

    # Counting interactions for each user and item
    for l in parse(file_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        countU[rev] += 1
        countP[asin] += 1

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    review_dict = {}
    name_dict = {'title': {}, 'description': {}}

    # Reading metadata
    with open(f'../../data/amazon/meta_{fname}.json', 'r') as f:
        json_data = f.readlines()
    data_list = [json.loads(line.strip()) for line in json_data]
    meta_dict = {l['asin']: l for l in data_list}

    for l in parse(file_path):
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']

        # Threshold adjustment for specific datasets
        threshold = 5
        if 'Beauty' in fname or 'Toys' in fname:
            threshold = 4

        if countU[rev] < threshold or countP[asin] < threshold:
            continue

        # User mapping
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []

        # Item mapping
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])

        # Review and summary handling
        if itemid in review_dict:
            review_dict[itemid]['review'][userid] = l.get('reviewText', "")
            review_dict[itemid]['summary'][userid] = l.get('summary', "")
        else:
            review_dict[itemid] = {'review': {userid: l.get('reviewText', "")},
                                   'summary': {userid: l.get('summary', "")}}

        # Metadata handling
        try:
            name_dict['description'][itemid] = (
                meta_dict[asin].get('description', ['Empty description'])[0]
            )
            name_dict['title'][itemid] = meta_dict[asin].get('title', 'No title')
        except KeyError:
            name_dict['description'][itemid] = 'Empty description'
            name_dict['title'][itemid] = 'No title'

    # Save processed data
    output_path = f'../../data/amazon/{fname}_text_name_dict.json.gz'
    with gzip.open(output_path, 'wb') as tf:
        pickle.dump(name_dict, tf)

    # Sort user interactions by timestamp
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(f"Total users: {usernum}, Total items: {itemnum}")

    # Write user-item interactions to text file
    with open(f'../../data/amazon/{fname}.txt', 'w') as f:
        for user, interactions in User.items():
            for interaction in interactions:
                f.write(f"{user} {interaction[1]}\n")

    print(f"Preprocessing completed for {fname}. Data saved.")

# Example execution
if __name__ == "__main__":
    preprocess("Movies_and_TV")
