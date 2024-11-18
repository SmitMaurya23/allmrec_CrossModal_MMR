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
    for l in tqdm(g):
        yield json.loads(l)

def preprocess(fname):
    """
    Preprocesses the dataset by counting user-item interactions, mapping users and items,
    and extracting metadata for the recommendation model.
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
    f = open(f'../../data/amazon/meta_{fname}.json', 'r')
    json_data = f.readlines()
    f.close()
    data_list = [json.loads(line[:-1]) for line in json_data]
    meta_dict = {}

    for l in data_list:
        asin = l.get('asin', None)
        if not asin:
            continue
        meta_dict[asin] = {
            'title': l.get('title', ''),
            'description': l.get('description', [])
        }
    
    # Debugging: Metadata size
    print(f"Metadata entries: {len(meta_dict)}")

    # Filtering and mapping users/items
    for l in parse(file_path):
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']

        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if rev not in usermap:
            usermap[rev] = usernum
            usernum += 1
            User[usermap[rev]] = []

        if asin not in itemmap:
            itemmap[asin] = itemnum
            itemnum += 1

        User[usermap[rev]].append((itemmap[asin], time))
        review_dict[itemmap[asin]] = l.get('reviewText', '')
        name_dict['title'][itemmap[asin]] = meta_dict.get(asin, {}).get('title', '')
        name_dict['description'][itemmap[asin]] = meta_dict.get(asin, {}).get('description', '')

    # Debugging: Dataset size
    print(f"Total Users: {len(usermap)}, Total Items: {len(itemmap)}")

    # Ensure mapping matches similarity matrix dimensions
    assert len(itemmap) > 0, "No items found after filtering. Check preprocessing logic."
    assert len(usermap) > 0, "No users found after filtering. Check preprocessing logic."

    # Save mappings and processed data
    os.makedirs(f'../../data/amazon/processed', exist_ok=True)
    with open(f'../../data/amazon/processed/{fname}_usermap.pkl', 'wb') as f:
        pickle.dump(usermap, f)
    with open(f'../../data/amazon/processed/{fname}_itemmap.pkl', 'wb') as f:
        pickle.dump(itemmap, f)
    with open(f'../../data/amazon/processed/{fname}_user.pkl', 'wb') as f:
        pickle.dump(User, f)
    with open(f'../../data/amazon/processed/{fname}_reviews.pkl', 'wb') as f:
        pickle.dump(review_dict, f)
    with open(f'../../data/amazon/processed/{fname}_names.pkl', 'wb') as f:
        pickle.dump(name_dict, f)

    print(f"Preprocessed data saved for {fname}.")

# Example of running the preprocessing function
if __name__ == "__main__":
    preprocess("Movies_and_TV")
