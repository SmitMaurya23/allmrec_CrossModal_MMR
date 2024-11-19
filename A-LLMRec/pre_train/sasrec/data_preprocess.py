import os
import gzip
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

def parse(path):
    """Parse JSON data from a gzip file."""
    g = gzip.open(path, 'rb')
    for l in tqdm(g, desc="Parsing gzip file"):
        yield json.loads(l)

def preprocess(fname):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)

    file_path = f'../../data/amazon/{fname}.json.gz'

    # Counting interactions for each user and item
    print("Counting user and item interactions...")
    for l in parse(file_path):
        asin = l.get('asin')
        rev = l.get('reviewerID')
        if asin and rev:
            countU[rev] += 1
            countP[asin] += 1

    usermap = {}
    itemmap = {}
    User = defaultdict(list)
    review_dict = {}
    name_dict = {'title': {}, 'description': {}, 'category': {}}

    # Load metadata
    print("Loading metadata...")
    try:
        with open(f'../../data/amazon/meta_{fname}.json', 'r') as f:
            json_data = [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file for {fname} not found!")
    
    meta_dict = {l['asin']: l for l in json_data if 'asin' in l}

    # Process interaction data
    print("Processing interaction data...")
    usernum = 0
    itemnum = 0
    for l in parse(file_path):
        asin = l.get('asin')
        rev = l.get('reviewerID')
        time = l.get('unixReviewTime')

        if not asin or not rev or not time:
            continue

        # Apply thresholds
        threshold = 4 if ('Beauty' in fname or 'Toys' in fname) else 5
        if countU[rev] < threshold or countP[asin] < threshold:
            continue

        # Map user and item IDs
        if rev not in usermap:
            usernum += 1
            usermap[rev] = usernum
        if asin not in itemmap:
            itemnum += 1
            itemmap[asin] = itemnum

        userid = usermap[rev]
        itemid = itemmap[asin]
        User[userid].append([time, itemid])

        # Collect review and summary
        if itemid not in review_dict:
            review_dict[itemid] = {'review': {}, 'summary': {}}
        review_dict[itemid]['review'][userid] = l.get('reviewText', 'No Review')
        review_dict[itemid]['summary'][userid] = l.get('summary', 'No Summary')

        # Populate metadata
        item_meta = meta_dict.get(asin, {})
        name_dict['title'][itemid] = item_meta.get('title', 'Unknown Title').strip()
        name_dict['description'][itemid] = item_meta.get('description', ['No Description'])[0].strip()
        name_dict['category'][itemid] = item_meta.get('category', ['No Category'])[0].strip()

    # Save metadata
    print("Saving metadata...")
    try:
        with open(f'../../data/amazon/{fname}_text_name_dict.json.gz', 'wb') as tf:
            pickle.dump(name_dict, tf)
    except Exception as e:
        raise IOError(f"Error saving metadata: {e}")

    # Sort user interactions by time
    for userid in User:
        User[userid].sort(key=lambda x: x[0])

    # Write interactions to file
    print(f"Total Users: {usernum}, Total Items: {itemnum}")
    try:
        with open(f'../../data/amazon/{fname}.txt', 'w') as f:
            for user, interactions in User.items():
                for _, item in interactions:
                    f.write(f"{user} {item}\n")
    except Exception as e:
        raise IOError(f"Error saving interactions: {e}")

    print(f"Metadata Saved: {len(name_dict['title'])} Titles, {len(name_dict['description'])} Descriptions, {len(name_dict['category'])} Categories")
