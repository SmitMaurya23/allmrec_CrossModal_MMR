import os
import os.path
import gzip
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

def parse(path):
    """
    Generator function to parse gzip-compressed JSON lines
    """
    g = gzip.open(path, 'rb')
    for l in tqdm(g):
        yield json.loads(l)

def preprocess(fname):
    """
    Preprocess the dataset by counting interactions and extracting metadata.
    """
    countU = defaultdict(lambda: 0)  # Counting interactions for each user
    countP = defaultdict(lambda: 0)  # Counting interactions for each item
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

    usermap = dict()  # Mapping from reviewerID to userID
    usernum = 0  # User counter
    itemmap = dict()  # Mapping from asin to itemID
    itemnum = 0  # Item counter
    User = dict()  # User-item interactions
    review_dict = {}  # Dictionary of reviews and summaries for items
    name_dict = {'title':{}, 'description':{}}  # Name dictionary for items

    # Loading metadata (titles, descriptions, etc.)
    with open(f'../../data/amazon/meta_{fname}.json', 'r') as f:
        json_data = f.readlines()
    
    data_list = [json.loads(line[:-1]) for line in json_data]  # Load metadata into list
    meta_dict = {}
    for l in data_list:
        meta_dict[l['asin']] = l

    # Processing interaction data
    for l in parse(file_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']

        # Set threshold for user-item interactions
        threshold = 5
        if ('Beauty' in fname) or ('Toys' in fname):
            threshold = 4

        if countU[rev] < threshold or countP[asin] < threshold:
            continue

        # Assign user ID if not already assigned
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []

        # Assign item ID if not already assigned
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])

        # Process reviews and summaries
        if itemmap[asin] in review_dict:
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                pass
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                pass
        else:
            review_dict[itemmap[asin]] = {'review': {}, 'summary':{}}
            try:
                review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
            except:
                pass
            try:
                review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
            except:
                pass
        
        # Safely retrieve the description and title from metadata
        try:
            item_meta = meta_dict.get(asin, {})
            description = item_meta.get('description', '').strip()  # Default to empty string if not found

            # Check if description is available, else set default value
            if not description:
                name_dict['description'][itemmap[asin]] = 'No Description'
            else:
                name_dict['description'][itemmap[asin]] = description

            # Retrieve title with a fallback if not found
            name_dict['title'][itemmap[asin]] = item_meta.get('title', 'No Title')  # Default to 'No Title' if missing
        except KeyError as e:
            print(f"KeyError: Missing data for ASIN {asin}. Error: {e}")
        except Exception as e:
            print(f"Unexpected error with ASIN {asin}: {e}")

    # Save the name dictionary to a file (Pickle format)
    with open(f'../../data/amazon/{fname}_text_name_dict.pkl', 'wb') as tf:
        pickle.dump(name_dict, tf)

    # Sort user interactions by time
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(usernum, itemnum)

    # Write user-item interactions to a file
    with open(f'../../data/amazon/{fname}.txt', 'w') as f:
        for user in User.keys():
            for i in User[user]:
                f.write('%d %d\n' % (user, i[1]))
