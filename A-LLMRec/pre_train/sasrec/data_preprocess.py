def preprocess(fname):
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
    name_dict = {'title':{}, 'description':{}}

    # Loading metadata
    with open(f'../../data/amazon/meta_{fname}.json', 'r') as f:
        json_data = f.readlines()
    
    data_list = [json.loads(line[:-1]) for line in json_data]
    meta_dict = {}
    for l in data_list:
        meta_dict[l['asin']] = l

    # Processing interaction data
    for l in parse(file_path):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']

        threshold = 5
        if ('Beauty' in fname) or ('Toys' in fname):
            threshold = 4

        if countU[rev] < threshold or countP[asin] < threshold:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []

        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])

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
        
        try:
            # Safely retrieve the description from metadata
            item_meta = meta_dict.get(asin, {})
            description = item_meta.get('description', '')

            # Check if description is a valid non-empty string
            if not description or not isinstance(description, str) or description.strip() == '':
                name_dict['description'][itemmap[asin]] = 'No Description'  # Default if missing or empty
            else:
                name_dict['description'][itemmap[asin]] = description.strip()

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
