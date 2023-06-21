import json

# Read the JSON file
with open('F:/SS23/DILAB_project/SS2023_DI-Lab_Precitaste/RPCdataset/annotations/instances_val2019.json', 'r') as file:
    data = json.load(file)

# CHANGE THE CAT ID IN ANNOTATIONS:
all_supercat = sorted({k['supercategory'] for k in data['categories']})  
### VERSION 1
# Create a dictionary to map category IDs to supercategory IDs
# category_to_supercategory = {}
# ids_supercat = list(range(17))
# for item in data['categories']:
#     name_wo_num = '_'.join(item['name'].split('_')[1:])
#     if name_wo_num == item['supercategory']:
#         category_to_supercategory[item['id']] = all_supercat.index(name_wo_num)
### VERSION 2
id2id = {idx+1: all_supercat.index('_'.join(cat['name'].split('_')[1:])) for idx, cat in enumerate(data['categories'])} 

# Iterate through the bounding box information
for bbox_info in data["annotations"]:
    # Replace the category ID with the supercategory ID
    # category_id = bbox_info['category_id']
    # bbox_info['category_id'] = category_to_supercategory[category_id]
    bbox_info['category_id'] = id2id[bbox_info['category_id']]

# CHANGE THE CATEGORIES FROM 200 TO 17:
new_cat = {idx:supercat for idx, supercat in enumerate(all_supercat)}
categories = []
for category_id, category_name in new_cat.items():
    category = {
        "supercategory": category_name,
        "id": category_id,
        "name": category_name
    }
    categories.append(category)
data['categories'] = categories

# Save the modified JSON file
with open('F:/SS23/DILAB_project/SS2023_DI-Lab_Precitaste/RPCdataset/annotations/instances_val2019.json', 'w') as file:
    json.dump(data, file)
