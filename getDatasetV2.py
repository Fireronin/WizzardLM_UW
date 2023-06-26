#%%
#load oasst1-train.csv
import pandas as pd
from tqdm import tqdm
df = pd.read_csv('oasst1-train.csv')


# %%
root_nodes = []

for index, row in df.iterrows():
    if pd.isna(row['parent_id']):
        root_nodes.append(row['message_id'])


# df
root_nodes_id = pd.DataFrame(root_nodes, columns=['message_id'])

#
set_of_root_nodes = set(root_nodes)

# %%
print(len(root_nodes))
#%%
# import json

# collumns_to_stack = []

# for index, row in tqdm(df.iterrows()):
#     # read detoxify as python dict
#     # replace ' with " in row['detoxify']
#     if pd.isna(row['detoxify']):
#         continue
#     row2 = row['detoxify'].replace("'", '"')
    
#     detoxify = json.loads(row2)
#     is_toxic = False
#     for key in detoxify.keys():
#         if detoxify[key] > 0.01:
#             is_toxic = True
#             print(row['text'])
#             break
#     if not is_toxic:
#         collumns_to_stack.append(row)


# df = pd.concat(collumns_to_stack, axis=1).T

# %%
# save root nodes and immediate children to csv

df2 = pd.DataFrame()

collumns_to_stack = []

answer = {}

for index, row in tqdm(df.iterrows()):
    if row['message_id'] in set_of_root_nodes:
        collumns_to_stack.append(row)
    if row['parent_id'] in set_of_root_nodes:
        answer[row['parent_id']] = row['message_id']
        # add new collumn text_answer 
        collumns_to_stack[-1]['text_answer'] = row['text']
        

df3 = pd.concat(collumns_to_stack, axis=1).T
#%%
#detoxify {'toxicity': 0.00044308538781479, 'severe_toxicity': 3.252684837207198e-05, 'obscene': 0.00023475120542570949, 'identity_attack': 0.0001416115992469713, 'insult': 0.00039489680784754455, 'threat': 4.075629112776369e-05, 'sexual_explicit': 2.712695459194947e-05}
# remove toxicity




columns=['message_id', 'parent_id', 'created_date', 'text', 'role', 'lang' , 'message_tree_id','text_answer', 'detoxify']


# take only columns from columns list'
df3 = df3[columns]

# take only rows with lang == 'en'
df3 = df3[df3['lang'] == 'en']


# save to csv
df3.to_csv('oasst1-train-tree.csv', index=False)

print(len(df3))

# %%

