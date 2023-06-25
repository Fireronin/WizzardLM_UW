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
# %%
# save root nodes and immediate children to csv

df2 = pd.DataFrame()

collumns_to_stack = []


for index, row in tqdm(df.iterrows()):
    if row['message_id'] in set_of_root_nodes:
        collumns_to_stack.append(row)
    if row['parent_id'] in set_of_root_nodes:
        collumns_to_stack.append(row)
        

df2 = pd.concat(collumns_to_stack, axis=1).T

columns=['message_id', 'parent_id', 'created_date', 'text', 'role', 'lang' , 'message_tree_id']

# take only columns from columns list'
df2 = df2[columns]
# save to csv
df2.to_csv('oasst1-train-tree.csv', index=False)



# %%

