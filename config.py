########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
raw_dir = "/mnt/hdd.data/hzc/atlas_notrm"

# The directory to save all artifacts
artifact_dir = "/mnt/hdd.data/hzc/atlas_moe/M5_only/"

# The directory to save the vectorized graphs
graphs_dir = artifact_dir + "graphs/"

# The directory to save the models
models_dir = artifact_dir + "models/"

# The directory to save the results after testing
test_re = artifact_dir + "test_re/"

# The directory to save all visualized results
vis_re = artifact_dir + "vis_re/"



########################################################
#
#               Database settings
#
########################################################

# Database name
database = 'tc_e5_theia_dataset_db'

#             Name             |  Owner   | Encoding |   Collate   |    Ctype    |   Access privileges   
# -----------------------------+----------+----------+-------------+-------------+-----------------------
#  optc_db                     | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  postgres                    | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  streamspot                  | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_cadet_dataset_db         | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_clearscope3_dataset_db   | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_e5_cadets_dataset_db     | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_e5_clearscope_dataset_db | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_e5_theia_dataset_db      | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 
#  tc_theia_dataset_db         | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | 

# Only config this setting when you have the problem mentioned
# in the Troubleshooting section in settings/environment-settings.md.
# Otherwise, set it as None
host = 'localhost'

# Database user
user = 'postgres'

# The password to the database user
password = '123456'

# The port number for Postgres
port = '5432'


########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
# edge_reversed = [
#     "EVENT_ACCEPT",
#     "EVENT_RECVFROM",
#     "EVENT_RECVMSG"
# ]

# # The following edges are the types only considered to construct the
# # temporal graph for experiments.
# include_edge_type=[
#     "EVENT_WRITE",
#     "EVENT_READ",
#     "EVENT_CLOSE",
#     "EVENT_OPEN",
#     "EVENT_EXECUTE",
#     "EVENT_SENDTO",
#     "EVENT_RECVFROM",
# ]

# The map between edge type and edge ID
# rel2id for cadets

########################################################
#
#                   Model dimensionality
#
########################################################

# Node Embedding Dimension
node_embedding_dim = 16

# Node State Dimension
node_state_dim = 100

memory_dim = 100

# Neighborhood Sampling Size
neighbor_size = 20

# Edge Embedding Dimension
edge_dim = 100

# The time encoding Dimension
time_dim = 100

embedding_dim = 100


max_node_num = 324818


########################################################
#
#                   Train&Test
#
########################################################

# Batch size for training and testing
BATCH = 128
device = 'cuda:1'

# Parameters for optimizer
lr=0.00005
eps=1e-08
weight_decay=0.01

epoch_num=30

# The size of time window, 60000000000 represent 1 min in nanoseconds.
# The default setting is 15 minutes.
time_window_size = 60000000000 * 15


########################################################
#
#                   Threshold
#
########################################################

beta_day6 = 100
beta_day7 = 100

import os

if not os.path.exists(artifact_dir):
    os.system(f"mkdir -p {artifact_dir}")

if not os.path.exists(graphs_dir):
    os.system(f"mkdir -p {graphs_dir}")

if not os.path.exists(models_dir):
    os.system(f"mkdir -p {models_dir}")

if not os.path.exists(test_re):
    os.system(f"mkdir -p {test_re}")

if not os.path.exists(vis_re):
    os.system(f"mkdir -p {vis_re}")