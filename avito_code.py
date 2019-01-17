
from avito_utils import *
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import xgboost
import pandas as pd
import numpy as np
import argparse
import joblib

from lightfm import LightFM

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

print("starting...")
# from https://github.com/dmitryhd/lightfm/blob/master/lightfm/inference.py#L69
def _precompute_representation(features, feature_embeddings, feature_biases):
    representation = features.dot(feature_embeddings)
    representation_bias = features.dot(feature_biases)
    return representation, representation_bias


parser = argparse.ArgumentParser(description='Run Avito Recsys Challenge solution.')
parser.add_argument('data', help='training data_set folder')
parser.add_argument('challenge', help='path of the challenge_set file')

args = parser.parse_args()

path_data = args.data
path_challenge_set = args.challenge

create_df_data(path_data, path_challenge_set)
np.random.seed(0)

# read data
df_tracks = pd.read_hdf('df_data/df_tracks.hdf')
df_playlists = pd.read_hdf('df_data/df_playlists.hdf')
df_playlists_info = pd.read_hdf('df_data/df_playlists_info.hdf')
df_playlists_test = pd.read_hdf('df_data/df_playlists_test.hdf')
df_playlists_test_info = pd.read_hdf('df_data/df_playlists_test_info.hdf')

# delete
#############################################################""
# from sklearn.model_selection import train_test_split
# _,test_data=train_test_split(df_playlists_test_info, test_size=0.03,random_state=0)
# df_playlists_test_info=test_data
# df_playlists_test_info=df_playlists_test_info[((df_playlists_test_info.num_tracks!=242) & (df_playlists_test_info.num_tracks!=185) & (df_playlists_test_info.num_tracks!=220) & (df_playlists_test_info.num_tracks!=238) & (df_playlists_test_info.num_tracks!=195) & (df_playlists_test_info.num_tracks!=189) & (df_playlists_test_info.num_tracks!=157) & (df_playlists_test_info.num_tracks!=175)& (df_playlists_test_info.num_tracks!=25) & (df_playlists_test_info.num_tracks!=215) & (df_playlists_test_info.num_tracks!=45)&(df_playlists_test_info.num_tracks!=161)& (df_playlists_test_info.num_tracks!=69) & (df_playlists_test_info.num_tracks!=57) & (df_playlists_test_info.num_tracks!=105)&(df_playlists_test_info.num_tracks!=39) & (df_playlists_test_info.num_tracks!=127)& (df_playlists_test_info.num_tracks!=151)&(df_playlists_test_info.num_tracks!=60)&(df_playlists_test_info.num_tracks!=49)&(df_playlists_test_info.num_tracks!=65)&(df_playlists_test_info.num_tracks!=73)&(df_playlists_test_info.num_tracks!=150)&(df_playlists_test_info.num_tracks!=86)&(df_playlists_test_info.num_tracks!=160)&(df_playlists_test_info.num_tracks!=187)&(df_playlists_test_info.num_tracks!=83)&(df_playlists_test_info.num_tracks!=197)&(df_playlists_test_info.num_tracks!=84)&(df_playlists_test_info.num_tracks!=145)&(df_playlists_test_info.num_tracks!=67)&(df_playlists_test_info.num_tracks!=135)&(df_playlists_test_info.num_tracks!=129)&(df_playlists_test_info.num_tracks!=123)&(df_playlists_test_info.num_tracks!=155)&(df_playlists_test_info.num_tracks!=153)&(df_playlists_test_info.num_tracks!=154)&(df_playlists_test_info.num_tracks!=210)&(df_playlists_test_info.num_tracks!=159)&(df_playlists_test_info.num_tracks!=107)&(df_playlists_test_info.num_tracks!=97)&(df_playlists_test_info.num_tracks!=95)&(df_playlists_test_info.num_tracks!=106)&(df_playlists_test_info.num_tracks!=43)&(df_playlists_test_info.num_tracks!=239)&(df_playlists_test_info.num_tracks!=202)&(df_playlists_test_info.num_tracks!=166)&(df_playlists_test_info.num_tracks!=182)&(df_playlists_test_info.num_tracks!=232)&(df_playlists_test_info.num_tracks!=165)&(df_playlists_test_info.num_tracks!=167)&(df_playlists_test_info.num_tracks!=168)&(df_playlists_test_info.num_tracks!=243)&(df_playlists_test_info.num_tracks!=164)&(df_playlists_test_info.num_tracks!=163)&(df_playlists_test_info.num_tracks!=172)&(df_playlists_test_info.num_tracks!=193)&(df_playlists_test_info.num_tracks!=191)&(df_playlists_test_info.num_tracks!=190)&(df_playlists_test_info.num_tracks!=156)&(df_playlists_test_info.num_tracks!=144)&(df_playlists_test_info.num_tracks!=181)&(df_playlists_test_info.num_tracks!=184)&(df_playlists_test_info.num_tracks!=199)&(df_playlists_test_info.num_tracks!=204)&(df_playlists_test_info.num_tracks!=217))]
# df_playlists_test_info=df_playlists_test_info[((df_playlists_test_info.num_tracks!=222)&(df_playlists_test_info.num_tracks!=226)&(df_playlists_test_info.num_tracks!=228)&(df_playlists_test_info.num_tracks!=125))]


#########################################


num_tracks = df_playlists_info.groupby('num_tracks').pid.apply(np.array)

validation_playlists = {}
for i, j in df_playlists_test_info.num_tracks.value_counts().reset_index().values:
    validation_playlists[i] = np.random.choice(num_tracks.loc[i], 2 * j, replace=False)

val1_playlist = {}
val2_playlist = {}
for i in [0, 1, 5, 10, 25, 100]:

    val1_playlist[i] = []
    val2_playlist[i] = []

    value_counts = df_playlists_test_info.query('num_samples==@i').num_tracks.value_counts()
    for j, k in value_counts.reset_index().values:
        val1_playlist[i] += list(validation_playlists[j][:k])
        validation_playlists[j] = validation_playlists[j][k:]

        val2_playlist[i] += list(validation_playlists[j][:k])
        validation_playlists[j] = validation_playlists[j][k:]

val1_index = df_playlists.pid.isin(val1_playlist[0])
val2_index = df_playlists.pid.isin(val2_playlist[0])
for i in [1, 5, 10, 25, 100]:
    val1_index = val1_index | (df_playlists.pid.isin(val1_playlist[i]) & (df_playlists.pos >= i))

    val2_index = val2_index | (df_playlists.pid.isin(val2_playlist[i]) & (df_playlists.pos >= i))

train = df_playlists[~(val1_index | val2_index)]

val1 = df_playlists[val1_index]
val2 = df_playlists[val2_index]

val1_pids = np.hstack([val1_playlist[i] for i in val1_playlist])
val2_pids = np.hstack([val2_playlist[i] for i in val2_playlist])

train = pd.concat([train, df_playlists_test])

train.to_hdf('df_data/train.hdf', key='abc')

# store validation_data

val1.to_hdf('df_data/val1.hdf', key='abc')
val2.to_hdf('df_data/val2.hdf', key='abc')

joblib.dump(val1_pids, 'df_data/val1_pids.pkl')
joblib.dump(val2_pids, 'df_data/val2_pids.pkl')

# lightfm

# !mkdir models
# read data

df_tracks = pd.read_hdf('df_data/df_tracks.hdf')
df_playlists = pd.read_hdf('df_data/df_playlists.hdf')
df_playlists_info = pd.read_hdf('df_data/df_playlists_info.hdf')
df_playlists_test = pd.read_hdf('df_data/df_playlists_test.hdf')
df_playlists_test_info = pd.read_hdf('df_data/df_playlists_test_info.hdf')

train = pd.read_hdf('df_data/train.hdf')
val = pd.read_hdf('df_data/val1.hdf')
val1_pids = joblib.load('df_data/val1_pids.pkl')

user_seen = train.groupby('pid').tid.apply(set).to_dict()
val_tracks = val.groupby('pid').tid.apply(set).to_dict()

config = {
    'num_playlists': df_playlists_test_info.pid.max() + 1,
    'num_tracks': df_tracks.tid.max() + 1,
}

X_train = sp.coo_matrix(
    (np.ones(len(train)), (train.pid, train.tid)),
    shape=(config['num_playlists'], config['num_tracks'])
)

#config['model_path'] = 'models/lightfm_model.pkl'

val1_pids = val1_pids.astype(np.int32)

model = LightFM(no_components=200, loss='warp', learning_rate=0.02, max_sampled=400, random_state=1, user_alpha=1e-05)

print('training lightfm model')

best_score = 0
for i in range(60):

    model.fit_partial(X_train, epochs=5, num_threads=50)

    model.batch_setup(
        item_chunks={0: np.arange(config['num_tracks'])},
        n_process=50,
    )
    res = model.batch_predict(chunk_id=0, user_ids=val1_pids, top_k=600)
    model.batch_cleanup()

    score = []
    for pid in val1_pids:
        tracks_t = val_tracks[pid]
        tracks = [i for i in res[pid][0] if i not in user_seen.get(pid, set())][:len(tracks_t)]
        guess = np.sum([i in tracks_t for i in tracks])
        score.append(guess / len(tracks_t))

    score = np.mean(score)
    print(score)
    if score > best_score:
#        joblib.dump(model, open(config['model_path'], 'wb'))
        model1 = model
        best_score = score

model = model1

# lightfm_text

user_seen = train.groupby('pid').tid.apply(set).to_dict()
val_tracks = val.groupby('pid').tid.apply(set).to_dict()

zeros_pids = np.array(list(set(val1_pids).difference(train.pid.unique())))
no_zeros_pids = np.array(list(set(val1_pids).difference(zeros_pids))[:1000])
target_pids = np.hstack([zeros_pids, no_zeros_pids])

playlist_name1 = df_playlists_test_info.set_index('pid').name
playlist_name2 = df_playlists_info.set_index('pid').name
playlist_name = pd.concat([playlist_name1, playlist_name2]).sort_index()
playlist_name = playlist_name.reindex(np.arange(config['num_playlists'])).fillna('')

vectorizer = CountVectorizer(max_features=20000)
user_features = vectorizer.fit_transform(playlist_name)

user_features = sp.hstack([sp.eye(config['num_playlists']), user_features])

config['model_path'] = 'models/lightfm_model_text.pkl'

X_train = sp.coo_matrix(
    (np.ones(len(train)), (train.pid, train.tid)),
    shape=(config['num_playlists'], config['num_tracks'])
)

model = LightFM(
    no_components=200,
    loss='warp',
    learning_rate=0.03,
    max_sampled=400,
    random_state=1,
    user_alpha=1e-05,
)

best_score = 0

target_pids = target_pids.astype(np.int32)
print('training lightfm_text model')

for i in range(10):

    model.fit_partial(X_train, epochs=5, num_threads=50, user_features=user_features)

    model.batch_setup(
        item_chunks={0: np.arange(config['num_tracks'])},
        n_process=50,
        user_features=user_features,
    )
    res = model.batch_predict(chunk_id=0, user_ids=target_pids, top_k=600)
    model.batch_cleanup()

    score = []
    score2 = []

    for pid in zeros_pids:
        tracks_t = val_tracks[pid]
        tracks = [i for i in res[pid][0] if i not in user_seen.get(pid, set())][:len(tracks_t)]
        guess = np.sum([i in tracks_t for i in tracks])
        score.append(guess / len(tracks_t))

    for pid in no_zeros_pids:
        tracks_t = val_tracks[pid]
        tracks = [i for i in res[pid][0] if i not in user_seen.get(pid, set())][:len(tracks_t)]
        guess = np.sum([i in tracks_t for i in tracks])
        score2.append(guess / len(tracks_t))

    score = np.mean(score)
    score2 = np.mean(score2)

    print(score, score2)
    if score > best_score:
        # joblib.dump(model, open(config['model_path'], 'wb'))
        model_text = model
        best_score = score

    # joblib.dump(user_features, open('models/user_features.pkl', 'wb'))

# candidate selection
train=pd.read_hdf('df_data/train.hdf')
val2 = pd.read_hdf('df_data/val2.hdf')
val2_pids = joblib.load('df_data/val2_pids.pkl')

user_seen = set(zip(train.pid, train.tid))

print('candidate selection stage')
save_candidates(
    val1_pids,
    val1.pid.value_counts(),
    'df_data/ii_candidate.hdf',
    val1
)

save_candidates(
    val2_pids,
    val2.pid.value_counts(),
    'df_data/iii_candidate.hdf',
    val2
)

save_candidates(
    df_playlists_test_info.pid.values,
    df_playlists_test_info.set_index('pid').num_holdouts,
    'df_data/test_candidate.hdf'
)

# lightfm_features

print('create lightfm_features stage')
_user_repr, _user_repr_biases = _precompute_representation(
    features=user_features,
    feature_embeddings=model_text.user_embeddings,
    feature_biases=model_text.user_biases,
)

train = pd.read_hdf('df_data/ii_candidate.hdf')
val = pd.read_hdf('df_data/iii_candidate.hdf')
test = pd.read_hdf('df_data/test_candidate.hdf')

# _,test=train_test_split(test, test_size=0.0001,random_state=0)

create_lightfm_features(train)
create_lightfm_features(val)
create_lightfm_features(test)

train.to_hdf('df_data/ii_lightfm_features.hdf', key='abc')
val.to_hdf('df_data/iii_lightfm_features.hdf', key='abc')
test.to_hdf('df_data/test_lightfm_features.hdf', key='abc')

data = pd.read_hdf('df_data/train.hdf')
data = data.drop_duplicates(['pid', 'tid'])
num_items = data.tid.max() + 1
num_users = data.pid.max() + 1

print('create co_occurence features stage')

co_occurence = [defaultdict(int) for i in range(num_items)]
occurence = [0 for i in range(num_items)]
for q, (_, df) in enumerate(data.groupby('pid')):
    if q % 100000 == 0:
        print(q / 10000)
    tids = list(df.tid)
    for i in tids:
        occurence[i] += 1
    for k, i in enumerate(tids):
        for j in tids[k + 1:]:
            co_occurence[i][j] += 1
            co_occurence[j][i] += 1

train = pd.read_hdf('df_data/ii_candidate.hdf')
val = pd.read_hdf('df_data/iii_candidate.hdf')
test = pd.read_hdf('df_data/test_candidate.hdf')
# _,test=train_test_split(test, test_size=0.0001,random_state=0)


create_co_occurence_features(train)
create_co_occurence_features(val)
create_co_occurence_features(test)

train.to_hdf('df_data/ii_co_occurence_features.hdf', key='abc')
val.to_hdf('df_data/iii_co_occurence_features.hdf', key='abc')
test.to_hdf('df_data/test_co_occurence_features.hdf', key='abc')

data = pd.read_hdf('df_data/train.hdf')
data = data.drop_duplicates(['pid', 'tid'])
num_items = data.tid.max() + 1
num_users = data.pid.max() + 1

co_occurence = [defaultdict(int) for i in range(num_items)]
occurence = [0 for i in range(num_items)]
for q, (_, df) in enumerate(data.groupby('pid')):
    if q % 100000 == 0:
        print(q / 10000)
    tids = list(df.tid)
    for i in tids:
        occurence[i] += 1
    for k, i in enumerate(tids):
        for j in tids[k + 1:]:
            co_occurence[i][j] += 1
            co_occurence[j][i] += 1

# xgboost

print('trainig xgboost model')
pd.options.display.max_columns = 100
data = pd.read_hdf('df_data/train.hdf')
df_playlists_info = pd.read_hdf('df_data/df_playlists_info.hdf')
df_playlists_test_info = pd.read_hdf('df_data/df_playlists_test_info.hdf')
tracks_info = pd.read_hdf('df_data/df_tracks.hdf')

#############################
# _,test_data=train_test_split(df_playlists_test_info, test_size=0.03,random_state=0)
# df_playlists_test_info=test_data
# df_playlists_test_info=df_playlists_test_info[((df_playlists_test_info.num_tracks!=242) & (df_playlists_test_info.num_tracks!=185) & (df_playlists_test_info.num_tracks!=220) & (df_playlists_test_info.num_tracks!=238) & (df_playlists_test_info.num_tracks!=195) & (df_playlists_test_info.num_tracks!=189) & (df_playlists_test_info.num_tracks!=157) & (df_playlists_test_info.num_tracks!=175)& (df_playlists_test_info.num_tracks!=25) & (df_playlists_test_info.num_tracks!=215) & (df_playlists_test_info.num_tracks!=45)&(df_playlists_test_info.num_tracks!=161)& (df_playlists_test_info.num_tracks!=69) & (df_playlists_test_info.num_tracks!=57) & (df_playlists_test_info.num_tracks!=105)&(df_playlists_test_info.num_tracks!=39) & (df_playlists_test_info.num_tracks!=127)& (df_playlists_test_info.num_tracks!=151)&(df_playlists_test_info.num_tracks!=60)&(df_playlists_test_info.num_tracks!=49)&(df_playlists_test_info.num_tracks!=65)&(df_playlists_test_info.num_tracks!=73)&(df_playlists_test_info.num_tracks!=150)&(df_playlists_test_info.num_tracks!=86)&(df_playlists_test_info.num_tracks!=160)&(df_playlists_test_info.num_tracks!=187)&(df_playlists_test_info.num_tracks!=83)&(df_playlists_test_info.num_tracks!=197)&(df_playlists_test_info.num_tracks!=84)&(df_playlists_test_info.num_tracks!=145)&(df_playlists_test_info.num_tracks!=67)&(df_playlists_test_info.num_tracks!=135)&(df_playlists_test_info.num_tracks!=129)&(df_playlists_test_info.num_tracks!=123)&(df_playlists_test_info.num_tracks!=155)&(df_playlists_test_info.num_tracks!=153)&(df_playlists_test_info.num_tracks!=154)&(df_playlists_test_info.num_tracks!=210)&(df_playlists_test_info.num_tracks!=159)&(df_playlists_test_info.num_tracks!=107)&(df_playlists_test_info.num_tracks!=97)&(df_playlists_test_info.num_tracks!=95)&(df_playlists_test_info.num_tracks!=106)&(df_playlists_test_info.num_tracks!=43)&(df_playlists_test_info.num_tracks!=239)&(df_playlists_test_info.num_tracks!=202)&(df_playlists_test_info.num_tracks!=166)&(df_playlists_test_info.num_tracks!=182)&(df_playlists_test_info.num_tracks!=232)&(df_playlists_test_info.num_tracks!=165)&(df_playlists_test_info.num_tracks!=167)&(df_playlists_test_info.num_tracks!=168)&(df_playlists_test_info.num_tracks!=243)&(df_playlists_test_info.num_tracks!=164)&(df_playlists_test_info.num_tracks!=163)&(df_playlists_test_info.num_tracks!=172)&(df_playlists_test_info.num_tracks!=193)&(df_playlists_test_info.num_tracks!=191)&(df_playlists_test_info.num_tracks!=190)&(df_playlists_test_info.num_tracks!=156)&(df_playlists_test_info.num_tracks!=144)&(df_playlists_test_info.num_tracks!=181)&(df_playlists_test_info.num_tracks!=184)&(df_playlists_test_info.num_tracks!=199)&(df_playlists_test_info.num_tracks!=204)&(df_playlists_test_info.num_tracks!=217))]
# df_playlists_test_info=df_playlists_test_info[((df_playlists_test_info.num_tracks!=222)&(df_playlists_test_info.num_tracks!=226)&(df_playlists_test_info.num_tracks!=228)&(df_playlists_test_info.num_tracks!=125))]
####################################

tracks_info['album'] = LabelEncoder().fit_transform(tracks_info.album_uri)
tracks_info['artist'] = LabelEncoder().fit_transform(tracks_info.artist_uri)

train = pd.read_hdf('df_data/ii_candidate.hdf')
val = pd.read_hdf('df_data/iii_candidate.hdf')
test = pd.read_hdf('df_data/test_candidate.hdf')

# _,test=train_test_split(test, test_size=0.0001,random_state=0)

train_holdouts = pd.read_hdf('df_data/val1.hdf')
val_holdouts = pd.read_hdf('df_data/val2.hdf')

train_length = train_holdouts.groupby('pid').tid.nunique()
val_length = val_holdouts.groupby('pid').tid.nunique()
test_length = df_playlists_test_info.set_index('pid').num_holdouts
num_items = data.tid.max() + 1

create_features(train, train_length)
create_features(val, val_length)
create_features(test, test_length)

train_co = pd.read_hdf('df_data/ii_co_occurence_features.hdf').drop('target', axis=1)
val_co = pd.read_hdf('df_data/iii_co_occurence_features.hdf').drop('target', axis=1)
test_co = pd.read_hdf('df_data/test_co_occurence_features.hdf')

train_lightfm = pd.read_hdf('df_data/ii_lightfm_features.hdf').drop('target', axis=1)
val_lightfm = pd.read_hdf('df_data/iii_lightfm_features.hdf').drop('target', axis=1)
test_lightfm = pd.read_hdf('df_data/test_lightfm_features.hdf')

train = train.merge(train_co, on=['pid', 'tid'])
val = val.merge(val_co, on=['pid', 'tid'])
test = test.merge(test_co, on=['pid', 'tid'])

train = train.merge(train_lightfm, on=['pid', 'tid'])
val = val.merge(val_lightfm, on=['pid', 'tid'])
test = test.merge(test_lightfm, on=['pid', 'tid'])

cols = ['pid', 'tid', 'target']
xgtrain = xgboost.DMatrix(train.drop(cols, axis=1), train.target)

xgval = xgboost.DMatrix(val.drop(cols, axis=1), val.target)
xgtest = xgboost.DMatrix(test.drop(['pid', 'tid'], axis=1))
params = {
    'objective': 'binary:logistic',
    'eta': 0.1,
    'booster': 'gbtree',
    'max_depth': 7,
    'nthread': 50,
    'seed': 1,
    'eval_metric': 'auc',
}

a = xgboost.train(
    params=list(params.items()),
    early_stopping_rounds=30,
    verbose_eval=10,
    dtrain=xgtrain,
    evals=[(xgtrain, 'train'), (xgval, 'test')],
    num_boost_round=300,
)

p = a.predict(xgval)
val['p'] = p

scores = []
for pid, df, in val.sort_values('p', ascending=False).groupby('pid'):
    n = val_length[pid]
    scores.append(df[:n].target.sum() / n)
np.mean(scores)

test['p'] = a.predict(xgtest)
test = test.sort_values(['pid', 'p'], ascending=[True, False])
recs = test.groupby('pid').tid.apply(lambda x: x.values[:500])
track_uri = tracks_info.track_uri

sabmission = open('submission.csv', 'w')
sabmission.write('team_info,main,Aloui,aloui@eurecom.fr\n')

for pid, tids in recs.items():
    sabmission.write('{}, '.format(pid) + ', '.join(track_uri.loc[tids].values) + '\n')

sabmission.close()

print('end')
