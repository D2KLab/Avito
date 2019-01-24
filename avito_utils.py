import os
import joblib
import json
import numpy as np
import pandas as pd


# os.mkdir('df_data')


def create_df_data(path_set, parh_challenge_set):
    # create a directory in the current direct
    if not os.path.exists('df_data'):
        os.makedirs('df_data')
    print(path_set)

    path = path_set

    playlist_col = ['collaborative', 'duration_ms', 'modified_at',
                    'name', 'num_albums', 'num_artists', 'num_edits',
                    'num_followers', 'num_tracks', 'pid']
    tracks_col = ['album_name', 'album_uri', 'artist_name', 'artist_uri',
                  'duration_ms', 'track_name', 'track_uri']
    playlist_test_col = ['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']

    filenames = os.listdir(path)

    data_playlists = []
    data_tracks = []
    playlists = []

    tracks = set()

    for filename in filenames:
        fullpath = os.sep.join((path, filename))
        f = open(fullpath)
        js = f.read()
        f.close()

        mpd_slice = json.loads(js)

        for playlist in mpd_slice['playlists']:
            data_playlists.append([playlist[col] for col in playlist_col])
            for track in playlist['tracks']:
                playlists.append([playlist['pid'], track['track_uri'], track['pos']])
                if track['track_uri'] not in tracks:
                    data_tracks.append([track[col] for col in tracks_col])
                    tracks.add(track['track_uri'])

    f = open(parh_challenge_set)
    js = f.read()
    f.close()
    mpd_slice = json.loads(js)

    data_playlists_test = []
    playlists_test = []

    for playlist in mpd_slice['playlists']:
        data_playlists_test.append([playlist.get(col, '') for col in playlist_test_col])
        for track in playlist['tracks']:
            playlists_test.append([playlist['pid'], track['track_uri'], track['pos']])
            if track['track_uri'] not in tracks:
                data_tracks.append([track[col] for col in tracks_col])
                tracks.add(track['track_uri'])

    df_playlists_info = pd.DataFrame(data_playlists, columns=playlist_col)
    df_playlists_info['collaborative'] = df_playlists_info['collaborative'].map({'false': False, 'true': True})

    df_tracks = pd.DataFrame(data_tracks, columns=tracks_col)
    df_tracks['tid'] = df_tracks.index

    track_uri2tid = df_tracks.set_index('track_uri').tid

    df_playlists = pd.DataFrame(playlists, columns=['pid', 'tid', 'pos'])
    df_playlists.tid = df_playlists.tid.map(track_uri2tid)

    df_playlists_test_info = pd.DataFrame(data_playlists_test, columns=playlist_test_col)

    df_playlists_test = pd.DataFrame(playlists_test, columns=['pid', 'tid', 'pos'])
    df_playlists_test.tid = df_playlists_test.tid.map(track_uri2tid)

    df_tracks.to_hdf('df_data/df_tracks.hdf', key='abc')
    df_playlists.to_hdf('df_data/df_playlists.hdf', key='abc')
    df_playlists_info.to_hdf('df_data/df_playlists_info.hdf', key='abc')
    df_playlists_test.to_hdf('df_data/df_playlists_test.hdf', key='abc')
    df_playlists_test_info.to_hdf('df_data/df_playlists_test_info.hdf', key='abc')

#model=joblib.load(open('models/lightfm_model.pkl','rb'))
#model_text=joblib.load(open('models/lightfm_model_text.pkl','rb'))
#user_features=joblib.load(open('models/user_features.pkl','rb'))
def save_candidates(target_pids, df_size, file_name, df=None):
    target_pids = target_pids.astype(np.int32)
    target_pids_text = list(set(target_pids).difference(train.pid))
    target_pids_no_text = list(set(target_pids).difference(target_pids_text))

    model.batch_setup(
        item_chunks={0: np.arange(df_tracks.tid.max() + 1)},
        n_process=50,
    )
    res = model.batch_predict(chunk_id=0, user_ids=target_pids_no_text, top_k=10000)
    model.batch_cleanup()

    model_text.batch_setup(
        item_chunks={0: np.arange(df_tracks.tid.max() + 1)},
        n_process=50,
        user_features=user_features,
    )
    res2 = model_text.batch_predict(chunk_id=0, user_ids=target_pids_text, top_k=10000)
    model_text.batch_cleanup()

    res.update(res2)

    if df is not None:
        val_tracks = df.groupby('pid').tid.apply(set).to_dict()

    pids = []
    tids = []
    targets = []

    for pid in target_pids:

        l = max(df_size[pid] * 15, 700 + df_size[pid])
        # l = 2000
        pids += [pid] * l
        tids += list(res[pid][0][:l])

        if df is not None:
            tracks_t = val_tracks[pid]
            targets += [i in tracks_t for i in res[pid][0][:l]]

    candidates = pd.DataFrame()
    candidates['pid'] = np.array(pids)
    candidates['tid'] = np.array(tids)

    if df is not None:
        candidates['target'] = np.array(targets).astype(int)

    index = []
    for pid, tid in candidates[['pid', 'tid']].values:
        index.append((pid, tid) not in user_seen)

    candidates = candidates[index]

    candidates.to_hdf(file_name, key='abc')


def create_lightfm_features(df):
    df['pid_bias'] = model.user_biases[df.pid]
    df['tid_bias'] = model.item_biases[df.tid]

    pid_embeddings = model.user_embeddings[df.pid]
    tid_embeddings = model.item_embeddings[df.tid]

    df['lightfm_dot_product'] = (pid_embeddings * tid_embeddings).sum(axis=1)
    df['lightfm_prediction'] = df['lightfm_dot_product'] + df['pid_bias'] + df['tid_bias']

    df['lightfm_rank'] = df.groupby('pid').lightfm_prediction.rank(ascending=False)

    df['pid_bias_text'] = _user_repr_biases[df.pid]
    df['tid_bias_text'] = model_text.item_biases[df.tid]

    pid_embeddings = _user_repr[df.pid]
    tid_embeddings = model_text.item_embeddings[df.tid]

    df['lightfm_dot_product_text'] = (pid_embeddings * tid_embeddings).sum(axis=1)
    df['lightfm_prediction_text'] = df['lightfm_dot_product_text'] + df['pid_bias_text'] + df['tid_bias_text']

    df['lightfm_rank_text'] = df.groupby('pid').lightfm_prediction_text.rank(ascending=False)


def get_f(i, f):
    if len(i) == 0:
        return -1
    else:
        return f(i)


def create_co_occurence_features(df):
    pids = df.pid.unique()
    seed = data[data.pid.isin(pids)]
    tid_seed = seed.groupby('pid').tid.apply(list)

    co_occurence_seq = []
    for pid, tid in df[['pid', 'tid']].values:
        tracks = tid_seed.get(pid, [])
        co_occurence_seq.append(np.array([co_occurence[tid][i] for i in tracks]))

    df['co_occurence_max'] = [get_f(i, np.max) for i in co_occurence_seq]
    df['co_occurence_min'] = [get_f(i, np.min) for i in co_occurence_seq]
    df['co_occurence_mean'] = [get_f(i, np.mean) for i in co_occurence_seq]
    df['co_occurence_median'] = [get_f(i, np.median) for i in co_occurence_seq]

    co_occurence_seq = []
    for pid, tid in df[['pid', 'tid']].values:
        tracks = tid_seed.get(pid, [])
        co_occurence_seq.append(np.array([co_occurence[tid][i] / occurence[i] for i in tracks]))

    df['co_occurence_norm_max'] = [get_f(i, np.max) for i in co_occurence_seq]
    df['co_occurence_norm_min'] = [get_f(i, np.min) for i in co_occurence_seq]
    df['co_occurence_norm_mean'] = [get_f(i, np.mean) for i in co_occurence_seq]
    df['co_occurence_norm_median'] = [get_f(i, np.median) for i in co_occurence_seq]


def create_count(df):
    tid_count = data.tid.value_counts()
    pid_count = data.pid.value_counts()

    df['tid_count'] = df.tid.map(tid_count).fillna(0)
    df['pid_count'] = df.pid.map(pid_count).fillna(0)

    album_count = data.tid.map(tracks_info.album).value_counts()
    artist_count = data.tid.map(tracks_info.artist).value_counts()

    df['album_count'] = df.tid.map(tracks_info.album).map(album_count).fillna(0)
    df['artist_count'] = df.tid.map(tracks_info.artist).map(artist_count).fillna(0)

    album_count


def isin(i, j):
    if j is not np.nan:
        return i in j
    return False


def isin_sum(i, j):
    if j is not np.nan:
        return (i == j).sum()
    return 0


def creaet_artist_features(df):
    data_short = data[data.pid.isin(df.pid)]
    pid_artist = data_short.tid.map(tracks_info.artist).groupby(data_short.pid).apply(np.array)
    df_playlist = df.pid.map(pid_artist)
    df_artist = df.tid.map(tracks_info.artist)

    share_unique = pid_artist.apply(np.unique).apply(len) / pid_artist.apply(len)

    df['share_of_unique_artist'] = df.pid.map(share_unique).fillna(-1)
    df['sim_artist_in_playlist'] = [isin_sum(i, j) for i, j in zip(df_artist, df_playlist)]
    df['mean_artist_in_playlist'] = (df['sim_artist_in_playlist'] / df.pid.map(pid_artist.apply(len))).fillna(-1)


def creaet_album_features(df):
    data_short = data[data.pid.isin(df.pid)]
    pid_album = data_short.tid.map(tracks_info.album).groupby(data_short.pid).apply(np.array)
    df_playlist = df.pid.map(pid_album)
    df_album = df.tid.map(tracks_info.album)

    share_unique = pid_album.apply(np.unique).apply(len) / pid_album.apply(len)

    df['share_of_unique_album'] = df.pid.map(share_unique).fillna(-1)
    df['sim_album_in_playlist'] = [isin_sum(i, j) for i, j in zip(df_album, df_playlist)]
    df['mean_album_in_playlist'] = (df['sim_album_in_playlist'] / df.pid.map(pid_album.apply(len))).fillna(-1)


def create_features(df, df_length):
    create_count(df)
    creaet_artist_features(df)
    creaet_album_features(df)
    df['tracks_holdout'] = df.pid.map(df_length)
