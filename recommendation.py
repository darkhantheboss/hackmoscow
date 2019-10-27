import requests
from bs4 import BeautifulSoup
from sklearn.neighbors.dist_metrics import DistanceMetric

SPOTIFY_API_URL = 'https://api.spotify.com/v1'
SPOTIFY_TOKEN = 'Bearer BQA_AITdC2wIE2yEaEp7NJTn4jKr2akegB8WA3l3ebfWtVmnoe_nUqWUz5Q0C580xzn_9xSQodJQSDztxkHvUCshNv' \
                '4-O2i3bNN_AJ2aGRIcy3YQz2eYbva7sszn0UK3i6F3262__nb07WZr-km02xHU3Djeu-ggrDhraw'


def parse_tracks_from_source(source, link):
    tracks = []
    if source == 'youtube':
        if link:
            source_code = requests.get(link).text
            soup = BeautifulSoup(source_code, 'html.parser')
            for link in soup.find_all("a", {"dir": "ltr"}):
                if 'channel' not in link.get('href'):
                    tracks.append(link.text.replace('\n', '').replace('    ', '').replace('  ', ''))
            return tracks
        else:
            return ['bad_guy', 'someone like you']
    else:
        if link:
            source_code = requests.get(link).text
            soup = BeautifulSoup(source_code, 'html.parser')
            for link in soup.find_all("a", {"class": "d-track__title deco-link deco-link_stronger"}):
                tracks.append(link.text)
            return tracks
        return ['not afraid', 'mockingbird']


def get_track_ids(tracks):
    track_ids = []
    artist_ids = []
    for track in tracks:
        track_data = requests.get(SPOTIFY_API_URL + '/search', params=dict(q=track, type='track,artist', limit=1),
                                  headers={'Accept': 'application/json', 'Authorization': SPOTIFY_TOKEN,
                                           'Content-Type': 'application/json'}).json()
        if track_data and track_data.get('tracks') and track_data['tracks'].get('items') and \
                len(track_data['tracks']['items']):
            track_ids.append(track_data['tracks']['items'][0]['id'])
            artist_ids.append(track_data['tracks']['items'][0]['album']['artists'][0]['id'])
    return set(track_ids), set(artist_ids)


def recommend(track_ids, artist_ids):
    attributes_api_endpoint = SPOTIFY_API_URL + "/audio-features?ids=" + ",".join(track_ids)
    attributes_response = requests.get(attributes_api_endpoint, headers={'Authorization': SPOTIFY_TOKEN})
    attributes = attributes_response.json()['audio_features']

    phantom_average_track = {}
    target_attributes = ['energy', 'liveness', 'tempo', 'speechiness', 'acousticness', 'instrumentalness',
                         'danceability', 'loudness']
    for attribute in target_attributes:
        track_sums = 0
        track_count = 0
        for track in attributes:
            if track:
                track_sums += track[attribute]
                track_count += 1
        phantom_average_track[attribute] = track_sums / track_count
    target_energy = str(round(phantom_average_track['energy'], 2))
    target_liveness = str(round(phantom_average_track['liveness'], 2))
    target_tempo = str(round(phantom_average_track['tempo'], 2))
    target_speechiness = str(round(phantom_average_track['speechiness'], 2))
    target_acousticness = str(round(phantom_average_track['acousticness'], 2))
    target_instrumentalness = str(round(phantom_average_track['instrumentalness'], 2))
    target_danceability = str(round(phantom_average_track['danceability'], 2))
    target_loudness = str(round(phantom_average_track['loudness'], 2))
    recommendations_api_endpoint = SPOTIFY_API_URL + "/recommendations?seed_artists=" + ",".join(list(artist_ids)[:5])+\
                                   "&target_energy=" + target_energy + "&target_liveness=" + target_liveness + \
                                   "&target_tempo=" + target_tempo + "&target_speechiness=" + target_speechiness + \
                                   "&target_acousticness=" + target_acousticness + "&target_instrumentalness=" + \
                                   target_instrumentalness + "&target_danceability=" + target_danceability + \
                                   "&target_loudness=" + target_loudness + "&limit=20"
    recommendations_response = requests.get(recommendations_api_endpoint, headers={'Authorization': SPOTIFY_TOKEN})
    recommendation_data = dict(ids=[], data=[], artists=[], images=[], titles=[], attributes=[])
    recommendation_data['data'] = recommendations_response.json()['tracks']
    for track in recommendation_data['data']:
        if track['id'] not in recommendation_data['ids'] and track['id'] not in track_ids:
            recommendation_data['titles'].append(track['name'])
            recommendation_data['artists'].append(track['artists'][0]['name'])
            recommendation_data['ids'].append(track['id'])
            recommendation_data['images'].append(track['album']['images'][0]['url'])
    attributes_api_endpoint = SPOTIFY_API_URL + "/audio-features?ids=" + ",".join(recommendation_data['ids'])
    attributes_response = requests.get(attributes_api_endpoint, headers={'Authorization': SPOTIFY_TOKEN})
    recommendation_data['attributes'] = attributes_response.json()['audio_features']
    recommendation_track_attributes = []
    for track in recommendation_data['attributes']:
        track_float_values = {}
        for attribute in target_attributes:
            track_float_values[attribute] = track[attribute]
        recommendation_track_attributes.append(track_float_values.values())
    recommendation_distances = [phantom_average_track.values()] + recommendation_track_attributes
    # print phantom_average_track.values()
    # print "------"
    # print recommendation_track_attributes
    dist = DistanceMetric.get_metric('euclidean')
    distances = dist.pairwise(recommendation_distances)[0]
    recommendation_data['distances'] = distances[1:len(distances)]

    sorted_recommendation_indexes = recommendation_data['distances'].argsort()[:len(recommendation_data['distances'])]
    for key in recommendation_data.keys():
        if recommendation_data[key] != []:
            recommendation_data[key] = [recommendation_data[key][i] for i in sorted_recommendation_indexes]
    return recommendation_data


def get_recommendation(tracks):
    track_ids, artist_ids = get_track_ids(tracks)
    resp = recommend(track_ids, artist_ids)
    return resp['titles'], resp['images'], resp['distances']
