INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file lastfm.inter comprising the weight of users over the artists.
Each record/line in the file has the following fields: user_id, artist_id, weight, tag_value

user_id: the id of the user, and its type is token.
artist_id: the id of the artist, and its type is token.
weight: the listening count for each [user, artist] pair, and its type is float.
tag_value: the value of tags, and its type is token_seq.

ARTIST DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file lastfm.item comprising the information of the artists.
Each record/line in the file has the following fields: id, name, url, picture_url

id: the id of the artist, and its type is token.
name: the name of the artist, and its type is token.
url: the url of the artist, and its type is token.
picture_url: the picture url of the artist, and its type is token.