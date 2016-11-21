import tweepy
import io

consumer_key='b6qtNWS37CRzXrDJEj3BPOKzk'
consumer_secret='7RLPKjjYRcvRhipPeZNZz6Gp5lVzKUCo7tyTlH22yLzjtacRdy'
access_token='788260604964253696-aII3vdKFL1ETwPYUvIUYXK1zluukQ5H'
access_secret='g2HTfrq7oy3eCfXSNX2lDmgf9oVr80uFFhNdepLM2hw6c'

def get_api():
    auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_secret)
    return tweepy.API(auth)

def get_text(api,id):
    text='#'
    try:
        status=api.get_status(id)
        text=status.text
    except tweepy.error.TweepError as e:
        text=e.message[0]['message']
    return text

#extract text data from id file
def extract_text(api,id_file,out_file):
    fout=io.open(out_file,'w',encoding='utf8')
    with open(id_file) as fin:
        lines=fin.readlines()
        for line in lines:
            ids=line.split()
            for id in ids:
                text=get_text(api,id)
                fout.write(text+u'\t')
            fout.write(u'\n')
    fout.close()

if __name__=='__main__':
    api=get_api()
    extract_text(api,'./TweetIDs/TweetIDs_Train.txt','train_text.txt')