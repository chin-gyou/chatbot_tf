import os
import re
import xml.dom.minidom
import gzip

def remove_parenthesis(str):
    regex = re.compile('\(.+?\)')
    return regex.sub('', str)

def switchboard_extract(fin,ftarget):
    with open(ftarget,'a') as fout:
        with open(fin) as f:
            lines=f.readlines()
            for line in lines[17:]:
                act=line.split(':')
                if len(act)>=2:
                    fout.write(act[1])
        fout.write('\n')

def blog_diag_extract(fin,ftarget):
    with open(ftarget,'a') as fout:
        with open(fin) as f:
            lines=f.readlines()
            for line in lines:
                act=line.split(':')
                if len(act)>=2:
                    fout.write(act[1])
        fout.write('\n')

def extract_switchboard(root,output):
    for p in os.listdir(root):
        for disc in os.listdir(os.path.join(root,p)):
            for f in os.listdir(os.path.join(root,p, disc)):
                if f.endswith('txt'):
                    switchboard_extract(os.path.join(root,p,disc,f),output)

def stat(targetfile):
    with open(targetfile) as f:
        lines=f.readlines()
        print('Number of dialogues: ',len([e for e in lines if e=='\n']))
        print('Number of utterances: ', len([e for e in lines if e != '\n']))

def extract_blog_diag(root,output):
    for f in os.listdir(os.path.join(root)):
        if f.endswith('txt'):
            blog_diag_extract(os.path.join(root,f), output)

def extract_cornell(ftarget,output):
    last=666255
    with open(output,'w') as fout:
        with open(ftarget,errors='ignore') as f:
            lines=f.readlines()
            for line in reversed(lines):
                s=line.split('+++$+++')
                this=int(s[0][1:])
                if this!=last+1:
                    fout.write('\n')
                fout.write(s[-1].strip()+'\n')
                last=this

def extract_iemocap(root,output):
    with open(output,'a') as fout:
        for session in os.listdir(root):
            if session.startswith('Session'):
                for trans in os.listdir(os.path.join(root,session,'dialog/transcriptions')):
                    with open(os.path.join(root,session,'dialog/transcriptions',trans)) as fin:
                        lines=fin.readlines()
                        last_speaker=''
                        for line in lines:
                            s=line.split(':')
                            speaker=s[0].split('_')[-1][0]
                            if speaker!=last_speaker:
                                fout.write('\n')
                            else:
                                fout.write(' ')
                            words=[w.strip() for w in s[1:]]
                            fout.write(' '.join(words))
                            last_speaker=speaker
                    fout.write('\n')

def opensubtitle_extract(ftarget,output):
    with open(output,'a') as fout:
        DOMTree = xml.dom.minidom.parse(ftarget)
        collection = DOMTree.documentElement
        sentences=collection.getElementsByTagName("s")
        for sentence in sentences:
            words=sentence.getElementsByTagName('w')
            for w in words:
                word=w.firstChild.data
                fout.write(word+' ')
            fout.write('\n')
        fout.write('\n')

def extract_opensubtitle(root,output):
    i = 0
    for r,dir,f in os.walk(root):
        for filename in f:
            if filename.endswith('gz'):
                i+=1
                print(filename)
                opensubtitle_extract(gzip.open(os.path.join(r,filename)),output)
    print(i)

def extract_imdb(ftarget,output):
    fout=open(output,'a')
    a,b=None,None
    with open(ftarget) as f:
        lines=f.readlines()
        for line in lines:
            line=remove_parenthesis(line)
            if line!='':
                if line.isupper():
                    pass
                else:
                    if a==None or b==None:
                        pass
                    else:
                        fout.wr

if __name__=='__main__':
    #extract_switchboard('/projects/corpora/switchboard/trans','./switchboard.txt')
    #extract_opensubtitle('../OpenSUbtitles/en','./data/opensubtitles.txt')
    #extract_iemocap('../IEMOCAP_full_release','./data/iemocap.txt')
    stat('./data/opensubtitles.txt')
    #extract_cornell('./cornell movie-dialogs corpus/movie_lines.txt','./data/cornell_movie.txt')
    #extract_blog_diag('./blog_dialogs_no_gesture','./data/blog_dialog.txt')