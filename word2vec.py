from gensim.models import Word2Vec
import numpy as np
import pickle

class word2vec:
    def __init__(self, vecfile):
        self.oov=0
        self.pretrained=Word2Vec.load_word2vec_format(vecfile,binary=True)

    # given a word, return the embedding and index
    # if not found, return a random normal distributed vector
    def word_to_vec(self,word):
        if word in self.pretrained.vocab:
            return self.pretrained[word]
        else:
            self.oov+=1
            return np.random.normal(0, 1, 300)

    # given a word dict, return the embedding matrix
    # For word dict, index ranges from 1 to vocab_size, including <unknown>
    # When creating matrix, the 0-th vector is for class 0 when padding
    def embedding_matrix(self,dicts):
        embed=np.zeros(shape=(len(dicts)+1,300),dtype=np.float32)
        for w,i,_,_ in dicts:
            print(w)
            embed[i+1]=self.word_to_vec(w.lower())
        return embed

    # save the embedding matrix to fout
    def save_wordvecs(self,dicts,fout):
        embed=self.embedding_matrix(dicts)
        with open(fout,'wb') as f:
            pickle.dump(embed,f)

if __name__=='__main__':
    model=word2vec('./data/GoogleNews-vectors-negative300.bin')
    with open('data/UbuntuDialogueCorpus/Dataset.dict.pkl','rb') as f:
        dicts=pickle.load(f)
        model.save_wordvecs(dicts,'embedding.mat')
        print(model.oov)