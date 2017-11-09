## A Conditional Variational Framework for Dialog Generation

### Description
This repository hosts the hierarchical recurrent encoder-decoder with separated context model (SPHRED) and 
the conditional VHRED model for generative dialog modeling as described by Shen and Su et al. 2017

### Creating Datasets
1. Download the original Ubuntu Dialogue Corpus as released by Lowe et al. (2015) which can be found : http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/
2. Create the dictionary from the corpus and Serialize the dicitonary and corpus.   
3. Download Word2Vec trained by GoogleNes: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM.
4. Changing dataproducer.py to generate tfrecord from the serialized corpus text(We use TFRecord for fast and stable training process)


###

### References

    A Conditional Variational Framework for Dialog Generation. Xiaoyu Shen, Hui Su, Yanran Li, Wenjie Li, Shuzi Niu, Yang Zhao, Akiko Aizawa, Guoping Long. 2017. https://arxiv.org/abs/1705.00316
    The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems. Ryan Lowe, Nissan Pow, Iulian Serban, Joelle Pineau. 2015. SIGDIAL. http://arxiv.org/abs/1506.08909.
