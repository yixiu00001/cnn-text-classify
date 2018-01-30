
# coding: utf-8

# In[1]:


from gensim.models import word2vec
import os
import codecs
import numpy as np


# In[2]:

model_dir = "model/"
#model_name="pretrained_keyword_embeddings_new-seg.txt"
model_name="pretrained_keyword_embeddings_new-keyword.txt"

#train_corpus = "datasets/disWordW2VIn.txt"
#train_corpus = "datasets/disWordAndType.txt"
#train_corpus = "datasets/disWordAndTypeV1.txt"
#train_corpus = "disWordAndTypeV2icd10.txt"
#train_corpus = "word2vecDatasets/word2vecTrainCat.txt"
#train_corpus = "word2vecDatasets/word2vecTrainCat20171221.txt"
#train_corpus = "word2vecDatasets/word2vecTrainCatAll.txt"
train_corpus = "word2vecDatasets/word2vecTrainCat.txt"


# In[3]:

window = 10#window = 5
negative = 5
sample = 1e-6
iterNum = 10
vector_size=300#vector_size=100
sg = 1 #skip gram
hs=0#default 0,hierarchical softmax 1,negative sampling
max_vocab_size=None


# In[6]:

def trainWord2Vec(window, negative, sample, iterNum, vector_size, nameIndex):
        #train doc2vec model
    sentences = word2vec.Text8Corpus(train_corpus)
    #skip-gram + hierachical 
    model = word2vec.Word2Vec( sentences, sg=1,hs=1, size=vector_size, window=window, min_count=1, workers=8, iter=iterNum,max_vocab_size=max_vocab_size)
    model.save(model_dir + "word2VecModelsh.bin"+ nameIndex)
    #skip-gram + negative  
    model = word2vec.Word2Vec( sentences, sg=1,negative=5,hs=0, size=vector_size, window=window, min_count=1, workers=8, iter=iterNum,max_vocab_size=max_vocab_size)
    model.save(model_dir+"word2VecModelsn.bin"+nameIndex)
    #cbow +hierachical
    model = word2vec.Word2Vec( sentences, sg=0, hs=1 ,size=vector_size, window=window, min_count=1, workers=8, iter=iterNum,max_vocab_size=max_vocab_size)
    model.save(model_dir + "word2VecModelch.bin"+ nameIndex)
    #cbow+negative
    model = word2vec.Word2Vec( sentences, sg=0, negative=5,hs=0, size=vector_size, window=window, min_count=1, workers=8, iter=iterNum,max_vocab_size=max_vocab_size)
    model.save(model_dir + "word2VecModelcn.bin" + nameIndex)


# In[ ]:

window_size=5 ;negative=5 ;sample = 1e-6 ;iterNum = 10;vector_size=100;
index = 0 
#for window in range(7,10):
for window in [ 5,10, 15,3]:
    index=0
    for vector_size in [ 300, 200, 100 ]:
        for sample in [ 1e-5, 1e-6]:
            for negative in [ 10 , 15]:
                trainWord2Vec(window, negative, sample, iterNum, vector_size,str(window)+"_"+str(vector_size)+"_"+str(sample)+"_"+str(negative));index+=1





# In[ ]:




# In[ ]:

def doGetWordVec(word, m):
    s = ("%s" %(word))
    res = m.most_similar(s.decode("utf8"))
    for item in res:
        print (item[0], item[1])


# In[ ]:

def testModel(m):
    doGetWordVec("重度贫血", m1)  
    doGetWordVec("呼吸衰竭", m1)
    doGetWordVec("妊娠", m1)  
    doGetWordVec("冠心病", m1)
    doGetWordVec("胰岛素", m1)  
    doGetWordVec("前列腺", m1)


# In[8]:


# In[ ]:


def loadData(inName):
    disList = list()
    sentences = word2vec.Text8Corpus(os.path.join(model_dir,inName ))
    

    model = word2vec.Word2Vec(sentences,
                                   size=300,
                                   window=10,
                                   min_count=2,
                                   workers=8,
                                     iter=100)
    
    #model = word2vec.Word2Vec(sentences, size=20)  # 默认window=5\
    
    model.save(os.path.join(model_dir,model_name))
    #y1 = model.similarity(u"不错", u"好")
    #print u"【不错】和【好】的相似度为：", y1
    
    #print("%d %s %s " %(len(disDict), disDict.keys()[0], disDict[disDict.keys()[0]]))
#inName="word2VecIn.txt"
#inName='word2VecIn-seg.txt'
#inName="hospital-Word2VecIn.txt"
inName="disWordW2VIn.txt"
#loadData(inName)


# In[31]:

def doGetWordVec(word, m):
    s = ("%s" %(word))
    print( s)
    res = m.most_similar(s.decode("utf8"))
    for item in res:
        print (item[0], item[1])


# In[10]:


#model = word2vec.Word2Vec.load(os.path.join(model_dir,model_name))


# In[11]:

#inStr = "中度贫血"
tmp = inStr.decode("utf8")
##print type(inStr), type(tmp)
#res = model[tmp]
#print res


# In[12]:

#for w in model.most_similar(u'脑萎缩'):
#for w in model.most_similar(u'食管中下段'):
#    print w[0], w[1]

    


# In[25]:

#print model.similarity(u'中度贫血', u'低蛋白血症')


# In[ ]:



