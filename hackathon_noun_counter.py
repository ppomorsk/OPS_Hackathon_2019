
import nltk
import pandas as pd

def process_law_text(input_text):
    text_law=input_text.replace("\x92","'").replace("\x96","-").replace('\x93','"').replace("\x94",'"')
    tokens = nltk.word_tokenize(text_law)
    tokens_tagged = nltk.pos_tag(tokens)
    tokens_tagged_nouns=list(filter(  lambda x: x[1][0]=="N" and x[1][1]=="N", tokens_tagged))
    nouns = list( map(lambda x: x[0],tokens_tagged_nouns) )
    nouns_lower= list(map(lambda x: x.lower(), nouns))
    nouns_unique = list(set(nouns_lower))
    nouns_unique.sort()
    return nouns_unique


import time
start=time.time()
df1 = pd.read_csv("elawsCorpus-output.csv",encoding='latin-1')
df1.drop(df1.index[5:],inplace=True)
df1["noun_list"]=df1["Text"].apply(lambda x: process_law_text(x))
df1.drop(columns=["Text","hyperlink"], inplace=True)
df1.rename(index=str, columns={"Unnamed: 0":"law"},inplace=True)

df2 = pd.DataFrame(df1.noun_list.tolist(), index=df1.law)

df3=df2.stack().reset_index(level=1, drop=True).reset_index().rename(columns={0:"noun_list"})

df3["law"]=df3["law"].apply(lambda x: [x,])

df4=df3.groupby("noun_list").agg({"law":"sum"})
end=time.time()
print("time= ",end-start)


