import findspark
findspark.init('/home/ubuntu/spark-2.4.6-bin-hadoop2.7/')
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import re
import pyarrow
import pandas as pd
import numpy as np
import requests
import json
from SS import *
from pyspark.sql.functions import col
from pyspark.sql import Row
from pyspark.sql import SparkSession

def get_data_from_db(entity_model):
    host = 'https://search-solytics-tzxvnvrvkscgklaedz5lr3iqxu.ap-south-1.es.amazonaws.com/news_test_list/_search'
    json_body = '''{
        "query": {
                "bool": {
                    "must_not": {
                        "exists":{
                            "field":"sentiment_ML"
                            }
                        }
                    }
                }
    }'''
    headers = {
        'Content-Type': 'application/json',
    }
    params = {
        'size':1000
    }
    resp = requests.get(host,params=params, headers=headers, data=json_body)
    resp_text = json.loads(resp.text)
    document_list = []
    # print(resp)
    for data in resp_text['hits']['hits']:
        content_list = {}
        content_list["id"] = data["_id"]
        content_list["ents"] = entity_model.spark_extract(data["_source"]["Content"])
        document_list.append(content_list) 
    return document_list

def write_data_in_db(id,result):
    host = 'https://search-solytics-tzxvnvrvkscgklaedz5lr3iqxu.ap-south-1.es.amazonaws.com/news_test_list/_update/'+str(id)
    headers = {
        'Content-Type': 'application/json',
    }
    post_body = {
        "doc" : {
        "sentiment_ML" : result
    },
    "detect_noop": False
    }

    post_body = json.dumps(post_body)
    response = requests.post(url=host,headers=headers,data=post_body)
    return response



def Structuredata(text):
    iid=[]
    tity=[]
    txt=[]

    print(type(text))
    for i in text:
        #print(i['ents'])
    
        for j in i['ents']:
            iid.append(i['id'])
            tity.append(j['entity'])
            txt.append(j['text'])
            #print(type(i))
    
    #iid
    #tity
    #txt
    schema = StructType([
        StructField("id",StringType(),False),
        StructField("entity",StringType(),False),
        StructField("text",StringType(),False)
    ])

    df=spark.createDataFrame(zip(iid, tity,txt), schema=['id','entity', 'text'])
    df.show()
    df.schema
    return df

'''

def relspark(df):
    def CustomFunc1(row):
        topic_list = []
        text_list = []
        idd=row.id
        tity=row.entity
        x=rel.relevant_text(row.text)#relevant_text
        for i in x:
            topic_list.append(i)
            text_list.append(x[i])
            #rows_list.append(Row(id=idd,entity=tity,topic=topic,text=txt))
        
        #rdd = sc.parallelize(rows_list)
        return Row(id=idd,entity=tity,topic=topic_list,text=text_list)
    rt=df.rdd.map(CustomFunc1)
    #rt.collect()
    return rt
    
'''


def Sentiment_predict(df):
    def CustomFunc(row):
        #print(row)
        idd=row.id
        tity=row.entity
        top=row.topic
        _sent = sent.score(row.text)
        topic_list=[]
        x=sent.score(row.text)
        for topic in x:
            print(topic)
            topic_list.append(topic)
        Sentiment = _sent[0]
        Score=str(_sent[1])
    
        return Row(id=idd,entity=tity,Sentiment=Sentiment,Score=Score,topic=top)
    
    xp=df.rdd.map(CustomFunc)

    schema = StructType([
        StructField("id",StringType(),False),
        StructField("entity",StringType(),False),
        StructField("Sentiment",StringType(),False),
        StructField("Score",StringType(),False),
        StructField("topic" , StringType(),False),
    ])
    x=spark.createDataFrame(xp,schema)
    x = x.withColumn('Score', col('Score').cast('float'))
    x.show()
    return x
#spark = SparkSession.builder.appName('Zeher2').getOrCreate()
spark = SparkSession.builder.appName('Zeher').getOrCreate()
#sc = spark.sparkContext

s3 = s3_manager()
s3.verify_bucket()
s3.download_ner()
entity_model = entity()
text = get_data_from_db(entity_model)

df=Structuredata(text)

#df.show()
final_sentiment = []
final_score = []
manager = s3_manager()
manager.download_learner()
sent = sentiment()
local=False
if local:
    rel = relevancy(manager.download_word2vec())
else:
    rel = relevancy(manager.word2vec_link())
    
pop=df.toPandas()
idd=[]
tity=[]
topic=[]
txt=[]
for row in pop.iterrows(): 
    t=rel.relevant_text(row[1][2])
    for i in t:
        idd.append(row[1][0])
        tity.append(row[1][1])
        topic.append(i)
        txt.append(t[i])
                
final = pd.DataFrame({'id':idd,'entity':tity,'topic':topic,'text':txt})
print(final)

dfp=spark.createDataFrame(final) 
dfp.show()

result=Sentiment_predict(dfp)
result.show()

'''
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.functions import *
from pyspark.sql.types import *
def rellspark(df):
    Schema = StructType([
       StructField("id" , StringType()),
       StructField("entity" , StringType()),
       StructField("topic" , StringType()),
       StructField("text" , StringType())
    ])
    
    @pandas_udf(Schema, functionType=PandasUDFType.GROUPED_MAP)
    def predict_model(df):
        idd=[]
        tity=[]
        topic=[]
        txt=[]
        for row in df.iterrows(): 
            t=rel.relevant_text(row[1][2])
            for i in t:
                idd.append(row[1][0])
                tity=append(row[1][1])
                topic.append(i)
                txt.append(t[i])
                
        final = pd.DataFrame({'id':idd,'entity':tity,'topic':topic,'text':txt})
        return final
    
    drf=df.groupBy('id').apply(predict_model)
    drf.show()
    return drf
op=rellspark(df)
'''

res = [result.toJSON().map(lambda j: json.loads(j)).collect()]
print(res)
