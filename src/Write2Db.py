'''
Created on Mar 15, 2018

@author: abhishek
'''

from pymongo import MongoClient
from _collections import defaultdict
import gridfs

class Write2Db(object):
    
    def __init__(self,name,set):
        
        self.client = MongoClient('localhost', 27017)
        self.db =self.client[name]
        self.document = defaultdict()
        self.collection = self.db[set]
    
    def Insert(self,key,column,weight,zeroweight,matrices):
        matrix_flatten_list = []
        
        for matrix in matrices:
            matrix_flatten_list.append(matrix.flatten().tolist())
        print key
        self.document = {
            'key' : key,
            'column' : int(column),
            'weight' : int(weight),
            'zeroweight': int(zeroweight),
            'matrices' : matrix_flatten_list
        }
        #json_data = json.dumps(self.document)
        if(key != "56_448_0"):
            result=self.db.reviews.insert(self.document)
        else:
            fs = gridfs.GridFS(self.db)
            result= fs.put(self.document, fileName="All1's")
        print("Inserted Key = "+ key,result)

'''
    def Get(self):
        pass

    def Modify(self):
        pass
'''