'''
Created on Mar 13, 2018

@author: abhishek
@summary: class which collects the classification of objects
@param PatternObjectList: dictionary of Lists 
'''
from collections import defaultdict

class ListObjects(object):
    
    PatternObjectList = defaultdict(list)
    debug = False
    def __init__(self):
        
        self.key = None
    
    '''
    @summary: checks if the classification exists
              classification is based on 'key'
    '''
    def CheckKey(self,matrix):
        
        if self.PatternObjectList.has_key(self.key):
            self.InsertPattern(matrix)
            if self.debug == True:
                print "debug"
                self.PrintClass()
            return True
        else :
            return False
    
    '''
    @summary: Inserts matrix PatternObjectList dictionary
    '''
    def InsertPattern(self,matrix):
        self.PatternObjectList[str(self.key)].append(matrix)
        #self.PrintClass()
    '''
    @summary: prints the entire class to console
    '''
    def PrintClass(self):
        print "Key = " + str(self.key)
        for key in self.PatternObjectList :
            for matrix in self.PatternObjectList[key]:
                print matrix
    
    def Erase(self):
        self.PatternObjectList.clear()
                