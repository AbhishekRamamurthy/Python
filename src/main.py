'''
Created on Mar 13, 2018

@author: abhishek
'''
import sys
import os
import pandas as pd
import re
import ListObjects 
import Classifiers as group
import Write2Db as db
import HeatMap as HM
import ImagePreProcess as ipp

#path = os.getcwd()+"/../../TestPatterns/"
#path_heatMap = os.getcwd()+"/../../TestExPatterns/" 
#path_image = os.getcwd()+"/../../../../Desktop"  
path_image = os.getcwd()+"/../../../../" 
def HeatMap():
    for ExFile in os.listdir(path_heatMap):
        
        HM.PlotHeatMap(path_heatMap,ExFile)

def Cluster():
    regex = re.compile("^.*.xlsx$")
    Lobj = ListObjects.ListObjects()
    
    #Group based on number of non-zero columns
    if os.path.exists(path):
        for Exfile in os.listdir(path):
            if regex.match(Exfile):
                read = pd.ExcelFile(path+"/"+Exfile)
                for sheet in read.sheet_names:
                    dframe = read.parse(sheet,header=None)
                    dframe = dframe.as_matrix()
                    key = group.GetNonZeroColumn(dframe,Exfile,sheet)
                    if key != None:
                        group.ClassifyPattern(key,dframe,Lobj)
                
    else:
        print "Path " + path + " Doesn't exist"
        
    #Group Based on weight
    group.SubClassify("weight",Lobj)
    #Not worth classifying patterns on rows, as there are cases which gets
    #are affected due to internal column rotation
    #group.SubClassify("row",Lobj)
    group.SubClassify("zeroweight",Lobj)
    keys = Lobj.PatternObjectList.keys()
    #group.ExcelWrite(Lobj)
    DB =db.Write2Db("pattern","set1")
    for key in keys:
        (column,weight,zeroweight) = key.split("_")
        matrices = Lobj.PatternObjectList[key]
        DB.Insert(key,column,weight,zeroweight,matrices) 
    return

def main():
    if len(sys.argv) > 1 :
        if sys.argv[1] == "cluster":
            Cluster()
        else:
            HeatMap()
    else:
        ipp.PreProcess(path_image)
        
    print "Check the Results"

if __name__ == '__main__':
    main()
