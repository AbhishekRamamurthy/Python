'''
Created on Mar 13, 2018

@author: abhishek
'''

import numpy as np
import pandas as pd
import os
import math
'''
@param matrix: pattern matrix from formatted_patterns
@param Exfile: excel file path
@param sheet: sheet in excel file  
@summary: computes the weight of the matrix
'''
def GetWeight(matrix,Exfile = None,sheet = None):
    
        col = matrix.shape[1]
        
        if(col == 56):
            weight = matrix.sum()
            if math.isnan(weight):
                weight = np.nansum(matrix)+1
            return int(weight)
        else:
            print "Weight Verify Matrix Size , file = " + Exfile + " sheet = " + sheet
            return None
        

'''
@param matrix: pattern matrix from formatted_patterns
@param Exfile: excel file path
@param sheet: sheet in excel file  
@summary: returns count of non zero resultant columns
'''
def GetNonZeroColumn(matrix,Exfile,sheet):
    
    col = matrix.shape[1]
        
    if(col == 56):
        temp = np.asarray(matrix)
        #return zero columns
        count = np.where(~temp.any(axis=0))[0]
        if count.size == 0:
            return col - count.sum()
        else:
            return abs(col-count.size)
    else:
        print "Column Verify Matrix , path/key = " + Exfile + " sheet/level = " + sheet
        return None
    return None

'''
@param matrix: pattern matrix from formatted_patterns
@param Exfile: excel file path
@param sheet: sheet in excel file  
@summary: returns count of non zero resultant rows
'''
def GetNonZeroRow(matrix,Exfile,sheet):
    
    row = matrix.shape[0]
        
    if(row == 8):
        temp = np.asarray(matrix)
        #return zero columns
        count = np.where(~temp.any(axis=1))[0]
        if count.size == 0:
            return row - count.sum()
        else:
            return abs(row-count.size)
    else:
        print "Zero Row Verify Matrix , path/key = " + Exfile + " sheet/level = " + sheet
        return None
    return None

'''
@param matrix: pattern matrix from formatted_patterns
@param Exfile: excel file path
@param sheet: sheet in excel file  
@summary: returns the concentration of zero in each pattern
'''
def GetZeroWeight(matrix,Exfile,sheet):
    
    row = matrix.shape[0]
    if(row == 8):
        temp = pd.DataFrame(matrix)
        temp = (temp ==0).astype(int).sum(axis=0)
        temp = np.asarray(temp)
        temp = temp[temp != 0]
        return len(temp)
    else:
        print " Zero Weight Verify Matrix , path/key = " + Exfile + " sheet/level = " + sheet
        return None
    return None

'''
@param key: weight corresponding to matrix
@param Lobj: ListObject instance
@summary: Classifies the pattern based on weight  
'''
def ClassifyPattern(key,matrix,Lobj):
    
    key = str(key)
    Lobj.key=key
    if not Lobj.CheckKey(matrix):
        Lobj.InsertPattern(matrix)
        #Lobj.PrintClass()

'''
@param classifer: type of classification
@param Lobj: ListObjects type instance
@summary: For sub-classification of patterns classified in 
          first section, can be further classified based on the
          level of abstraction needed.
'''   
def SubClassify(classifier,Lobj):
    
    firstLevel = Lobj.PatternObjectList.copy()
    Lobj.Erase()
    
    #cluster based on weight of the matrix
    # weight is summation of i from 0 to N of Xi
    if classifier == "weight":
        for pattern in firstLevel:
            for matrix in firstLevel[pattern]:
                key = GetWeight(matrix, pattern,"lower Level")
                Lobj.key = pattern+"_"+ str(key)
                if(key == None):
                    print "(weight) pattern = " +pattern
                #Lobj.debug = True
                check = Lobj.CheckKey(matrix)
                if not check :
                    Lobj.InsertPattern(matrix)
    elif classifier == "row":
        for pattern in firstLevel:
            for matrix in firstLevel[pattern]:
                key = GetNonZeroRow(matrix, pattern,"lower Level")
                Lobj.key = pattern+"_"+ str(key)
                #Lobj.debug = True
                if(key == None):
                    print "(row) pattern = " +pattern
                check = Lobj.CheckKey(matrix)
                if not check :
                    Lobj.InsertPattern(matrix)
    elif classifier == "zeroweight":
        for pattern in firstLevel:
            for matrix in firstLevel[pattern]:
                key = GetZeroWeight(matrix, pattern,"lower Level")
                Lobj.key = pattern+"_"+ str(key)
                #Lobj.debug = True
                if(key == None):
                    print "(zeroweight) pattern = " +pattern
                check = Lobj.CheckKey(matrix)
                if not check :
                    Lobj.InsertPattern(matrix)
    else:
        pass
'''
@param Lobj:ListObjects type instance
@summary: dump the clustered data to respective excel file
          file name is of form col_weight_row_class.xlsx
'''
def ExcelWrite(Lobj):
    
    final = Lobj.PatternObjectList.copy()
    Lobj.Erase()
    if not os.path.exists(os.getcwd()+"/../Results"):
        os.mkdir(os.getcwd()+"/../Results")
    
    for pattern in final:
        writer = pd.ExcelWriter(os.getcwd()+"/../Results/"+pattern+"_class.xlsx",engine = 'openpyxl')
        i=1;
        for matrix in final[pattern]:
            matrix = pd.DataFrame(matrix)
            matrix.to_excel(writer,sheet_name= "Sheet"+str(i))
            i=i+1
            writer.save()
        writer.close()
    
                   