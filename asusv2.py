import csv as csv
import df as np
import time,timeit
from scipy.optimize import curve_fit
import pandas as pd
import matplot


repairs = pd.read_csv('RepairTrain.csv')
def month_and_years(df,column):
    years = []
    months = []
    ym_sale = pd.to_datetime(df[column])
    for each_tuple in ym_sale:
        years.append(each_tuple.year)
        months.append(each_tuple.month)
    return years,months
    
year_repair,month_repair = month_and_years(repairs,"year/month(repair)")
data_frame = repairs[['module_category','component_category','number_repair']]
data_frame['year'] = year_repair
data_frame['month'] = month_repair
df       = data_frame.groupby(['module_category','component_category','year','month', \
                               ],as_index = False)
df_mod   = df.sum()



def modify(dataframe,flag=0):
    """group by mod and by component if flag =0; the default
    is to modify by each both!"""
    dataframe['time'] = 12*(dataframe['year']-2008)+dataframe['month']
    return dataframe
df_mod = modify(df_mod)   


#code for gauss from 
#http://stackoverflow.com/questions/11507028/fit-a-gaussian-function
p0 = [1., 0., 1.]    
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
def modifys(dataframe,flag=0):
    """group by mod and by component if flag =0; the default
    is to modify by each both!"""
    dataframe['time'] = 12*(dataframe['year']-2008)+dataframe['month']
    return dataframe
    

    
    
    
    
def model_returner(train):    
    models = {}
    bad_models ={}
    mods = train['module_category'].unique()
    for each_mod in mods:
        cats = train[train['module_category']==each_mod]['component_category'].unique()
        for each_cat in cats:
            new_data = train[(train['module_category']==each_mod) \
                    & (train['component_category']==each_cat)]
            if new_data.shape[0] <= 3:
                coeff =[1,1,1]
                bad_models[(each_mod,each_cat)] = coeff
                print each_mod,each_cat, "let's adjust this"
            else:
                try:
                    coeff, var_matrix = curve_fit(gauss, np.array(new_data['time']),\
                            np.array(new_data['number_repair']), p0=p0)
                except RuntimeError:
                    coeff = [1,1,1]
                    bad_models[(each_mod,each_cat)] = [each_mod,each_cat]
                    print each_mod,each_cat, "let's adjust this because no good fit found"
            
            #print each_mod,each_cat
    return models,bad_models


def predictions(dataframe,models):
    """we assume model has been trained; 
    we also assume that the mapper has been under modifys"""
    mods = dataframe['module_category'].unique()
    dataframe['predictions']=5
    
    
    for each_mod in mods:
        cats = dataframe[dataframe['module_category']==each_mod]['component_category'].unique()
        for each_cat in cats:
            key =(each_mod,each_cat)
            if key in models[0]:
                A,mu,sigma = models[0][key]
                
                #A*np.exp(-(x-mu)**2/(2.*sigma**2))
                dataframe[(dataframe['module_category']==each_mod) \
                    & (dataframe['component_category']==each_cat)]['predictions'] = \
                  A*np.exp(-(dataframe[(dataframe['module_category']==each_mod) \
                    & (dataframe['component_category']==each_cat)]['time'] - mu)**2/(2.*sigma**2))
    dataframe['predictions'].round()
    return dataframe
                
              
    
    
    
    
    
    
    
    
    
    
    