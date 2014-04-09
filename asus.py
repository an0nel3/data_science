"""originally written in a ipython notebook;
this contains most of the functions I used"""


%matplotlib inline
import csv as csv
import numpy as np
import time,timeit
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt


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



def modifys(dataframe,flag=0):
    """group by mod and by component if flag =0; the default
    is to modify by each both!"""
    dataframe['time'] = 12*(dataframe['year']-2008)+dataframe['month']
    return dataframe
"""        
originally tried predicting by fitting a gaussian curve to the data;
working on implementing in a much more traditional 
time series approach

"""      
p0 = [1., 0., 1.]    
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


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
                
                bad_models[(each_mod,each_cat)] = [1,1,1]
                #print each_mod,each_cat, "let's adjust this"
            else:
                try:
                    coeff, var_matrix = curve_fit(gauss, np.array(new_data['time']),\
                            np.array(new_data['number_repair']), p0=p0)
                    models[(each_mod,each_cat)] = coeff
                except RuntimeError:
                    
                    bad_models[(each_mod,each_cat)] = [each_mod,each_cat]
                    #print each_mod,each_cat, "let's adjust this because no good fit found"
            
            #print each_mod,each_cat
    return models,bad_models
    
df_mod2 = df.sum()
df_mod2 = modifys(df_mod2)
my_models = model_returner(df_mod2)
mapper2 = pd.read_csv('Output_TargetID_Mapping.csv')
mapper2=modifys(mapper2)
my_models = model_returner(df_mod2)





def predictions(dataframe,models):
    """we assume model has been trained; 
    we also assume that the mapper has been under modifys"""
    #mods = dataframe['module_category'].unique()
    values =dataframe.values
    preds = np.zeros(values.shape[0])
    for i in xrange(values.shape[0]):
        mod = values[i,0]
        cat = values[i,1]
        key = (mod,cat)
        #print key
        
        if key in models[0]:
            A,mu,sigma = models[0][key]
            
            preds[i]=A*np.exp(-((values[i,-2]-mu)**2/(2.*sigma**2)))
            #print preds
        
    return preds
   
def plot_data(models,dataframe,flag = 0):
    """need good and bad models, plots all the data"""
    if flag == 0:
        for key in models[0]:
            g=dataframe[(dataframe['module_category']==key[0]) & \
            (dataframe['component_category']==key[1])]
            plt.scatter(g['time'],g['number_repair'],c =np.random.rand(3,1))
            plt.xlabel("Time")
            plt.ylabel("number of repairs")
            plt.title("%s, and %s" %(key[0], key[1]))
            plt.show()
    if flag ==1:
        
        for key in models[1]:
            g=dataframe[(dataframe['module_category']==key[0]) & \
            (dataframe['component_category']==key[1])]
            plt.scatter(g['time'],g['number_repair'],c =np.random.rand(3,1))
            plt.xlabel("Time")
            plt.ylabel("number of repairs")
            
            if models[1][key] == [1,1,1]:
                plt.title("too little data: %s, and %s" %(key[0],key[1]))
            else:
                plt.title("no curve fit: %s, and %s" %(key[0], key[1]))
            plt.show()
#plot_data(my_models,df_mod2,flag=1)

def plot_each(key,dataframe,models):
    """ given a single key, will plot the day"""
    g=dataframe[(dataframe['module_category']==key[0]) & \
        (dataframe['component_category']==key[1])]
    plt.scatter(g['time'],g['number_repair'],c =np.random.rand(3,1))
    plt.xlabel("Time")
    plt.ylabel("number of repairs")
            
    if models[1][key] == [1,1,1]:
        plt.title("too little data: %s, and %s" %(key[0],key[1]))
    else:
        plt.title("no curve fit: %s, and %s" %(key[0], key[1]))
    plt.show()
#def plot_good(models,dataframe
#plot_each(('M7','P01'),df_mod2,my_models)

preds = predictions(mapper2,my_models)
mapper2['preds'] = preds


def plot_against(key,data,maps,flag=0):
    """plots the data against the predictions by each key"""
    #check that key is tuple
    assert type(key)== tuple
    g=data[(data['module_category']==key[0]) & \
            (data['component_category']==key[1])]
    m=maps[(maps['module_category']==key[0]) & \
            (maps['component_category']==key[1])]
    plt.scatter(g['time'],g['number_repair'],label='orig data',c='r')
    plt.scatter(m['time'],m['preds'],c='b',label='preds')
    plt.xlabel("Time")
    plt.ylabel("number of repairs")
    plt.legend(loc='best')
    if flag ==0:
        plt.title("Good curve_fit: %s, and %s" %(key[0], key[1]))
    if flag ==1:
        plt.title("too little or weird: %s, and %s" %(key[0],key[1]))
    plt.show()
    
def plot_check(key,data,maps,model,flag=0):
    """plot the good data against the curve and includes the predictions"""
    #check that key is tuple
    assert type(key)== tuple
    g=data[(data['module_category']==key[0]) & \
            (data['component_category']==key[1])]
    m=maps[(maps['module_category']==key[0]) & \
            (maps['component_category']==key[1])]
    plt.scatter(g['time'],g['number_repair'],label='orig data',c='r')
    plt.scatter(m['time'],m['preds'],c='b',label='preds')
    plt.xlabel("Time")
    plt.ylabel("number of repairs")
    plt.legend(loc='best')
    if flag ==0:
        plt.title("Good curve_fit: %s, and %s" %(key[0], key[1]))
    min_start = g['time'].min()
    max_end   = m['time'].max()
    time_stuff = np.arange(min_start,max_end,.4)
    A,mu,sigma = model[key]
    y_vals = A*np.exp(-(time_stuff-mu)**2/(2.*sigma**2))
    plt.plot(time_stuff,y_vals)
    plt.show()
#for keys in my_models[0]:
#    plot_check(keys,df_mod2,mapper2,my_models[0])












