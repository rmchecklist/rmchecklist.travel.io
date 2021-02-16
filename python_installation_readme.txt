1. Python anaconda download installation instruction ==> https://docs.anaconda.com/anaconda/install/windows
2. Python download link ==> https://www.anaconda.com/products/individual#windows
3. conda env create -f environment.yml
4. activate pyfinance
5. jupyter notebook

6. Numpy ==> import numpy as np - This can convert the regular list object into numpy array
          *****************************numpy array******************************************
          numpy.array() ==> converts list to array
          numpy.arange(0,5) ==> creates a numpy array from range 
          numpy.zeros(row) ==> creates 1D array with floating type zeros
          numpy.zeros((row, col)) ==> it allows tuple argument and creates 2D array with floating type zeros
          numpy.ones(row) ==> similar to zeros, it generates 1 based upon the input arguments
          linear space ==> numpy.linspace(start, stop, num) ==> generates number start and evently spaced by number
                    e.g. numpy.linspace(1, 10, 3) ==> [1, 5.5, 10]
          Identity matrix ==> numpy.eye(N) ==> print diagonally 1 and all other values are filled with 0
                    e.g. numpy.eye(2)  ==> array([[1., 0.],
                                                  [0., 1.]])
          Random number ==> returns uniformly distribute the randon numbers ==> numpy.randon.rand(number)
                    e.g. numpy.random.rand(2) ==> array([0.99482662, 0.69422727])
          Random number in standard normal distribution ==> numpy.randon.randn(n)
                    e.g. numpy.random.rand(2) ==> array([ 0.64988124, -0.09935966])
          Randon integers ==> numpy.randon.randint(start(inclusive), end(exclusive), numberofitems(optional))
                    e.g. numpy.random.randint(1, 10) ==> 8
                         numpy.random.randint(1, 10, 10) ==> array([3, 1, 8, 1, 5, 1, 2, 4, 9, 7])
          Reshape the array ==> reshape the current current array to nD array
                    e.g. na = numpy.arange(0,10) ==> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
                         na.reshape(2, 5) ==> array([[0, 1, 2, 3, 4],
                                                   [5, 6, 7, 8, 9]])
                    
          shape ==> return the current length of the N dimentional array
                    e.g. numpy.shape(a ==>(1 to 25)) ==> (25, )
                    e.g. numpy.reshape(5,2).shape ==> (5,2)
          find the data type ==> arr = numpy.array([1,2,3,4])
                    e.g. arr.dtype ==> dtype('int32')
          numpy find max value ==> an = numpy.random.randint(1,10,10) ==> an.max()
          find max number array index ==> an.argmax()
          find min number and array index ==> an.min(), an.argmin()
          ***************************************************************************************************
          
          *********************Numpy operation*************************
          
          Adding 2 arrays ==> arr1 + arr2 ==> sum the array of each indexes and return the result
          
                    e.g. an = np.arange(1,10) ==> array([1, 2, 3, 4, 5, 6, 7, 8, 9])
                         an + an ==> array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])
                         
                         use add, subtraction, multiply, divide ==> if value 0/0 it return nan value and 1/0 return inf
                         
                         Additionally we can square and add value in to the exisitng array
                         
                         an**2, an+100
                         
                         square root ==> numpy.sqrt(arr)
                         
                         exponential ==> numpy.exp(arr)
                         
                         arr.max() or numpy.max(arr), both are same, arr.max() internally calls numpy.max(arr)
                         
                         numpy.sin(arr)
                         
                         numpy.log(arr)
                         
                ***********************Numpy indexing***************************
                
                Bracket indexing works same as python array indexing.
                
                arr[0] , arr[0:], arr[:2], arr[1:2]
                
                broadcasting ==> assing the value by slicing 
                  
                           e.g. array = numpy.arange(0,11) ==> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
                                array[0:5] = 100 ==> array([100, 100, 100, 100, 100,   5,   6,   7,   8,   9,  10])
                                
                                array[:] = 99 ==> set all array values to 99
                                
                           e.g. array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
                                    
                                    slice_of_array = array[0:6] ==> array([0, 1, 2, 3, 4, 5])
                                    slice_of_array[:] = 99 ==> array([99, 99, 99, 99, 99, 99])
                                    
                                    it also udpate the values of array ==> array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])
                  Array copy ==> arr_copy = array.copy() ==> this will copy the array rather than using the reference copy, 
                  so change of this array won't affect array which we copied from
                  
                  e.g. array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]) 
                       array_copy = array.copy()
                       array_copy[:] = 100 ==> array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
                       array ==> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]) 
                       
              Matrix or 2D array value can be accessed by index of row and column indexes 
              
                  e.g. mat = numpy.array([[1,2,3], [4,5,6], [7,8,9]]) == > 
                           array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]])
                                    
                         Value 1 ==> mat[0,0] or mat[0][0]
                         
                Slice the matric by mat[:2, 1:] ==> array([[2, 3],
                                                               [5, 6]])
                                                               
                                                               
                                                               
Pandas(named after Panel data)

         Series ==> 
                  converts array/list/dict into index values series
                           e.g. pandas.Series(list/array/dict)
                  
         Data frames ==> 
                  Form a table matric with rows and columns Names
                           e.g. df = pandas.DataFrame(numpy.random.rand(5,4), ['A','B', 'C', 'D', 'E'], ['W', 'X','Y','Z'])
                  Adding a new column df['new] = df['X'] + df['Y']
                  
                  Dropping a column or row
                           e.g. df.drop('new', axis=1) axis = 0[default] represents x axis or row, 1 represents Y axis or col
                           use inplace = True to replace permantly from date frame object ==> df.drop('new', axis=1, inplace=True)
                           
                           multi column series==> df[['W','X']]
                           multi row series ==> df.loc['A']
                           
                           indexed localtion ==> df.iloc[2]
                           
                           Return subset by label ==> df.loc[['A','B'],['X','Y']]
                           
                           
                  Conditional selection ==>  df[df >0] , multiple condition ==> df[(df['W'] > 0)|(df['X'] > .05)]
                  
                  reset index ==> df.reset_index() ==> create new 0 based index and current index will be new column value
                  
                  Add new column  ==> df['state] = 'CA GA PA OH NY'.split()
                  
                  set new index ==> df.set_index('state')
                  
                  
                  Drop missing value ==> df.dropna() ==> remove all nan values from dataframe
                  
                           e.g. df.dropna(thresh=2) ==> return data frames of 2 non-nan values
                           
                           
                  Fill missing value ==> df.fillna(value='any value')
                  
                           e.g. df['B'].fillna(value=df['B'].mean())
                           
                  Group by with pandas:
                  =====================
                           df.groupby('Company') ==> you can call the aggregate function such as min, max, mean, std, avg
                           
                           e.g. df.groupby('Company').describe()
                                    	Sales
                                                      count	mean	std	min	25%	50%	75%	max
                                                      Company								
                                                      FB	2.0	296.5	75.660426	243.0	269.75	296.5	323.25	350.0
                                                      GOOG	2.0	160.0	56.568542	120.0	140.00	160.0	180.00	200.0
                                                      MSFT	2.0	232.0	152.735065	124.0	178.00	232.0	286.00	340.0
                                                      
                                                      
                                                      
                                                      
                  df.groupby('Company').describe().transpose()
                  
                  e.g. 
                           Company	FB	GOOG	MSFT
                           Sales	count	2.000000	2.000000	2.000000 
                           mean	296.500000	160.000000	232.000000
                           std	75.660426	56.568542	152.735065
                           min	243.000000	120.000000	124.000000
                           25%	269.750000	140.000000	178.000000
                           50%	296.500000	160.000000	232.000000
                           75%	323.250000	180.000000	286.000000
                           max	350.000000	200.000000	340.000000
                           
                           
                  Data frame - Concat, merge and Join:
                  ====================================
                  
                  concatenation ==> pd.concat([df1, df2, df3], axis=? ? represents row or col concatenation)
                  
                  pd.merge(left, right, how='innner', on="Key')
                  
                  left.join(right, how="inner/outer")
                  
                  Pandas common operators:
                  ========================
                  
                  1. df.head() as same returning value df
                  2. Unique value ==> df['col2'].unique()
                  3. No of unique value ==> df['col2'].nunique()
                  4. value occurance of the columns ==> df['col2'].value_counts()
                  5. Conditional selection ==> 
                           e.g. 
                                    operation df[df['col2'] > 1]
                                    df[(df['col1'] >2) & (df['col2'] == 666)]
                  6. apply ==> broadcast the function to dataframe
                           def times2(x):
                                    return x*2
                           
                           df['col1'].apply(times2)
                           
                           lambda ==> df['col1'].apply(lambda x : x*2)
                  
                  
                  7. Drop columns  ==> df.drop('col2', axis=1, inplace=True)
                  
                  8. Columns names => df.columns
                  
                  9. index values ==> df.index
                  
                  10. Sort the values ==> df.sort_values('col2')
                  
                  11. Check null values ==> df.isnull()
                  
                  12. Pivot table
                           e.g. df.pivot_table(values='D', index=['A','B'], columns=['C'])
                           
                                                      A	B	C	D
                                             0	foo	one	x	1
                                             1	foo	one	y	3
                                             2	foo	two	x	2
                                             3	bar	two	y	5
                                             4	bar	one	x	4
                                             5	bar	one	y	1
                                             Values --> Values needs to be grouped
                                             index --> columns needs to be grouped
                                             columns --> new pivot columns 
                                             
                                             Output:
                                             
                                             	C	x	y
                                             A	B		
                                             bar	one	4.0	1.0
                                             two	NaN	5.0
                                             foo	one	1.0	3.0
                                             two	2.0	NaN
                                             
Data input and output:
         1. Read Csv file ==> pd.read_csv('file_name')
         2. pd.to_csv('new_file_name', index=False)
         3. Read excel file ==> df = pd.read_excel('Excel_Sample.xlsx', sheet_name="Sheet1")
         
         
Example: 
         Show the head of the dataframe  ==> data.head()
         
         What are the column names? ==> data.columns
         
         How many States (ST) are represented in this data set? ==> data['ST'].nunique()
         
         Get a list or array of all the states in the data set. ==> data['ST'].unique()
         
         What are the top 5 states with the most failed banks? ==>
         
         data.groupby('ST').count().sort_values('Bank Name', ascending=False).iloc[:5]['Bank Name']
         
          What are the top 5 acquiring institutions? ==> data['Acquiring Institution'].value_counts().iloc[:5]
          
          What is the most common city in California for a bank to fail in?
          
          data[data['ST'] == 'CA'].groupby('City').count().sort_values('Bank Name', ascending=False).head(1)
          
           How many failed banks don't have the word "Bank" in their name?
           
           sum(data['Bank Name'].apply(lambda name : 'Bank' not in name))
           
           How many bank names start with the letter 's' ? 
           
           sum(data['Bank Name'].apply(lambda name : 'S' == name[0]))
           
           How many CERT values are above 20000 ? 
           
           sum(data['CERT'].apply(lambda cert : cert > 20000))
            or 
            sum(data['CERT'] > 20000)
           
           How many bank names consist of just two words? (e.g. "First Bank" , "Bank Georgia" )
           
           sum(data['Bank Name'].apply(lambda name : len(name.split()) == 2))
           
           Data info ==> data.info()
           
           
           How many banks closed in the year 2008? (this is hard because we technically haven't learned about time series with pandas yet! Feel free to skip this one!
           
           sum(data['Closing Date'].apply(lambda name : name.split('-')[2] == '08'))
            or 
            
           sum(data['Closing Date'].apply(lambda name : name[-2:] == '08'))
           
           or 
           
           sum(pd.to_datetime(data['Closing Date']).apply(lambda date : date.year == 2008))
           
           
           
           
  Visualization in python: Matplotlib
  
  pip install matplotlib
  
  import matplotlib.plyplot as plt
  
  import numpy as np
  
  %matplotlib inline
  x = np.linspace(0, 5, 11)
  y = x**2

  #Functioanl way to create a plot
  -------------------------------
  plt.plot(x, y)
  
         plt.subplot(1,2,1)
         plt.plot(x,y,'r')
         plt.subplot(1,2,2)
         plt.plot(y,x,'b')
         plt.plot(x, y, 'r-')
         plt.xlabel('X Label')
         plt.ylabel('Y Label')
         plt.title('Titile')
         plt.show() 
         
  
  plt.show() -- if you are not using jupyter
  
  #OO way to create a plot
  ------------------------
         fig = plt.figure()
         axes = fig.add_axes([0.1,0.1, 0.8, .8])
         axes.plot(x,y)
         axes.set_xlabel('X Label')
         axes.set_ylabel('Y Label')
         axes.set_title('Set title')
  
  
Creating inner plot:
--------------------
fig =plt.figure()

axes1 = fig.add_axes([.1,.1,.8,.8])

axes2 = fig.add_axes([.2,.5,.3,.3])

axes1.plot(x,y)
axes2.plot(y,x)


Subplots:

fig, axes = plt.subplots(nrows=1, ncols =3)
fig.tight_layout()


Figure size and DPI
-------------------
Plot:

fig = plt.figure(figsize=(3,2))
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y)

Subplot:

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3,2))

axes[0].plot(x,y)

Legend:(Differtiate between the graphs)

fig = plt.figure()
axes  = fig.add_axes([0,0,1,1])

axes.plot(x, x**2, label='X sq')
axes.plot(x, x**3, label='X cu')

axes.legend()


axes.legend(loc=(.4,.7))

Loc code will used to map the legends(Best - 0, 1,2 ...10) or else specify by tuple(.2,.2)

Plot appearance:
-----------------

Colors:
=======
fig = plt.figure()
axes = fig.add_axes([0,0,1,1])
axes.plot(x,y, color='orange') ==> color can be RGB Hex code e.g. #3463FGCC

Linewidth:
==========

axes.plot(x,y, color='orange', linewidth=5) ==> Graph line width, linewidth/lw ==> both will work.

Line transparancy(alpha) ==> axes.plot(x,y, color='orange', lw=5, alpha=.5)

Line stype ==> axes.plot(x,y, color='orange', lw=5, linestyle='dashed')

Marker: 
=======
axes.plot(x,y, color='orange', lw=1, ls='-', marker='o')

Makersize(makersize/ms)
-----------------------
axes.plot(x,y, color='orange', lw=1, ls='-', marker='+', ms=20)

MarkerFace color and edg width:
--------------------------------

axes.plot(x,y, color='green', lw=1, ls='-', marker='o', ms=20,
         markerfacecolor='yellow', markeredgewidth=3)

  
         
Display X and Y range on the plot

axes.set_xlim([0,1])
axes.set_ylim([0,1])

It displays range plot from 0-1
         
         
         
                  
Example:                  
                  
fig, ax = plt.subplots(1,2)

ax[0].set_xlim(0,100)
ax[0].set_ylim(0,200)

ax[1].set_ylim(0,10000)
ax[1].set_xlim(0,100)

ax[0].plot(x,y, color='blue', linestyle='dashed', lw=3)
ax[1].plot(x,z, color='red', linestyle='-', lw=3)

plt.tight_layout()
                  
Histogram:

df['A].plot(kind='hist', bins=30)
or
df['A].hist(bins=30)
or
df['A'].plot.hist()

df.plot.area()

df2.plot.bar(stacked=True)

df.plot.line(figsize=(12,3), lw =1)

df.plot.scatter(x='A', y='B')

df.plot.scatter(x='A', y='B', c = 'C', cmap='coolwarm')

df.plot.scatter(x='A', y='B', s=df['C']*100)

df.plot.box()

df = pd.DataFrame(np.random.randn(1000, 2), columns=['a','b'])

df.plot.hexbin(x='a', y='b')

df.plot.hexbin(x='a', y='b', gridsize=25)

df.plot.hexbin(x='a', y='b', gridsize=25, cmap='coolwarm')

Kernal density estimation ==> df['a'].plot.kde() or df['a'].plot.density()

mcdon['Adj. Close'].plot()





mcdon['Adj. Close'].plot(xlim=['2007-10-1', '2009-01-01'])


import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

fig,ax = plt.subplots()
ax.plot_date(idx, stock, '-')
fig.autofmt_xdate()


Datetime index:
================

from datetime import datetime

my_date = datetime(year, month, day, hour, minute, sec)

read year, month, day and other date time related information by their name

my_date.year ...etc

Create Datetime index ==> dt_ind = pd.DatetimeIndex(datetime list object)

Create data frame usign same date time index

data = np.random.randn(2,2)

cols = ['a','b']

df = pd.DataFrame(data, dt_ind, cols )

	         a	b
2015-01-02	-1.914149	1.235571
2016-01-01	0.929838 	-0.562650


df.index.argmax() ==> index max position

df.index.max() ==> return actual value of the index max


df.index.argmin() ==> return min index position
df.index.min() ==> return min index position value



Time Resampling:
================


df = pd.read_csv('xyz.csv', index_col = 'Date', parse_dates=True) ==> If this column is Object string then we need manually converts them to datetime object.

if index_col is set then parse_dates will converts the index column to datatime object.

Conver the object to datatime ==> df['col_name'] = pd.to_datetime(df[col_name'])

or 

df['Date'] = df['Date'].apply(pd.to_datetime)

df.index ==> return the index column


Read stock data from web:
=========================

import pandas_datareader
import datetime
import pandas_datareader.data as web

start  =datetime.datetime(2012,1,1)
end  =datetime.datetime(2017,1,1)

tesla = web.DataReader('TSLA', 'yahoo', start, end)

tesla.head()


** Recreate this linear plot of all the stocks' Open price ! Hint: For the legend, use label parameter and plt.legend()**

df_tesla['Open'].plot(label='Tesla', figsize=(12,8), title= 'Open Price')
df_ford['Open'].plot(label='Ford')
df_gm['Open'].plot(label='GM')
plt.legend();


max volume traded:

df_ford['Volume'].max()

Let's practice plotting out some MA (Moving Averages). Plot out the MA50 and MA200 for GM. **

# Code here
df_gm['MA50'] = df_gm['Open'].rolling(50).mean()
df_gm['MA200'] = df_gm['Open'].rolling(200).mean()

df_gm[['Open', 'MA50', 'MA200']].plot(figsize=(16,8))


from pandas.plotting import scatter_matrix





-------------------------------------------------------

Daily percent change ==> 𝑟𝑡=(𝑝𝑡/𝑝𝑡−1)−1
	It means (current day close value/ previous  day close value) -1 
	previous day close value can identified by shift(1)


 
         
                                                               
                                                               
                  
                  
                                
                         
           
          
                                        
          
                                        
                    
