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
                           
                  Group by with pandas
                  
                 
                  
                  
                  
                  
                  
         
                                                               
                                                               
                  
                  
                                
                         
           
          
                                        
          
                                        
                    
