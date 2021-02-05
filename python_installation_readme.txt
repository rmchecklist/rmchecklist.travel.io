1. Python anaconda download installation instruction ==> https://docs.anaconda.com/anaconda/install/windows
2. Python download link ==> https://www.anaconda.com/products/individual#windows
3. conda env create -f environment.yml
4. activate pyfinance
5. jupyter notebook

6. Numpy ==> import numpy as np - This can convert the regular list object into numpy array
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
                                        
                    
