Data from [Hand Dataset](http://www.robots.ox.ac.uk/~vgg/data/hands/)   
which should be placed as following:    
```
test_data/images/\*.jpg  
test_data/annotations/\*.mat
```
and same in train_data and validation_data.     
`predo.py` will generate dataset label including `filename, [[x1, y1, x2, y2], ...]` in each data folder.   
