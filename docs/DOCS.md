# What is pyrebel?

Pyrebel is a pure python library that implements abstraction of data.

# The class "Abstract" handles methods for abstraction of data.

`__init__(self,bound_data_ordered_h,n_bounds,bound_abstract_h,shape_h,is_closed,threshold_h=-1)`

    1. bound_data_ordered_h - 1D array containing indices of boundaries of blobs in the image.
    2. n_bounds - Number of blobs in the 1D array.
    3. bound_abstract_h - 1D array having the same length as bound_data_ordered_h containing indices of initial abstract points.
    4. shape_h - Shape of the 2D image.
    5. is_closed - True for 2D images with closed boundaries and False for discrete data sets.
    6. threshold_h - 1D array of threshold of abstraction for each blob.

`do_abstract_all(self,ba_threshold=-1)`

    1. ba_threshold - (Optional) Overrides the threshold of abstraction passed to `__init__`
    
    Returns 1D array containing indices of abstract points of all the layers of abstraction of boundaries of blobs in the image. 

`do_abstract_one(self)`

    Returns 1D array containing indices of abstract points of another layer of abstraction of boundaries of blobs in the image. 

`get_sign(self)`

    Returns 1D array containing -1 or 1 depending on the convexity of the abstract point.

`get_abstract_size(self)`
    
    Returns 1D array of the count of abstract points of each boundaries of blobs in the image.

# The class "Preprocess" handles methods for obtaining 1D array containing indices of boundaries of blobs in the image

`__init__(self,img_array)`

    1. img_array - 2D array of the input image.

`get_bound_size(self)`

    Returns a 1D array containing length of boundaries of each blob. The length of the array equals the number of blobs in the image.

`get_bound_data(self)`

    Returns a 1D array containing indices of boundaries of blobs in the image.

`get_init_abstract(self)`

    Returns a 1D array containing indices of initial abstract points.

`set_bound_size(self,min_size,max_size)`

    1. min_size - Sets the minimum length of boundaries of blobs in the image.
    2. max_size - Sets the maximum length of boundaries of blobs in the image.

