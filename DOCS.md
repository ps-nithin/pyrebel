What is pyrebel?
Pyrebel is a pure python library that implements abstraction of data.


The class "Abstract" handles methods for abstraction of data.

__init__(self,bound_data_ordered_h,n_bounds,bound_abstract_h,shape_h,is_closed,ba_threshold_pre)

    1. bound_data_ordered_h - 1D array containing indices of boundaries of blobs in the image.
    2. n_bounds - Number of blobs in the 1D array.
    3. bound_abstract_h - 1D array having the same length as bound_data_ordered_h containing indices of initial abstract points.
    4. shape_h - Shape of the 2D image.
    5. is_closed - True for 2D images with closed boundaries and False for discrete data sets.
    6. ba_threshold_pre - The threshold of abstraction.

get_abstract_all(self)

    1. Returns 1D array containing indices of abstract points of all the layers of boundaries of blobs in the image. 

get_sign(self)

    1. Returns 1D array containing -1 or 1 depending on the convexity of the abstract point.



The class "Preprocess" handles methods for obtaining 1D array containing indices of boundaries of blobs in the image

__init__(self,img_array)

    1. img_array - 2D array of the input image.

get_bound_size(self)

    1. Returns a 1D array containing length of boundaries of each blob. The length of the array equals the number of blobs in the image.

get_bound_data(self)

    1. Returns a 1D array containing indices of boundaries of blobs in the image.

get_init_abstract(self)

    1. Returns a 1D array containing indices of initial abstract points.

set_bound_size(self,min_size,max_size)

    1. Sets the minimum and maximum length of boundaries of blobs in the image.


