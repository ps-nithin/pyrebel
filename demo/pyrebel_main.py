# Copyright (C) 2024-2025 Nithin PS.
# This file is part of Pyrebel.
#
# Pyrebel is free software: you can redistribute it and/or modify it under the terms of 
# the GNU General Public License as published by the Free Software Foundation, either 
# version 3 of the License, or (at your option) any later version.
#
# Pyrebel is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
# PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Pyrebel.
# If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
from PIL import Image
import math,argparse,time
from pyrebel.preprocess import Preprocess
from pyrebel.abstract import Abstract
from pyrebel.utils import *

# This is a demo of abstraction of boundaries of blobs in the image.
# When you run this program the output is written to 'output.png'.
# The boundaries of blobs is in grey color and the abstract points are in white.
# The level of abstraction can be changed by giving '--threshold' argument. 
# The default value of threshold is 5.

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name.")
parser.add_argument("-t","--abs_threshold",help="Threshold of abstraction.")
args=parser.parse_args()
if args.abs_threshold:
    abs_threshold=int(args.abs_threshold)
else:
    abs_threshold=5
        
while 1:
    start_time=time.time()    
    if args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
    else:
        print("No input file.")
    
    # Initialize the preprocessing class.
    pre=Preprocess(img_array)
    
    # Set the minimum and maximum size of boundaries of blobs in the image. Defaults to a minimum of 64.
    pre.set_bound_size(32)
    
    # Perform the preprocessing to get 1D array containing boundaries of blobs in the image.
    pre.preprocess_image()
    
    # Get the 1D array.
    bound_data=pre.get_bound_data()
    bound_data_d=cuda.to_device(bound_data)
    
    # Initialize the abstract boundary.
    init_bound_abstract=pre.get_init_abstract()
    
    # Get 1D array containing size of boundaries of blobs in the array.
    bound_size=pre.get_bound_size()
    print("len(bound_data)=",len(bound_data))
    
    shape_d=cuda.to_device(img_array.shape)
    scaled_shape=[img_array.shape[0]*3,img_array.shape[1]*3]
    scaled_shape_d=cuda.to_device(scaled_shape)
    
    bound_data_orig=np.zeros(len(bound_data),dtype=np.int32)
    bound_data_orig_d=cuda.to_device(bound_data_orig)
    
    scale_down_pixels[len(bound_data),1](bound_data_d,bound_data_orig_d,scaled_shape_d,shape_d,3)
    cuda.synchronize()
    
    # Initialize the abstraction class
    abs=Abstract(bound_data,len(bound_size),init_bound_abstract,scaled_shape,True)
    
    # Get the abstract points
    abs.do_abstract_all(abs_threshold)
    abs_points=abs.get_abstract()
    
    abs_draw=decrement_by_one_cuda(abs_points)
    abs_draw_d=cuda.to_device(abs_draw)

    out_image=np.zeros(img_array.shape,dtype=np.int32)
    out_image_d=cuda.to_device(out_image)

    # Draw the boundaries to the output image.
    draw_pixels_cuda(bound_data_orig_d,50,out_image_d)
    draw_lines[len(abs_draw),1](abs_draw_d,bound_data_orig_d,out_image_d,125)
    cuda.synchronize()
    
    # Draw the abstract points to the output image.
    draw_pixels_from_indices_cuda(abs_draw_d,bound_data_orig_d,255,out_image_d)
    out_image_h=out_image_d.copy_to_host()
    
    # Save the output to disk.
    Image.fromarray(out_image_h).convert('RGB').save("output.png")
    print("Finished in total of",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
    break
