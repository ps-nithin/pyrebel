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
import math,argparse,time,sys
from pyrebel.preprocess import Preprocess
from pyrebel.abstract import Abstract
from pyrebel.learn import Learn
from pyrebel.edge import Edge
from pyrebel.utils import *
from pyrebel.getnonzeros import *     
        
# This is a demo of forming 2D sketch using abstraction of data.
# When you run this program the output is written to 'output.png'.
#

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name.")
parser.add_argument("-at","--abs_threshold",help="Threshold of abstraction.")
parser.add_argument("-et","--edge_threshold",help="Threshold of edge detection.")
parser.add_argument("-s","--bound_threshold",help="Threshold of boundary size.")
args=parser.parse_args()

if args.edge_threshold:
    edge_threshold=int(args.edge_threshold)
else:
    edge_threshold=5   
if args.abs_threshold:
    abs_threshold=int(args.abs_threshold)
else:
    abs_threshold=10    
if args.bound_threshold:
    bound_threshold=int(args.bound_threshold)
else:
    bound_threshold=100
 
while 1:
    start_time=time.time()    
    if args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
    shape_d=cuda.to_device(img_array.shape)
    
    threadsperblock=(16,16)
    blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
    blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
    blockspergrid=(blockspergrid_x,blockspergrid_y)
    img_array_d=cuda.to_device(img_array)
    edge=Edge(img_array)
    edge.find_edges(edge_threshold)
    edges=edge.get_edges_bw()
    
    # Initialize the preprocessing class.
    pre=Preprocess(edges)
    # Set the minimum and maximum size of boundaries of blobs in the image. Defaults to a minimum of 64.
    pre.set_bound_size(bound_threshold)
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
    print("n_blobs=",len(bound_size))
    
    scaled_image=pre.get_image_scaled()
    scaled_image_d=cuda.to_device(scaled_image)
    scaled_shape=scaled_image.shape
    scaled_shape_d=cuda.to_device(scaled_shape)
    
    # Initialize the abstraction class
    abs=Abstract(bound_data,len(bound_size),init_bound_abstract,scaled_shape,True)
    abs.do_abstract_all(abs_threshold)
    abs_points=abs.get_abstract()
    abs_size=abs.get_abstract_size()
    abs_size_d=cuda.to_device(abs_size)
    abs_size_cum_=np.cumsum(abs_size)
    abs_size_cum=np.delete(np.insert(abs_size_cum_,0,0),-1)
    abs_size_cum_d=cuda.to_device(abs_size_cum)    
    
    abs_draw=decrement_by_one_cuda(abs_points)
    abs_draw_d=cuda.to_device(abs_draw)
    
    out_image=np.full(img_array.shape,255,dtype=np.int32)
    out_image_d=cuda.to_device(out_image)
    
    bound_data_orig=np.zeros(len(bound_data),dtype=np.int32)
    bound_data_orig_d=cuda.to_device(bound_data_orig)
    
    scale_down_pixels[len(bound_data),1](bound_data_d,bound_data_orig_d,scaled_shape_d,shape_d,3)
    cuda.synchronize()
    
    draw_lines[len(abs_draw),1](abs_draw_d,bound_data_orig_d,out_image_d,0)
    cuda.synchronize()
    
    out_image_h=out_image_d.copy_to_host()
    
    # Save the output to disk.
    Image.fromarray(out_image_h).convert('RGB').save("output.png")
    print("Finished in",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
    break
