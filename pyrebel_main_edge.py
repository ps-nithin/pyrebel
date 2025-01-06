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

# This is a demo of edge detection using abstraction of data.
# When you run this program the output is written to 'output.png'.
# The level of abstraction can be changed by giving '--threshold' argument. 
# The default value of threshold is 5.
#
# The edges are detected by abstracting the image horizontally and vertically. Then, they are
# combined to get the final result.
#

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name.")
parser.add_argument("-t","--threshold",help="Threshold of abstraction.")
args=parser.parse_args()
if args.threshold:
    abs_threshold=int(args.threshold)
else:
    abs_threshold=5
 
while 1:
    start_time=time.time()    
    if args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
    else:
        print("No input file.")
    shape_orig=img_array.shape
    i=0
    while 1:
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        img_array_d=cuda.to_device(img_array)
        img_wave=np.zeros(img_array.shape[0]*img_array.shape[1],dtype=np.int32)
        img_wave_d=cuda.to_device(img_wave)
        
        # Get image wave from image array.
        image_to_wave[blockspergrid,threadsperblock](img_array_d,img_wave_d)
        cuda.synchronize()
        img_wave_h=img_wave_d.copy_to_host()

        init_bound_abstract=np.zeros(img_array.shape[0]*img_array.shape[1],dtype=np.int32)
        init_bound_abstract_d=cuda.to_device(init_bound_abstract)
        
        # Initialize the abstract boundary.
        init_abstract[img_array.shape[0]*img_array.shape[1],1](img_array_d,init_bound_abstract_d)
        cuda.synchronize()
        init_bound_abstract_h=init_bound_abstract_d.copy_to_host()

        # Initialize the abstraction class
        abs=Abstract(img_wave_h,img_array.shape[0],init_bound_abstract_h,img_array.shape,False,abs_threshold)
        
        # Get the abstract points
        abs_points=abs.get_abstract_all()
        
        # Get the convexity array.
        abs_sign=abs.get_sign()
        abs_sign_d=cuda.to_device(abs_sign)
        white_count=np.count_nonzero(abs_sign==1)
        black_count=np.count_nonzero(abs_sign==-1)
        if white_count>black_count:
            invert=1
        else:
            invert=0
        print("len(abs_points)=",len(abs_points))

        abs_draw=decrement_by_one_cuda(abs_points)

        abs_draw_d=cuda.to_device(abs_draw)
        out_image=np.zeros(img_array.shape,dtype=np.int32)
        out_image_d=cuda.to_device(out_image)
        
        # Draw the abstract points to output image.
        draw_pixels_cuda2(abs_draw_d,abs_sign_d,invert,255,out_image_d)
        if i==0:
            out_image_hor_d=out_image_d
            img_array_rot=np.rot90(img_array,k=1,axes=(0,1))
            img_array=np.ascontiguousarray(img_array_rot)
        elif i==1:
            out_image_ver_h=out_image_d.copy_to_host()
            break
        i+=1
        
    out_image_ver_rot=np.ascontiguousarray(np.rot90(out_image_ver_h,k=3,axes=(0,1)))
    out_image_ver_d=cuda.to_device(out_image_ver_rot)
    threadsperblock=(16,16)
    blockspergrid_x=math.ceil(shape_orig[0]/threadsperblock[0])
    blockspergrid_y=math.ceil(shape_orig[1]/threadsperblock[1])
    blockspergrid=(blockspergrid_x,blockspergrid_y)
    
    # Combine the results of horizontal and vertical abstraction.
    clone_image[blockspergrid,threadsperblock](out_image_ver_d,out_image_hor_d,255)
    cuda.synchronize()
    out_image_h=out_image_hor_d.copy_to_host()
    
    # Save the output to disk.
    Image.fromarray(out_image_h).convert('RGB').save("output.png")
    print("Finished in total of",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
