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
from pyrebel.edge import Edge
from numba import cuda
from pyrebel.utils import *

# This is a demo of abstract painting. The output of edge detection is 
# painted to obtain the result.
#
                                                
parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name.")
parser.add_argument("-et","--edge_threshold",help="Threshold of edge detection.")
parser.add_argument("-b","--block_threshold",help="Block threshold.")
parser.add_argument("-p","--paint_threshold",help="Paint threshold.")
args=parser.parse_args()

if args.edge_threshold:
    edge_threshold=int(args.edge_threshold)
else:
    edge_threshold=10
if args.block_threshold:
    block_threshold=int(args.block_threshold)
else:
    block_threshold=20
if args.paint_threshold:
    paint_threshold=int(args.paint_threshold)
else:
    paint_threshold=5
      
while 1:
    start_time=time.time()    
    if args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
        img_array_rgb=np.array(Image.open(args.input).convert('RGB'))
    else:
        print("No input file.")
    img_array_rgb_d=cuda.to_device(img_array_rgb)
    edge=Edge(img_array,False)
    edge.find_edges(edge_threshold)
    edges=edge.get_edges()
    edges_img_d=cuda.to_device(edges)    
    block_img=np.zeros(img_array_rgb.shape,dtype=np.uint8)
    block_img_d=cuda.to_device(block_img)
    threadsperblock=(16,16)
    blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
    blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
    blockspergrid=(blockspergrid_x,blockspergrid_y)
    
    k=0
    while 1:
        n=block_threshold
        while 1:
            draw_blocks[blockspergrid,threadsperblock](img_array_rgb_d,edges_img_d,block_img_d,n)
            cuda.synchronize()
            if n<2:
                break
            n-=1
        k+=1
        print(k,"/",paint_threshold)
        if k==paint_threshold:
            break
        img_array_rgb_d=block_img_d
    block_img_h=block_img_d.copy_to_host()
    
    # Save the output to disk.
    Image.fromarray(block_img_h).save("output.png")
    print("Finished in",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
    break
    
