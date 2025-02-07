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
from pyrebel.edge import Edge
from pyrebel.utils import *

# This is a demo of edge detection using abstraction of data.
# When you run this program the output is written to 'output.png'.
# The level of abstraction can be changed by giving '--edge_threshold' argument. 
# The default value of threshold is 5.
#
# The edges are detected by abstracting the image horizontally,vertically and
# diagonally. Then, they are combined to get the final result.
#

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name.")
parser.add_argument("-t","--edge_threshold",help="Threshold of abstraction.")
args=parser.parse_args()
if args.edge_threshold:
    edge_threshold=int(args.edge_threshold)
else:
    edge_threshold=5
 
while 1:
    start_time=time.time()    
    if args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
    else:
        print("No input file.")
    
    
    edge=Edge(img_array)
    edge.find_edges(edge_threshold)
    edges=edge.get_edges()  # Or edges=edge.get_edges_bw() for black and white.
    
    # Save the output to disk.
    Image.fromarray(edges).convert('RGB').save("output.png")
    print("Finished in",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
    break
