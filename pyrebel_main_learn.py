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
import math,argparse,time,os
from pyrebel.preprocess import Preprocess
from pyrebel.abstract import Abstract
from pyrebel.learn import Learn
from pyrebel.utils import *

# This is a demo of abstraction of boundaries of blobs in the image.
# When you run this program the output is written to 'output.png'.
# The boundaries of blobs is in grey color and the abstract points are in white.
# The level of abstraction can be changed by giving '--threshold' argument. 
# The default value of threshold is 5.

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name.")
parser.add_argument("-t","--threshold",help="Threshold of abstraction.")
parser.add_argument("-l","--learn",help="Symbol to learn.")
parser.add_argument("-r","--recognize",help="Recognize the signature.")

args=parser.parse_args()
if args.threshold:
    abs_threshold=int(args.threshold)
else:
    abs_threshold=5

if args.learn:
    learn_n=0
    if args.learn[-1]!='/':
        learn_single=True
        sign_name=args.learn.split("/")[-1]
        ip_files_n=1
    else:
        learn_single=False
        ip_files=os.listdir(args.learn)
        ip_files_n=len(ip_files)
while 1:
    start_time=time.time()    
    if args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
    
    if args.recognize:
        img_array=np.array(Image.open(args.recognize).convert('L'))
    
    if args.learn:
        if learn_single:
            #img_array=open_image(args.learn).astype('int32')
            img_array=np.array(Image.open(args.learn).convert('L'))
        else:
            sign_name=ip_files[learn_n]
            #img_array=open_image(args.learn+ip_files[learn_n]).astype('int32')
            img_array=np.array(Image.open(args.learn+ip_files[learn_n]).convert('L'))
            
    # Initialize the preprocessing class.
    pre=Preprocess(img_array)
    # Set the minimum and maximum size of boundaries of blobs in the image. Defaults to a minimum of 64.
    pre.set_bound_size(32,50000000)    
    # Perform the preprocessing to get 1D array containing boundaries of blobs in the image.
    pre.preprocess_image()
    # Get the 1D array.
    bound_data=pre.get_bound_data()
    # Initialize the abstract boundary.
    init_bound_abstract=pre.get_init_abstract()
    # Get 1D array containing size of boundaries of blobs in the array.
    bound_size=pre.get_bound_size()
    print("len(bound_data)=",len(bound_data))

    scaled_shape=[img_array.shape[0]*3,img_array.shape[1]*3]
    
    # Initialize the abstraction class
    abs=Abstract(bound_data,len(bound_size),init_bound_abstract,scaled_shape,True,abs_threshold)
    
    # Initialize learning class
    l=Learn()
 
    print("len(know_base)=",len(l.get_know_base()))
    i=0
    n_layers=30
    while 1:
        if i==n_layers:
            break
        # Do one layer of abstraction
        is_finished=abs.do_abstract_one()
        ba_sign=abs.get_sign()
        ba_size=abs.get_bound_size() 
        # Get the signatures for the layer       
        l.get_signatures(ba_sign,ba_size)
        i+=1
        
    blob_index=2
    top_n=3
    if args.recognize:        
        print("symbols found=",l.recognize_symbols(blob_index,top_n))
    if args.learn:
        print("learning",sign_name,l.learn(blob_index,sign_name))
        l.write_know_base()  
    
    print("Finished in total of",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
    if args.learn:
        learn_n+=1
        if learn_n==ip_files_n:
            break
    if args.recognize:
        break
