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

# This is a demo of learning and recognizing of patterns using 
# abstraction of data.
#


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

init_time=time.time()
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
    bound_seed=pre.get_bound_seed()
    bound_seed_d=cuda.to_device(bound_seed)
    
    print("len(bound_data)=",len(bound_data))
    print("len(bound_size)=",len(bound_size))
    scaled_shape=pre.get_image_scaled().shape
    scaled_shape_d=cuda.to_device(scaled_shape)
    
    if len(bound_size)<3:
        print("No blobs found.")
        continue
    # Select largest blob
    blob_index=np.argsort(bound_size[2:])[-1]+2
    print("blob_index=",blob_index)
    
    bound_size_i_d=cuda.to_device(bound_size)
    increment_by_one[len(bound_size),1](bound_size_i_d)
    cuda.synchronize()
    bound_size_i=bound_size_i_d.copy_to_host()
    bound_size_i_cum_=np.cumsum(bound_size_i)
    bound_size_i_cum=np.delete(np.insert(bound_size_i_cum_,0,0),-1)
    bound_size_i_cum_d=cuda.to_device(bound_size_i_cum)
    
    out_image=np.zeros(scaled_shape,dtype=np.int32)
    out_image_d=cuda.to_device(out_image)
    
    
    # Find blobs not inside another blob
    is_inside=np.zeros(len(bound_size),dtype=np.int32)            
    is_inside_d=cuda.to_device(is_inside)
    threadsperblock=(16,16)
    blockspergrid_x=math.ceil(len(bound_size)/threadsperblock[0])
    blockspergrid_y=math.ceil(len(bound_size)/threadsperblock[1])
    blockspergrid=(blockspergrid_x,blockspergrid_y)
    is_blob_inside[blockspergrid,threadsperblock](bound_size_i_d,bound_size_i_cum_d,bound_data_d,bound_seed_d,scaled_shape_d,is_inside_d)
    cuda.synchronize()
    is_inside_h=is_inside_d.copy_to_host()
    blob_indices=[i for i, elem in enumerate(is_inside_h) if elem==0][2:]

    #print(is_inside_h,n_blobs_inside)
    for blob_ins in range(len(is_inside_h)):
        if not is_inside_h[blob_ins]:            
            blob_index_data=bound_data[bound_size_i_cum[blob_ins]:bound_size_i_cum[blob_ins]+bound_size_i[blob_ins]]
            blob_index_data_d=cuda.to_device(blob_index_data)            
            draw_pixels_cuda(blob_index_data_d,250,out_image_d)
            
    out_image_h=out_image_d.copy_to_host()            
    # Save the outer blob to disk.
    Image.fromarray(out_image_h).convert('RGB').save("inside.png")                    
    
    
    # Initialize the abstraction class
    abs=Abstract(bound_data,len(bound_size),init_bound_abstract,scaled_shape,True)
    
    n_layers=30
    # Initialize learning class
    l=Learn(n_layers,len(bound_size),3)
    print("len(know_base)=",len(l.get_know_base()))
    fst=time.time()
    while 1:
        # Do one layer of abstraction
        is_finished_abs=abs.do_abstract_one(abs_threshold)
        ba_sign=abs.get_sign()
        ba_size=abs.get_abstract_size()
        
        # Find signatures for the layer    
        is_finished=l.find_signatures2(ba_sign,ba_size)    
        if is_finished or is_finished_abs:
            break
            
    print("found signatures in",time.time()-fst)
    
    top_n=3
    if args.recognize:
        rt=time.time()
        recognized=l.recognize_sym(blob_indices,top_n)
        print("symbols found=")
        for i in recognized[0]:
            print(i)
        print(recognized[1])
        print("recognize time=",time.time()-rt)
        time.sleep(3)
    if args.learn:
        lt=time.time()
        signs=l.learn_sym(blob_indices,sign_name)
        sign_len=len(signs)
        print("learning",[sign_name],signs,"len(signs)=",sign_len)
        print("learn time=",time.time()-lt)
        l.write_know_base()  
    
    print("Finished in",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
    if args.learn:
        learn_n+=1
        if learn_n==ip_files_n:
            break
    if args.recognize:
        continue

print("Finished in total of",time.time()-init_time,"seconds.")

