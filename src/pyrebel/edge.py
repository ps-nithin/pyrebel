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

from pyrebel.abstract import Abstract
from pyrebel.utils import *
import math
import numpy as np

class Edge:
    def __init__(self,img_array,invert=-1):
        self.img_array_orig=img_array
        self.invert_arg=invert
        self.out_image_hor_h=[]
        self.final_image_h=[]
        
    def find_edges(self,abs_threshold):
        img_array=self.img_array_orig
        shape_orig=self.img_array_orig.shape
        i=0
        while 1:
            threadsperblock=(16,16)
            blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
            blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
            blockspergrid=(blockspergrid_x,blockspergrid_y)
            img_array_d=cuda.to_device(img_array)
            img_rev_rot45=np.zeros(img_array.shape,dtype=np.int32)
            img_rev_rot45_d=cuda.to_device(img_rev_rot45)
            if img_array.shape[0]<=img_array.shape[1]:
                img_rot45=np.full([img_array.shape[0]+img_array.shape[1]-1,img_array.shape[0]],255,dtype=np.int32)
                img_rot45_mask=np.full([img_array.shape[0]+img_array.shape[1]-1,img_array.shape[0]],-500,dtype=np.int32)
            else:
                img_rot45=np.full([img_array.shape[0]+img_array.shape[1]-1,img_array.shape[1]],255,dtype=np.int32)           
                img_rot45_mask=np.full([img_array.shape[0]+img_array.shape[1]-1,img_array.shape[1]],-500,dtype=np.int32)           
            
            img_array_flip=np.flip(img_array,0)
            img_array_flip=np.ascontiguousarray(np.flip(img_array_flip,1))
            img_array_flip_d=cuda.to_device(img_array_flip)
            
            img_rot45_d=cuda.to_device(img_rot45)
            img_rot45_mask_d=cuda.to_device(img_rot45_mask)
            image_rotate45[blockspergrid,threadsperblock](img_array_flip_d,img_rot45_d,img_rot45_mask_d)
            cuda.synchronize()
            img_rot45_h=img_rot45_d.copy_to_host()
            img_rot45_h=np.ascontiguousarray(np.flip(img_rot45_h,0))
            img_rot45_d=cuda.to_device(img_rot45_h)
            
            threadsperblock=(16,16)
            blockspergrid_x45=math.ceil(img_rot45.shape[0]/threadsperblock[0])
            blockspergrid_y45=math.ceil(img_rot45.shape[1]/threadsperblock[1])
            blockspergrid_rot45=(blockspergrid_x45,blockspergrid_y45)
            fill_column_zero[blockspergrid_rot45,threadsperblock](img_rot45_d)
            cuda.synchronize()
            
            image_rotate45[blockspergrid,threadsperblock](img_array_d,img_rot45_d,img_rot45_mask_d)
            cuda.synchronize()
            
            img_wave45=np.zeros(img_rot45_h.shape[0]*img_rot45_h.shape[1],dtype=np.int32)
            img_wave45_d=cuda.to_device(img_wave45)
            
            # Get image wave from image array.
            image_to_wave[blockspergrid_rot45,threadsperblock](img_rot45_d,img_wave45_d)
            cuda.synchronize()
            img_wave45_h=img_wave45_d.copy_to_host()
            
            init_bound_abstract=np.zeros(img_rot45_h.shape[0]*img_rot45_h.shape[1],dtype=np.int32)
            init_bound_abstract_d=cuda.to_device(init_bound_abstract)
            
            # Initialize the abstract boundary.
            init_abstract[img_rot45_h.shape[0]*img_rot45_h.shape[1],1](img_rot45_d,init_bound_abstract_d)
            cuda.synchronize()
            init_bound_abstract_h=init_bound_abstract_d.copy_to_host()
            
            # Initialize the abstraction class
            abs45=Abstract(img_wave45_h,img_rot45_h.shape[0],init_bound_abstract_h,img_rot45_h.shape,False)
            
            # Get the abstract points
            abs45.do_abstract_all(abs_threshold)
            abs_points45=abs45.get_abstract()
            
            # Get the convexity array.
            abs_sign45=abs45.get_sign()
            abs_sign45_d=cuda.to_device(abs_sign45)
            if self.invert_arg==-1:
                white_count45=np.count_nonzero(abs_sign45==1)
                black_count45=np.count_nonzero(abs_sign45==-1)
                if white_count45>black_count45:
                    invert45=True
                else:
                    invert45=False
                self.invert_arg=invert45
            print("len(abs_points)=",len(abs_points45),"diagonal",i)

            abs_draw45=decrement_by_one_cuda(abs_points45)

            abs_draw45_d=cuda.to_device(abs_draw45)
            out_image45=np.zeros(img_rot45_h.shape,dtype=np.int32)
            out_image45_d=cuda.to_device(out_image45)
            
            # Draw the abstract points to output image.
            draw_pixels_cuda2(abs_draw45_d,abs_sign45_d,self.invert_arg,255,out_image45_d)
            cuda.synchronize()
            
            image_rev_rotate45[blockspergrid_rot45,threadsperblock](out_image45_d,img_rot45_mask_d,img_rev_rot45_d)
            cuda.synchronize()
            
            if i==0:
                out_image_45_1_d=img_rev_rot45_d
                img_array_rot=np.rot90(img_array,k=1,axes=(0,1))
                img_array=np.ascontiguousarray(img_array_rot)
            elif i==1:
                out_image_45_2_h=img_rev_rot45_d.copy_to_host()
                break
            i+=1
        
        out_image_45_2_rot=np.ascontiguousarray(np.rot90(out_image_45_2_h,k=3,axes=(0,1)))
        out_image_45_2_d=cuda.to_device(out_image_45_2_rot)
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(shape_orig[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(shape_orig[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        
        # Combine the results of horizontal and vertical abstraction.
        clone_image[blockspergrid,threadsperblock](out_image_45_2_d,out_image_45_1_d,255)
        cuda.synchronize()
        
        # Start of vertical and horizontal abstraction
        
        i=0
        img_array=self.img_array_orig
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
            abs=Abstract(img_wave_h,img_array.shape[0],init_bound_abstract_h,img_array.shape,False)
            
            # Get the abstract points
            abs.do_abstract_all(abs_threshold)
            abs_points=abs.get_abstract()
            
            # Get the convexity array.
            abs_sign=abs.get_sign()
            abs_sign_d=cuda.to_device(abs_sign)
            white_count=np.count_nonzero(abs_sign==1)
            black_count=np.count_nonzero(abs_sign==-1)
            if self.invert_arg==-1:
                if white_count>black_count:
                    invert=True
                else:
                    invert=False
                self.invert_arg=invert
            print("len(abs_points)=",len(abs_points),"perpendicular",i)

            abs_draw=decrement_by_one_cuda(abs_points)

            abs_draw_d=cuda.to_device(abs_draw)
            out_image=np.zeros(img_array.shape,dtype=np.int32)
            out_image_d=cuda.to_device(out_image)
            
            # Draw the abstract points to output image.
            draw_pixels_cuda2(abs_draw_d,abs_sign_d,self.invert_arg,255,out_image_d)
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
        
        clone_image[blockspergrid,threadsperblock](out_image_45_1_d,out_image_hor_d,255)
        cuda.synchronize()
        clean_quant_img[blockspergrid,threadsperblock](out_image_hor_d)
        cuda.synchronize()
        self.out_image_hor_h=out_image_hor_d.copy_to_host()
        img_array_orig_d=cuda.to_device(self.img_array_orig)
        final_image=np.zeros(self.img_array_orig.shape,dtype=np.int32)
        final_image_d=cuda.to_device(final_image)
        clone_image2[blockspergrid,threadsperblock](img_array_orig_d,out_image_hor_d,final_image_d,not self.invert_arg)
        cuda.synchronize()
        self.final_image_h=final_image_d.copy_to_host()
        
    def get_edges_bw(self):
        return self.out_image_hor_h
    def get_edges(self):
        return self.final_image_h
