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

from numba import cuda

def draw_pixels_cuda(pixels,i,img):
    # Draws 'pixels' to 'img' with color 'i'
    draw_pixels_cuda_[pixels.shape[0],1](pixels,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_cuda_(pixels,i,img):
    # CUDA version of function 'draw_pixels_cuda()'
    cc=cuda.grid(1)
    if cc<pixels.shape[0]:
        r=int(pixels[cc]/img.shape[1])
        c=pixels[cc]%img.shape[1]
        img[r][c]=i

@cuda.jit
def increment_by_one(array_d):
    # Increments each item in device array 'array_d' by one. cuda version.
    ci=cuda.grid(1)
    if ci<len(array_d):
        array_d[ci]+=1
        cuda.syncthreads()

@cuda.jit
def decrement_by_one(array_d):
    # Decrements each item in device array 'array_d' by one. cuda version
    ci=cuda.grid(1)
    if ci<len(array_d):
        array_d[ci]-=1
        cuda.syncthreads()
        
def decrement_by_one_cuda(array):
    # Decrements each item in 'array' by one.
    array_d=cuda.to_device(array)
    decrement_by_one[len(array),1](array_d)
    cuda.synchronize()
    return array_d.copy_to_host()    

def draw_pixels_from_indices_cuda(indices,pixels,i,img):
    # Draws pixels from 'pixels' with indices 'indices' to image 'img' with color 'i'
    draw_pixels_from_indices_cuda_[indices.shape[0],1](indices,pixels,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_from_indices_cuda_(indices,pixels,i,img):
    # CUDA version of function 'draw_pixels_from_indices_cuda()'
    cc=cuda.grid(1)
    if cc<len(indices):
        r=int(pixels[indices[cc]]/img.shape[1])
        c=pixels[indices[cc]]%img.shape[1]
        img[r][c]=i
        
@cuda.jit
def image_to_wave(img_array_d,img_wave_pre_init_d):
    # Plot each row of 'img_array_d' in 2D space with color of pixels as y-coordinate.
    r,c=cuda.grid(2)
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        img_wave_pre_init_d[r*img_array_d.shape[1]+c]=img_array_d[r][c]*img_array_d.shape[1]+c

@cuda.jit
def init_abstract(img_array_d,bound_abstract_pre_d):
    ci=cuda.grid(1)
    if ci==0:
        bound_abstract_pre_d[ci]=ci+1
    elif ci<len(bound_abstract_pre_d) and ci%img_array_d.shape[1]==0:
        bound_abstract_pre_d[ci]=ci+1
        bound_abstract_pre_d[ci-1]=ci
    elif ci==len(bound_abstract_pre_d)-1:
        bound_abstract_pre_d[ci]=ci+1
        
def draw_pixels_cuda2(pixels,exclusions,invert,i,img):
    # Draws 'pixels' to image 'img' with 'exclusions' with color 'i'
    draw_pixels_cuda2_[pixels.shape[0],1](pixels,exclusions,invert,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_cuda2_(pixels,exclusions,invert,i,img):
    # CUDA version of function 'draw_pixels_cuda2()'
    cc=cuda.grid(1)
    if cc<pixels.shape[0]:
        if invert:
            if exclusions[cc]<0:
                r=int(pixels[cc]/img.shape[1])
                c=pixels[cc]%img.shape[1]
                img[r][c]=i
        else:
            if exclusions[cc]>0:
                r=int(pixels[cc]/img.shape[1])
                c=pixels[cc]%img.shape[1]
                img[r][c]=i
 
@cuda.jit
def clone_image(img_array,img_clone,color):
    # draws pixels in 'img_array' with color 'color' to 'img_clone'
    r,c=cuda.grid(2)
    if r<img_array.shape[0] and c<img_array.shape[1]:
        if img_array[r][c]==color:
            img_clone[r][c]=color

@cuda.jit
def clone_image2(img_array_orig,image_to_clone,img_cloned,inv):
    # draws pixels in 'image_to_clone' with color '255' to 'img_cloned' with the color 
    # of corresponding pixels in 'img_array_orig'
    r,c=cuda.grid(2)
    if r>0 and r<img_array_orig.shape[0] and c>0 and c<img_array_orig.shape[1]:
        if image_to_clone[r][c]==255:
            if inv:
                img_cloned[r][c]=img_array_orig[r][c]
            else:
                img_cloned[r][c]=255-img_array_orig[r][c]
            #cuda.atomic.add(count,0,1)

