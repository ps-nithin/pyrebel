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

# This is a demo of learning and recognizing of patterns using abstraction of data. 
# The demo is similar to `pyrebel_main_learn_multiple.py` but uses edge detection to obtain patterns.
# Thus, it works with any images like a USB, CSI camera or an image file. The program settings can be changed using
# keyboard shortcuts as defined in `on_press()`.
# 

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import math,argparse,time,os,itertools
from pynput import keyboard
import sys,termios
from pyrebel.preprocess import Preprocess
from pyrebel.abstract import Abstract
from pyrebel.learn import Learn
from pyrebel.edge import Edge
from pyrebel.utils import *
        
parser=argparse.ArgumentParser()
parser.add_argument("-t","--abs_threshold",help="Threshold of abstraction.")
parser.add_argument("-l","--learn",help="Symbol to learn.")
parser.add_argument("-r","--recognize",help="Recognize the signature.")
parser.add_argument("-c","--camera",help="Learn/recognize camera.")
parser.add_argument("-et","--edge_threshold",help="Threshold of edge detection.")
parser.add_argument("-b","--block_threshold",help="Block threshold.")
parser.add_argument("-p","--paint_threshold",help="Paint threshold.")
parser.add_argument("-s","--scale_factor",help="Scaling factor.")
parser.add_argument("-cw","--crop_width",help="Crop width.")
parser.add_argument("-cr","--crop_r",help="Crop position from top.")
parser.add_argument("-cc","--crop_c",help="Crop position from left.")
parser.add_argument("-or","--output_resolution",help="Output resolution.")

args=parser.parse_args()
if args.abs_threshold:
    abs_threshold=np.float64(args.abs_threshold)
else:
    abs_threshold=-1
if args.edge_threshold:
    edge_threshold=int(args.edge_threshold)
else:
    edge_threshold=20
if args.block_threshold:
    block_threshold=int(args.block_threshold)
else:
    block_threshold=10
if args.paint_threshold:
    paint_threshold=int(args.paint_threshold)
else:
    paint_threshold=5
if args.scale_factor:
    scale_factor=int(args.scale_factor)
else:
    scale_factor=2
if args.crop_width:
    crop_width=int(args.crop_width)
else:
    crop_width=-1
if args.crop_r:
    crop_r=int(args.crop_r)
else:
    crop_r=1.5
if args.crop_c:
    crop_c=int(args.crop_c)
else:
    crop_c=1.5
if args.output_resolution:
    OUTPUT_RESOLUTION=int(args.output_resolution)
else:
    OUTPUT_RESOLUTION=600   

def on_press(key):
    global scale_factor
    global crop_r,crop_c,crop_width
    global RESOLUTION_THRESHOLD
    global edge_threshold
    global learn
    step=crop_width//15
    
    # `l` key turns on learning mode
    if key==keyboard.KeyCode.from_char('l'):
        learn=True
        return False
        
    # `up arrow` key moves the crop position up.
    if key==keyboard.Key.up:
        if crop_r>0:
            crop_r-=step
            
    # `down arrow` key moves the crop position down.
    if key==keyboard.Key.down:
        if crop_r<img_height:
            crop_r+=step
            
    # `left arrow` key moves the crop position left.
    if key==keyboard.Key.left:
        if crop_c>0:
            crop_c-=step
            
    # `right arrow` key moves the crop position right.
    if key==keyboard.Key.right:
        if crop_c<img_width:
            crop_c+=step  
            
    # `+` Plus key zooms in the crop
    if key==keyboard.KeyCode.from_char('+'):
        if scale_factor<12:
            scale_factor+=1
            
    # `-` Minus key zooms out the crop
    if key==keyboard.KeyCode.from_char('-'):
        if scale_factor>1:
            scale_factor-=1
            
    # `t` Lowercase t increases `edge_threshold`
    if key==keyboard.KeyCode.from_char('t'):
        if edge_threshold<100:
            edge_threshold+=5
            
    # `T` Uppercase T decreases `edge_threshold`
    if key==keyboard.KeyCode.from_char('T'):
        if edge_threshold>10:
            edge_threshold-=5
            
    # `r` Lowercase r increases `RESOLUTION_THRESHOLD`
    if key==keyboard.KeyCode.from_char('r'):
        if RESOLUTION_THRESHOLD<0.25:
            RESOLUTION_THRESHOLD+=0.02
            
    # `R` Uppercase R decreases `RESOLUTION_THRESHOLD`
    if key==keyboard.KeyCode.from_char('R'):
        if RESOLUTION_THRESHOLD>0.02:
            RESOLUTION_THRESHOLD-=0.02
    print("crop_r",crop_r,"crop_c",crop_c,"scale_factor",scale_factor,"res_threshold",RESOLUTION_THRESHOLD,"edge_threshold",edge_threshold)

def learn_recognize(img_array,learn,sign_name=""):
    start_time=time.time()
    # Initialize the preprocessing class.
    pre=Preprocess(img_array)
    # Set the minimum and maximum size of boundaries of blobs in the image. Defaults to a minimum of 64.
    pre.set_bound_size(600)    
    # Perform the preprocessing to get 1D array containing boundaries of blobs in the image.
    pre.preprocess_image()
    img_scaled=pre.get_image_scaled()
    nz_a=pre.get_bound_seed()
    # Get the 1D array.
    bound_data=pre.get_bound_data()
    bound_data_d=cuda.to_device(bound_data)
    
    # Initialize the abstract boundary.
    init_bound_abstract=pre.get_init_abstract()
    # Get 1D array containing size of boundaries of blobs in the array.
    bound_size=pre.get_bound_size()
    bound_seed=pre.get_bound_seed()
    bound_seed_d=cuda.to_device(bound_seed)
    
    # Assign threshold of abstraction
    global RESOLUTION_THRESHOLD
    if abs_threshold==-1:
        threshold_h=pre.get_max_dist()*RESOLUTION_THRESHOLD
    else:
        threshold_h=np.full(len(bound_size),abs_threshold,dtype=np.float64)
        
    print("len(bound_data)=",len(bound_data))
    print("len(bound_size)=",len(bound_size))
    scaled_shape=pre.get_image_scaled().shape
    scaled_shape_d=cuda.to_device(scaled_shape)
    
    if len(bound_size)<3:
        print("No blobs found.")
        return

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
    blob_indices=[i for i, elem in enumerate(is_inside_h) if elem==1]

    #print(is_inside_h,n_blobs_inside)
    for blob_ins in range(len(is_inside_h)):
        if is_inside_h[blob_ins]==1:            
            blob_index_data=bound_data[bound_size_i_cum[blob_ins]:bound_size_i_cum[blob_ins]+bound_size_i[blob_ins]]
            blob_index_data_d=cuda.to_device(blob_index_data)            
            draw_pixels_cuda(blob_index_data_d,250,out_image_d)
            
    out_image_h=out_image_d.copy_to_host()            
    # Save the outer blob to disk.
    Image.fromarray(out_image_h).convert('RGB').save("inside.png")                    
    
    if learn:
        if len(blob_indices)>1:
            resume=input("Multiple blobs. Do you want to continue? (y/n) : ")
            if resume.lower()=="n" or resume=="":
                print("Skipping..")
                return    
    # Initialize the abstraction class
    abs=Abstract(bound_data,len(bound_size),init_bound_abstract,scaled_shape,True,threshold_h)
    
    n_layers=30
    # Initialize learning class
    l=Learn(n_layers,len(bound_size),3)
    print("len(know_base)=",len(l.get_know_base()))
    fst=time.time()
    while 1:
        # Do one layer of abstraction
        is_finished_abs=abs.do_abstract_one()
        ba_sign=abs.get_sign()
        ba_size=abs.get_abstract_size()
        
        # Find signatures for the layer    
        is_finished=l.find_signatures(ba_sign,ba_size)    
        if is_finished or is_finished_abs:
            ist=time.time()
            l.init_signatures()
            print("init signatures in",time.time()-ist)        
            break
            
    print("found signatures in",time.time()-fst)
    
    top_n=-1
    if not learn:
        rt=time.time()
        print(blob_indices)
        recognized=l.recognize_sym(blob_indices,top_n,"image")
        print("symbols found=")
        if recognized==None or len(recognized)==0:
            return
        for i in recognized[0]:
            print(i)
        print(recognized[1])
        print("recognize time=",time.time()-rt)
        draw_image=Image.fromarray(img_scaled).convert('RGB')
        draw = ImageDraw.Draw(draw_image)
        text_color=(255,255,0)
        font=ImageFont.load_default(size=30)
        for i,blob_i in enumerate(blob_indices):
            if len(list(itertools.chain.from_iterable(recognized[0][i])))==0:
                continue
            pos_i=nz_a[blob_i]
            r=int(pos_i/scaled_shape[1])
            c=pos_i%scaled_shape[1]           
            text=recognized[0][i][0][0]+"("+str(recognized[0][i][0][1])+")"
            draw.text((c,r),text,fill=text_color,font=font)
        draw_image.save('output_text.png')
        #time.sleep(3)
    if learn:
        lt=time.time()
        signs=l.learn_sym(blob_indices,sign_name.split(".")[0],"image")
        sign_len=len(signs)
        print("learning",[sign_name.split(".")[0]],signs,"len(signs)=",sign_len)
        print("learn time=",time.time()-lt)
        l.write_know_base()  
    
    print("Finished learn/recognize in",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")


def paint_edges(img_array_rgb,edges,block_threshold,paint_threshold):
    img_array_rgb_d=cuda.to_device(img_array_rgb)
    edges_img_d=cuda.to_device(edges)
    block_img=img_array_rgb
    block_img_d=cuda.to_device(block_img)
    threadsperblock=(16,16)
    blockspergrid_x=math.ceil(edges.shape[0]/threadsperblock[0])
    blockspergrid_y=math.ceil(edges.shape[1]/threadsperblock[1])
    blockspergrid=(blockspergrid_x,blockspergrid_y)
    
    k=0
    while 1:
        n=block_threshold
        while 1:
            draw_blocks_edges[blockspergrid,threadsperblock](img_array_rgb_d,edges_img_d,block_img_d,n)
            cuda.synchronize()
            if n<2:
                break
            n-=1
            img_array_rgb_d=cuda.to_device(block_img_d.copy_to_host())
        k+=1
        print("painting",k,"/",paint_threshold,end='\r')
        if k==paint_threshold:
            print()
            break        
    block_img_h=block_img_d.copy_to_host()
    return block_img_h

@cuda.jit
def clone_image_rgb(img_array_orig,image_to_clone,img_cloned,color):
    """Draws pixels in 'image_to_clone' with color '255' to 'img_cloned' with the color of corresponding pixels in 'img_array_orig'"""
    
    r,c=cuda.grid(2)
    if r>0 and r<img_array_orig.shape[0] and c>0 and c<img_array_orig.shape[1]:
        if image_to_clone[r][c]==color:
            img_cloned[r][c][0]=img_array_orig[r][c][0]
            img_cloned[r][c][1]=img_array_orig[r][c][1]
            img_cloned[r][c][2]=img_array_orig[r][c][2]
            #cuda.atomic.add(count,0,1)

@cuda.jit
def join_edges(edges_d,out_image_d,threshold,color):
    r,c=cuda.grid(2)
    if threshold<r<edges_d.shape[0]-threshold-1 and threshold<c<edges_d.shape[1]-threshold-1 and edges_d[r][c]==color:
        #if edges_d[r-1][c]==edges_d[r+1][c]==color or edges_d[r][c-1]==edges_d[r][c+1]==color:
        #    return
        for rrr in range(r-threshold,r+threshold+1):
            for ccc in range(c-threshold,c+threshold+1):
                if edges_d[rrr][ccc]==color:
                    #if edges_d[rrr-1][ccc]==edges_d[rrr+1][ccc]==color or edges_d[rrr][ccc-1]==edges_d[rrr][ccc+1]==color:
                    #    return
                    found_wall=False
                    x=c
                    y=r
                    cc=ccc
                    rr=rrr
                    dx=abs(x-cc)
                    dy=abs(y-rr)
                    sx=1 if cc<x else -1
                    sy=1 if rr<y else -1
                    err=dx-dy
                    while True:                        
                        if cc==x and rr==y:
                            break
                        e2=2*err
                        if e2>-dy:
                            err-=dy
                            cc+=sx
                        elif e2<dx:
                            err+=dx
                            rr+=sy
                        if edges_d[rr][cc]!=0 and edges_d[rr][cc]!=color:
                            found_wall=True
                            break
                    if found_wall:
                        continue
                    x=c
                    y=r
                    cc=ccc
                    rr=rrr
                    dx=abs(x-cc)
                    dy=abs(y-rr)
                    sx=1 if cc<x else -1
                    sy=1 if rr<y else -1
                    err=dx-dy
                    while True:
                        out_image_d[rr][cc]=color
                        if cc==x and rr==y:
                            break
                        e2=2*err
                        if e2>-dy:
                            err-=dy
                            cc+=sx
                        elif e2<dx:
                            err+=dx
                            rr+=sy

def crop_center(img, crop_width, crop_height, pos_r, pos_c):
    """
    Crops image to the specified width and height at the specified positon.
    """
    width, height = img.size
    #print(pos_r,pos_c)
    left=0 if (pos_c-crop_width//2)<0 else pos_c-crop_width//2
    top=0 if (pos_r-crop_height//2)<0 else pos_r-crop_height//2    
    if (left+crop_width)>width:
        right=width
        left=width-crop_width
    else:
        right=left+crop_width
    if (top+crop_height)>height:
        bottom=height
        top=bottom-crop_height
    else:
        bottom=top+crop_height
    #print(left,top,right,bottom)
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img
    
def process_image_edges(img_array_rgb_orig):
    start_time=time.time()
    img_array=np.array(Image.fromarray(img_array_rgb_orig).convert('L'))
    img_array_rgb_orig_d=cuda.to_device(img_array_rgb_orig)
    
    edge1=Edge(img_array)
    edge1.find_edges(edge_threshold)
    edges1=edge1.get_edges_both()
    edges_input_d=cuda.to_device(edges1)

    threadsperblock=(32,32)
    blockspergrid=(math.ceil(img_array.shape[0]/32),math.ceil(img_array.shape[1]/32))
    join_count=10
    n=1
    join_threshold=1
    depth_n=0
    depth_count=1
    while 1:
        out_image_d=cuda.to_device(edges_input_d.copy_to_host())
        join_edges[blockspergrid,threadsperblock](edges_input_d,out_image_d,join_threshold,100)
        cuda.synchronize()
        edges_input2_d=cuda.to_device(out_image_d.copy_to_host())    
        out_image2_d=cuda.to_device(edges_input2_d.copy_to_host())
        join_edges[blockspergrid,threadsperblock](edges_input2_d,out_image2_d,join_threshold,255)
        cuda.synchronize()
        edges_input_d=cuda.to_device(out_image2_d.copy_to_host())
        print("join count",n,"/",join_count,end='\r')
        if n==join_count:
            n=1
            depth_n+=1
            print()
        if depth_n==depth_count:
            break
        n+=1
        join_threshold+=1

    edges_joined=out_image2_d.copy_to_host()
    
    edges_single=np.zeros(img_array.shape,dtype=np.uint8)
    edges_single_d=cuda.to_device(edges_single)
    clone_image[blockspergrid,threadsperblock](out_image2_d,edges_single_d,100)
    cuda.synchronize()
    edges_single_h=edges_single_d.copy_to_host()
    
    edges_rgb_d=cuda.to_device(np.full(img_array_rgb_orig.shape,0,dtype=np.uint8))
    #clone_image_rgb[blockspergrid,threadsperblock](img_array_rgb_orig_d,out_image2_d,edges_rgb_d,255)
    #cuda.synchronize()
    clone_image_rgb[blockspergrid,threadsperblock](img_array_rgb_orig_d,out_image2_d,edges_rgb_d,100)
    cuda.synchronize()

    edges_rgb_h=edges_rgb_d.copy_to_host()
    edges_painted=paint_edges(edges_rgb_h,edges_joined,block_threshold,5)
        
    # Save the output to disk.    
    Image.fromarray(edges_painted).convert('RGB').save("edges_painted.png")
    Image.fromarray(edges_joined).convert('RGB').save("output_joined.png")
    Image.fromarray(edges_single_h).convert('RGB').save("output.png")
    Image.fromarray(edges1).convert('RGB').save("edges.png")
    Image.fromarray(img_array_rgb_orig).convert('RGB').save("frame.png")
    print("Finished edges in",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")  
    return edges_single_h

def learn_input(learn_args):
    learn_n=0
    if learn_args[-1]!='/':
        learn_single=True
        sign_name=learn_args.split("/")[-1]
        ip_files_n=1
    else:
        learn_single=False
        ip_files=os.listdir(learn_args)
        ip_files_n=len(ip_files)
      
    while 1:
        if learn_single:
            #img_array=open_image(args.learn).astype('int32')
            img_array_rgb=np.array(Image.open(learn_args).convert('RGB'))
        else:
            sign_name=ip_files[learn_n]
            #img_array=open_image(args.learn+ip_files[learn_n]).astype('int32')
            img_array_rgb=np.array(Image.open(learn_args+ip_files[learn_n]).convert('RGB'))
        
        global img_height
        global img_width
        global scale_factor
        global crop_r,crop_c
        img_height=img_array_rgb.shape[0]
        img_width=img_array_rgb.shape[1]
        if crop_r==1.5:
            crop_r=img_height//2
        if crop_c==1.5:
            crop_c=img_width//2    
        crop_width=OUTPUT_RESOLUTION//scale_factor
        if OUTPUT_RESOLUTION>img_width or OUTPUT_RESOLUTION>img_width:
            crop_width=-1
            crop_r=-1
            crop_c=-1
        print("crop_width=",crop_width)
        start_time=time.time()
        img_array_scaled=crop_and_scale_frame(img_array_rgb,scale_factor,crop_width,crop_r,crop_c)
        img_edges=process_image_edges(img_array_scaled)  
        learn_recognize(img_edges,True,sign_name)
        print("Finished in total of",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")
        learn_n+=1
        if learn_n==ip_files_n:
            break
            
def recognize_input(recognize_args):
    global img_height
    global img_width
    global scale_factor
    global crop_r,crop_c
    img_array_rgb=np.array(Image.open(recognize_args).convert('RGB'))
    img_height=img_array_rgb.shape[0]
    img_width=img_array_rgb.shape[1]
    if crop_r==1.5:
        crop_r=img_height//2
    if crop_c==1.5:
        crop_c=img_width//2 
    while 1:
        start_time=time.time()        
        crop_width=OUTPUT_RESOLUTION//scale_factor
        if OUTPUT_RESOLUTION>img_width or OUTPUT_RESOLUTION>img_width:
            crop_width=-1
            crop_r=-1
            crop_c=-1
        print("crop_width=",crop_width)
        img_array_scaled=crop_and_scale_frame(img_array_rgb,scale_factor,crop_width,crop_r,crop_c)
        
        img_edges=process_image_edges(img_array_scaled)  
        learn_recognize(img_edges,False)
        print("Finished in total of",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")  
        #break

def crop_and_scale_frame(img_array_rgb_orig,scale_factor=1,crop_width=-1,crop_r=-1,crop_c=-1):
    #scale_factor=int(2048/max(img_array_rgb_orig.shape))
    print("scale=",scale_factor)
    if crop_width==-1:
        crop_width_=img_array_rgb_orig.shape[1]
        crop_height_=img_array_rgb_orig.shape[0]        
    else:
        crop_height_=crop_width
        crop_width_=crop_width
    if crop_r==-1:
        crop_r_=img_array_rgb_orig.shape[0]//2
    else:
        crop_r_=crop_r
    if crop_c==-1:
        crop_c_=img_array_rgb_orig.shape[1]//2
    else:
        crop_c_=crop_c
    #img_array_i=crop_center(Image.fromarray(img_array_rgb_orig).convert('L'),crop_width,crop_height)
    img_array_rgb_i=crop_center(Image.fromarray(img_array_rgb_orig),crop_width_,crop_height_,crop_r_,crop_c_)
    
    #img_array=np.array(img_array_i.resize((int(img_array_rgb_i.width*scale_factor),int(img_array_rgb_i.height*scale_factor)),Image.LANCZOS))
    img_array_rgb=np.array(img_array_rgb_i.resize((int(img_array_rgb_i.width*scale_factor),int(img_array_rgb_i.height*scale_factor)),Image.LANCZOS))
    return img_array_rgb

def process_camera_gst():
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    # Initialize GStreamer
    Gst.init(None)
    global learn
    # GStreamer pipeline string for CSI camera
    csi_pipeline=" ! ".join([
        "nvarguscamerasrc sensor-id=1",  # Adjust sensor-id if multiple cameras
        "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1024, format=(string)NV12, framerate=(fraction)30/1",
        "nvvidconv",
        "video/x-raw, format=(string)BGRx",
        "videoconvert",
        "video/x-raw, format=(string)RGB",
        "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        ])
        
    # GStreamer pipeline string for USB camera
    usb_pipeline=" ! ".join([
        "v4l2src device=/dev/video0 ! video/x-raw, width=1920,height=1080",
        "videoconvert",
        "video/x-raw, format=(string)RGB", # Add `videoflip method=rotate-180` to rotate camera.
        "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        ])
        
    # Create pipeline
    pipeline = Gst.parse_launch(usb_pipeline) # Choose USB or CSI camera pipeline string as required
    appsink = pipeline.get_by_name("sink")

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    time.sleep(3)
    # Main loop
    while 1:
        start_time=time.time()
        sample = appsink.emit("try-pull-sample", 1000*Gst.MSECOND)
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_value("width")
            height = caps.get_structure(0).get_value("height")

            # Map buffer to numpy array
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                frame = frame_data.reshape((height, width, 3))  # BGR format
                # Now `frame` is a NumPy array (height, width, 3)
                #print(f"Captured frame: {frame.shape}")
                buffer.unmap(map_info)            
                Image.fromarray(frame).convert('RGB').save("camera.png")                
                global img_width
                global img_height
                global scale_factor
                global crop_r,crop_c,crop_width
                global learn,listener,sign_name
                img_width=width
                img_height=height
                if crop_r==1.5:
                    crop_r=img_height//2
                if crop_c==1.5:
                    crop_c=img_width//2
                crop_width=OUTPUT_RESOLUTION//scale_factor
                print("crop_width=",crop_width)
                frame_scaled=crop_and_scale_frame(frame,scale_factor,crop_width,crop_r,crop_c)                                
                img_edges=process_image_edges(frame_scaled)
                if learn:
                    termios.tcflush(sys.stdin, termios.TCIFLUSH)
                    sign_name=input("Enter pattern name (blank to skip):")
                    if sign_name=="":
                        print("Skipping..")
                        learn=False
                        if not listener.is_alive():
                            listener = keyboard.Listener(on_press=on_press)
                            listener.start()
                        print("Finished in total of",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.\n") 
                        continue
                    learn_recognize(img_edges,learn,sign_name)
                    if not listener.is_alive():
                            listener = keyboard.Listener(on_press=on_press)
                            listener.start()
                    learn=False
                else:
                    learn_recognize(img_edges,learn,sign_name)
                    
                print("Finished in total of",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.\n")                
            else:
                print("Buffer mapping failed.")

    # Clean up
    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped.")

img_width=-1
img_height=-1
RESOLUTION_THRESHOLD=0.05
crop_width=OUTPUT_RESOLUTION//scale_factor
listener = keyboard.Listener(on_press=on_press)
listener.start()
learn=False
sign_name=""
    
def main():
    if args.camera:
        process_camera_gst()
    if args.learn:
        learn_input(args.learn)
    if args.recognize:
        recognize_input(args.recognize)
        
if __name__ == '__main__':
    main()
