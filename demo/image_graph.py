import matplotlib.pyplot as plot
import numpy as np
from PIL import Image
import argparse
import time

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input filename.")

def main():
    args=parser.parse_args()
    if args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
    else:
        print("No input filename provided.")
        return
    plot.ion()
    pos=list(range(0,img_array.shape[1]))
    fig=plot.figure(figsize=(10,5))
    subplot=fig.add_subplot(111)
    plot.ylim(0,255)
    graph,=subplot.plot(pos,img_array[0],color='lightgrey',label='Data points')
    plot.xlabel('Width')
    plot.ylabel('Intensity')
    plot.legend()
    plot.title('2D plot of Intensity for each row in the input image')
    row=1
    while 1:
        graph.set_xdata(pos)
        graph.set_ydata(img_array[row-1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("row=",row,"/",img_array.shape[0],end='\r')
        if row==img_array.shape[0]:
            break
        row+=1
        time.sleep(0.1)

if __name__=="__main__":
        main()
