# pyrebel
# On Jetson Nano
Usage:<br>
```python3 pynvrebel.py --input <filename.png>```<br><br>
Optional arguments<br>
```--depth <value>``` Selects the depth of abstraction. Defaults to 10.<br>
```--blob <value>``` Selects the blob. Defaults to 0.<br>
```--layer <value>``` Selects the layer of abstraction. Defaults to 0.<br>
```--output <output filename>```Changes the output filename. Defaults to "output.png".<br><br>
For example,<br>
```python3 pynvrebel.py --input deep.png --blob 1 --depth 20 --layer 0```<br>

The expected input files are grayscale images of figures like <a href="https://github.com/ps-nithin/pyrebel/blob/main/letters.png">letters.png</a><br>

Using <a href="https://github.com/ps-nithin/pyrebel/blob/main/aaa.png">aaa.png</a> or <a href="https://github.com/ps-nithin/pyrebel/blob/main/sss.png">sss.png</a> demonstrates how the program responds to similar figures of different scales.<br>

Running the above program will 
1. Open an image showing the abstract pixels in the input image file as white pixels and saves the output to disk.
2. Display the layers of abstract pixels in the blob for the given depth.

# Edge detection demo
This is a demo of edge detection achieved using data abstraction.<br>
```python3 pyvision_edge_detection.py --preprocess <filename>```<br>

For eq.
```python3 pyvision_edge_detection.py --preprocess images/wildlife.jpg```<br>

Running the above program will show the edges in the image.<br>
<img src="images/small_wildlife.jpg"></img><br>Below is the output image<br><img src="images/output_wildlife.png"></img>
# Layers of abstraction
<img src="animation.gif"></img>

At layer zero the most abstract details in the figure is compared. As we move into deeper layers finer details are compared which gives distinctiveness to each figure.
# Read more about the methods <a href="https://github.com/ps-nithin/pyrebel/blob/main/intro-r2.pdf">here</a>

# Let the data shine!
