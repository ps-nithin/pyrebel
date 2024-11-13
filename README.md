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

# Layers of abstraction
<img src="animation.gif"></img>

At layer zero the most abstract details in the figure is compared. As we move into deeper layers finer details are compared which gives distinctiveness to each figure. Each layer is marked by the biggest drop in perpendicular distance.
# Read more about the logic implemented <a href="https://github.com/ps-nithin/pyrebel/blob/main/intro-r2.pdf">here</a>

# Let the data shine!
