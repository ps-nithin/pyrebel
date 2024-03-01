# pyrebel
# On Jetson Nano
Usage:<br>
```python3 pynvrebel.py --input <filename.png>```<br><br>
Optional arguments<br>
```--threshold <value>``` Change the threshold value. Defaults to 32.<br>
```--blob <value>``` Selects the blob. Defaults to 0.<br>
```--layer <value>``` Selects the layer of abstraction. Defaults to 0.<br>
```--output <output filename>```Change output filename. Defaults to "output.png".<br><br>
For example,<br>
```python3 pynvrebel.py --input letters.png --threshold 32```<br>

The expected input files are grayscale images of figures like <a href="https://github.com/ps-nithin/pyrebel/blob/main/letters.png">letters.png</a><br>

Using <a href="https://github.com/ps-nithin/pyrebel/blob/main/aaa.png">aaa.png</a> or <a href="https://github.com/ps-nithin/pyrebel/blob/main/sss.png">sss.png</a> demonstrates how the program responds to similar figures of different scales.<br>

Running the above program will open an image showing the abstract pixels in the input image file as white pixels and saves the output to disk.
# Layers of abstraction
At layer zero the most abstract details in the figure is compared. As we move into deeper layers finer details are compared which gives distinctiveness to each figure.
# Read more about the logic implemented <a href="https://github.com/ps-nithin/pyrebel/blob/main/abstract.pdf">here</a>

# Let the data shine!
