# How does the program learn and recognize patterns?<br>
Consider two images the first image is to be learned and the second is to be recognized. <br>
1.<img src="sign-images/s.png"></img>&nbsp;2.<img src="sign-images/s-ripple.png"></img>
<br><br>
(a) The first step is to find the farthest points in the pattern. This divides the pattern into two and the points 1 and 2 are the first abstract points.<br>
3.<img src="sign-images/learn/s2.png"></img>&nbsp;4.<img src="sign-images/recognize/s-ripple2.png"></img>
<br><br>
(b) The next step is to find a point fartest from the line joining abstract points in the previous step. Thus, we obtain point 3.<br>
5.<img src="sign-images/learn/s3.png"></img>&nbsp;6.<img src="sign-images/recognize/s-ripple3.png"></img>
<br><br>
(c) The next is to find the signature, which is the concavity or convexity of each abstract points as you travel along the pattern in a specific order (here, clockwise direction).
Thus we get the following signatures.<br>
7.<img src="sign-images/learn/s3-sign-join.png"></img>&nbsp;8.<img src="sign-images/recognize/s-ripple3-sign-join.png"></img>
<br><br>
(c) The process is repeated till the farthest point is less than the threshold.<br>
9.<img src="sign-images/learn/s4.png"></img>&nbsp;10.<img src="sign-images/recognize/s-ripple4.png"></img>
<br><br>
11.<img src="sign-images/learn/s4-sign-join.png"></img>&nbsp;12.<img src="sign-images/recognize/s-ripple4-sign-join.png"></img>
<br><br>
13.<img src="sign-images/learn/s5.png"></img>&nbsp;14.<img src="sign-images/recognize/s-ripple5.png"></img>
<br><br>
15.<img src="sign-images/learn/s5-sign-join.png"></img>&nbsp;16.<img src="sign-images/recognize/s-ripple5-sign-join.png"></img>
<br><br>
17.<img src="sign-images/learn/s6.png"></img>&nbsp;18.<img src="sign-images/recognize/s-ripple6.png"></img>
<br><br>
19.<img src="sign-images/learn/s6-sign-join.png"></img>&nbsp;20.<img src="sign-images/recognize/s-ripple6-sign-join.png"></img>
<br><br>
