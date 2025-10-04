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
(c) The process is repeated till the farthest point is less than the threshold.<br>
7.<img src="sign-images/learn/s4.png"></img>&nbsp;8.<img src="sign-images/recognize/s-ripple4.png"></img>
<br><br>
