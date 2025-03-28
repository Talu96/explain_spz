# Explaining Spermatozoa Vitality and Anomalies 


## How to run the code
To run the code to classify and eventually explain the prediction you have to give as input the filename of the image you want to analize.
``` 
python main.py image_filename
```

Then a new window will appear with the image and the predicted boxes: select the one you would like to explain and click on 'Explain' button.

The next window will ask you some characteristics of the spermatozoa that are not automatically retrivable. Select anything that seems appropriate.

If it is possible to detect the contour of the head, the window will show you some alternatives. Chose the one you think better approximate the shape or reject the contours.

The last window will show the image and the explanation, if it exists.