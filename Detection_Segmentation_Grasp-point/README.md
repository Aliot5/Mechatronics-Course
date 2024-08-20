# **Welcome to the Mechatronics-Course.**

In this project we want to get to know YOLO and use deep learning  in the realm of object detection to propose.
Only Look Once (YOLO) proposes using an end-to-end neural network that makes predictions of bounding boxes and class probabilities all at once.

Evaluation of deep was shown in the diagram below

![YOLO](https://github.com/user-attachments/assets/de5c98e8-7263-40a1-9b98-c473d056c90a)

Perform detection since previous research showcased that adding convolution and connected layers to a pre-trained network improves performance. YOLO’s final fully connected layer predicts both class probabilities and bounding box coordinates.

![image](https://github.com/user-attachments/assets/87afd68c-ed62-42f1-9a43-3cc2ac26cfb4)



### **1. YOLO different versions**

The progress of YOLO has been as follows:

![Yolo1](https://github.com/user-attachments/assets/da40a1f7-2873-41c0-b8b2-46a64a7de17c)

**YOLOv1 vs YOLOv2:**

YOLOv2 improved this by introducing _**anchor boxes**_, which allowed the model to predict multiple bounding boxes per grid cell with different shapes and sizes, leading to better accuracy.

**YOLOv2 vs YOLOv3:**

YOLOv3 introduced the concept of _**multi-scale predictions**_. Unlike YOLOv2, which made predictions only at a single scale, YOLOv3 made predictions at three different scales, allowing it to detect objects at different sizes more effectively. 

**YOLOv3 vs YOLOv4:**

YOLOv4 incorporated several new techniques, including _**bag of freebies**_ (BoF) and _**bag of specials**_ (BoS), making YOLOv4 both faster and more accurate than YOLOv3.

**YOLOv4 vs YOLOv5:**

Making it more accessible for real-world applications.

**YOLOv5 vs YOLOv6:**

YOLOv6 introduced _**RepVGG-based backbone architecture**_ for enhanced performance. This change was aimed at improving the model's efficiency and accuracy by utilizing a more advanced backbone that was both faster and more robust compared to the one used in YOLOv5.

**YOLOv6 vs YOLOv7:**

YOLOv7 focused on _**optimizing the architecture**_ further by introducing techniques like _**extended efficient layer aggregation networks**_ (E-ELAN), which allowed for better learning of gradient flow and more efficient network architecture.

**YOLOv7 vs YOLOv8:**

YOLOv8 introduced a modular design and improved neural architecture search (NAS). The modular approach in YOLOv8 allows for easier customization, enabling users to modify the architecture based on specific needs, such as balancing speed and accuracy.

Finally You can choose the most suitable version of YOLO for your work according to your type of use.



### **2.mAP score:**

**Precision:**  
It measures how many of the detected objects are actually correct. The ratio of true positive detections to the total number of detections (true positives + false positives).

**Recall:**  
It measures how many of the actual objects were correctly detected. The ratio of true positive detections to the total number of actual objects (true positives + false negatives). It measures how many of the actual objects were correctly detected.

**AP:** 
AP is calculated by measuring the area under the Precision-Recall curve, that a Precision-Recall (PR) curve is plotted by varying the confidence threshold of the model’s detections.

**mAP:** 
Finally mAP is average of all of the APs that calculated in the last part.

![image](https://github.com/user-attachments/assets/d2ea05f9-26b2-44a1-a985-9c83f7eab360)

**benefits of mAP:**

**Here’s why mAP is important:**
* Combines Precision and Recall
* Evaluates Detection Quality
* Guides Model Improvement
* Benchmarking and Comparison


### _**3.project datasets**_


_**3.1.Dataset analysis**_

For this part we have a well dataset that has 19 classes and 1408 image from this classes.

**The division of classes is as follows:**

![image](https://github.com/user-attachments/assets/e3d44eb3-fa42-4353-b922-b18a1f31e3de)


_**3.2.Why augmentation?**_

* As you see the number of class photos is unbalanced and we can use data augmentation to fix this.
* It is also possible to make the model more resistant to noise.
* And increase the number of photos.
* Reducing overfitting.

_**3.3.augmentation in this project:**_

In this project we use two kind of augmentation which include _**rotation**_ and _**bounding box noise**_.
**Rotation: ** between -15 degree to +15 degree.
**Bounding box noise: ** up to 5% of pixels

Maybe a little camera angle change, and the objects may rotate a little, thus protecting the system against these events 
We made the system resistant to noise and the camera's accuracy may be a little too low or too high It resists mistakes.



### _**4.Object detection**_

_**4.1.Choosing_YOLO**_

In this part according to your tasks you can choose one of the YOLO's version and then get the related link from Roboflow.com and use it in code.

_**4.2.Use_YOLO**_
I choose YOLOv5 for this project and i would use it in code.

_**4.3.YOLO_Results**_
After run the code with 50 epochs we got this scale:
![image](https://github.com/user-attachments/assets/c4f0282f-5575-4213-8e2d-4bddd6998963)

That show us this results:

* **Class: **
This column lists the different object classes that the model was trained to detect, such as "banana," "biscuit," "cup-laying," etc. The first row labeled "all" aggregates the results across all classes.

* **Images: **
This column shows the number of images used in the evaluation (212 images in this case).

* **Instances: **
The number of individual objects (instances) of each class present in the evaluation dataset. For example, there are 191 instances of the "tea" class.

* **P (Precision): **
Precision is the ratio of correctly predicted positive observations to the total predicted positives. Higher precision means fewer false positives. The precision for each class is listed here.

* **R (Recall): **
Recall is the ratio of correctly predicted positive observations to all observations in the actual class. Higher recall means fewer false negatives.

* **mAP50 (Mean Average Precision at IoU 0.50): **
This is the mean average precision at a specific Intersection over Union (IoU) threshold of 0.50. It measures how well the model is at correctly identifying the objects at this overlap threshold.

* **mAP50-95 (Mean Average Precision at IoU 0.50 to 0.95): **
This is a more stringent metric that averages the precision across multiple IoU thresholds (from 0.50 to 0.95). It provides a comprehensive measure of the model's detection performance.

And in the all, section we can see a general analysis of the system.

![image](https://github.com/user-attachments/assets/41b3f913-8fc1-484d-970a-9848b22fcddd)

The reason for the difference in accuracy and other criteria for each of the classes is due to various factors, for example, lack of 
There is a balance in the number of data, or the appearance of some classes are similar to each other, or there is a lot of difference in appearance 
A class from the rest of the classes or the existence of diversity in the photos of a class or even the complexity of one 
Class has caused these differences in results for different classes.

For 50 epochs we have this results:

_**mAP:**_

![image](https://github.com/user-attachments/assets/8e4c1fba-e9d6-48d5-bd0b-30efbb2d50c9)

_**Precision:**_

![image](https://github.com/user-attachments/assets/ee8ade3f-d50c-49fb-a4d9-829cb445067f)

_**Recall:**_

![image](https://github.com/user-attachments/assets/81215699-b342-4fd6-9cad-f99d7e816d8c)

As you can see in the pictures above, at first the accuracy and other criteria show small values, but 
After more photos, they will improve.

_**4.4.Train_Time**_

Average time consumption for each epoch is 30s.

![image](https://github.com/user-attachments/assets/fd40a3d8-f4e1-4de6-a1db-16d77be169c0)

Total time consumption for all 50 epochs is about 27min.

![image](https://github.com/user-attachments/assets/1dc7bfac-14ed-4922-bfda-957844b2ea9a)

_**4.5.Test_input**_

Now we want to check model with test inputs.
Some examples of test data are as follows:

![image](https://github.com/user-attachments/assets/c24582a1-db13-42c3-8b2b-a0200b0e1880)

![image](https://github.com/user-attachments/assets/d515e2e5-31c8-41dc-a440-5dbf248385c0)

![image](https://github.com/user-attachments/assets/1c9ccc70-20e8-4db6-b137-6069d1a5abbe)

![image](https://github.com/user-attachments/assets/fffc0d01-fcc0-426b-aa74-03d09e98b3ea)

As you see some examples has wrong outputs.
In general, the main reason for misdiagnosis can be the presence of similar objects, or the presence of overlap in the images can also cause mistakes in the result.




### _**5.Object_Segmentation**_

_**5.1.Import_SAM**_

To use SAM, it must be imported first.
And we use FastSAM in this project.

_**5.2.Using_FastSAM**_

Now we want to pass the data we gave to YOLO to sam and observe the results.
The results of some data are as follows:

![image](https://github.com/user-attachments/assets/79b23988-0df1-4d8f-a906-d431cb2275d3)

![image](https://github.com/user-attachments/assets/cfd569bb-c17a-455d-bc58-807bbfd19ffc)

![image](https://github.com/user-attachments/assets/241e0382-5063-4cbb-b0ab-03d7803619cf)

### _**6.Grasp point generation**_

In this part, we want to grasp the object by the robot, for this we need to know from which points We use lifting and in this question we want to find these points.

At first we convert image from RGB to Gray and make them to binary like this:

![image](https://github.com/user-attachments/assets/eba409f0-91d8-4479-ad35-bdcd8524967f)

Now we can make boundary box around objects and also find angle, center and corners. Then we can find grasp points we know center and angle of object so we can draw a line like this:

![image](https://github.com/user-attachments/assets/fac234e4-0370-43f8-925a-f9a78fe2762c)

After all we have this results:

![image](https://github.com/user-attachments/assets/2f1eb74b-3727-4251-b82b-4b9bf365a7a3)


## conclusion
In this project we tried to work in the field of deep learning and get to know a little about YOLO types and then use them, and we also got to know the SAM function, and finally, we finished the project by doing a task that was defining the grasp points.
In general, it can be said that in this project, we got to know the topics of classification, training, segmentation, and grasping.
