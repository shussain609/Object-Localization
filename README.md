# Flipkart Grid Challenge 2019-Object Localization

## The problem statement 
  Given an image, localize the predominant object in the image.
  
  ![](https://user-images.githubusercontent.com/28730618/65821648-f01c6100-e255-11e9-807a-12ff92a9b6de.png)
  
  Pre-trained models are prohibited but use of standard architectures is allowed. There are 24000 training images and 24045       testing images.
  
  The metric for scoring is mean intersection over union (IOU).
 
## Model Used
 Trained the images from scratch on RESNET34 architecture.
 
## Training Techniques
  Progressive size increments
  
     Used progressive size increment technique where we pass in downsized images first and progressively increase the size as    the learning begins to stagnate. This prevents the network from overfitting on the noise present in larger images and helps converge faster.Used images resized to 64, 128 and 224 in order. 
  
  Stochastic Gradient Descent with Restarts
  
     This is a learning rate scheduler.It gradually decreases the learning rate for a fixed number of epochs before resetting it again so that the network may easily pop out of local minima if stuck. 
  
## Loss Function
  Mean square error

## Accuracy
  Achieved an accuracy of 92.925% on the IOU metric.
