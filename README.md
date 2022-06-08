# TrashSegregationML

### Main goal
Create machine learning model for trash segregation based on photos. Trash can be classified to one of the five different classes:
- plastic/metal
- paper
- glass
- mixed
- biodegradable

When created model will be good enough, we plan to deploy it to the mobile application that is currently being developed [SegAI](https://github.com/martb00/SegAI). In order to combine application with the model, we are going to take advantage of Tensorflow Lite, instead of deploying it to the cloud services.

### Data collection
We have collected data from three different sources:
- Public datasets
We have searched on the internet in order to find some dataset with labeled photos of trash. We have found and used following datasets:
    - www.kaggle.com/datasets/roy2004/unsortedwaste
    - www.kaggle.com/datasets/nandinibagga/trashwaste-images-data?select=trash62.jpg
    - www.kaggle.com/datasets/asdasdasasdas/garbage-classification
    - www.kaggle.com/code/supermarkethobo/glass-vs-plastic-items-image-classification/notebook
    - github.com/garythung/trashnet
- Web crawler
We have created ourselves a crawler using Python and Scrapy framework. Its goal is to download photos from given websites. You can check it out in its own repository - [ScrapingImages](https://github.com/Kacper-Pietkun/ScrapingImages)
- Photos from users
We have created ourselves a mobile application, that you can install in order to send us some photos of trash. Application works only on Android. You can check it out in its own repository - [SegAI](https://github.com/martb00/SegAI). (Model that we are currently training will be eventually deployed in this application)

### Training process
We are taking advantage of transfer learning. So far, model has been fine-tuned on MobileNet architecture.

### Information about GPU usage for training
For this project I was using python 3.7.9 and tensorflow 2.9.1 (as it is listed in requirements.txt). In order to be able to carry out model training using GPU instead of CPU you would have to install:
- CUDA 11.2
- cuDNN 8.1

If you want to use different versions of python or tensorflow, just go to this page and make sure that you've downloaded proper versions of CUDA and cuDNN [tensorflow.org](https://www.tensorflow.org/install/source#gpu)
