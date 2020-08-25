# Deep Visualization(Pytorch)

Deep visualization is to show the visualization of an optimized image generated by learning an input image that maximizes the response to an objective filter. Hence, a pretrained model must be prepared in advance and the parameters of the model are frozen. Through the result, we can investigate where and how the model sees in an image. Normally, it has been known that shallow layers in the network tend to capture small patterns such as edges, and deeper layers capture larger textural structures. 

![Layer 28 Filter 15](https://user-images.githubusercontent.com/52735725/91221764-6c494e00-e71e-11ea-9975-f44b662c1a86.png)


# 1. Preparation for your model
* Add your model class in models.py 
* The name of your feature extraction module must be "features".
* A pretrained model must be prepared.(e.g. pretrained AlexNet on ImageNet dataset)
* No dataset is necessary.


# 2. Implementation
## layer 5
```
!python main.py --img_size 512 --layer_num 5 --filter_start 0 --filter_end 100 --step_size 3 --model_name vgg16
```
<img width="988" alt="5" src="https://user-images.githubusercontent.com/52735725/91223078-62284f00-e720-11ea-9428-e6d70492d356.png">

## layer 14

```
!python main.py --img_size 512 --layer_num 14 --filter_start 0 --filter_end 100 --model_name vgg16
```
<img width="988" alt="14" src="https://user-images.githubusercontent.com/52735725/91222318-4cfef080-e71f-11ea-8ad7-1619f9556a45.png">

## layer 21
```
!python main.py --img_size 512 --layer_num 21 --filter_start 0 --filter_end 100 --model_name vgg16
```
<img width="988" alt="21" src="https://user-images.githubusercontent.com/52735725/91222134-0610fb00-e71f-11ea-880a-f4a520759659.png">

## layer 28
```
!python main.py --img_size 512 --layer_num 28 --filter_start 0 --filter_end 100 --model_name vgg16
```
<img width="996" alt="28" src="https://user-images.githubusercontent.com/52735725/91221994-d3ff9900-e71e-11ea-9c4b-790f51c260e3.png">


# 3. References

Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson, Understanding Neural Networks Through Deep Visualization, arXiv preprint arXiv:1506.06579, 2015

Dumitru Erhan, Aaron Courville, Yoshua Bengio, and Pascal Vincent, Visualizing Higher-Layer Features of a Deep Network, Technical Report 1341, DIRO, Universite de Montreal, 2009

https://www.kaggle.com/carloalbertobarbano/convolutional-network-visualizations-deep-dream
