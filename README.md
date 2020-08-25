# Deep Visualization(Pytorch)

Deep visualization is to show the visualization of an optimized image generated by learning an input image that maximizes the response to an objective filter. Hence, a pretrained model must be prepared in advance and the parameters of the model are frozen. Through the result, we can investigate where and how the model sees in an image. Normally, it has been known that shallow layers in the network tend to capture small patterns such as edges, and deeper layers capture larger textural structures. 

![Layer 10 Filter 13](https://user-images.githubusercontent.com/52735725/90338172-f0495a80-dfe7-11ea-9bb2-10cda486c177.png)

# 1. Preparation for your model
* Add your model class in models.py 
* The name of your feature extraction module must be "features".
* A pretrained model must be prepared.(e.g. pretrained AlexNet on ImageNet dataset)
* No dataset is necessary.


# 2. Implementation
## layer 1
```
python main.py --layer_num 1 --img_size 512 --filter_start 0 --filter_end 40 --model_name alexnet
```
<img width="936" alt="layer1" src="https://user-images.githubusercontent.com/52735725/90338339-48349100-dfe9-11ea-9c29-e3ac321c5d43.png">

## layer 3
```
python main.py --layer_num 3
```
<img width="936" alt="layer3" src="https://user-images.githubusercontent.com/52735725/90338343-51256280-dfe9-11ea-927a-bf010ced0eca.png">


## layer 10
```
python main.py --layer_num 10 --model_name modified_alexnet
```
<img width="936" alt="layer10" src="https://user-images.githubusercontent.com/52735725/90338355-65695f80-dfe9-11ea-90c4-27465164d002.png">


# 3. References

Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson, Understanding Neural Networks Through Deep Visualization, arXiv preprint arXiv:1506.06579, 2015

Dumitru Erhan, Aaron Courville, Yoshua Bengio, and Pascal Vincent, Visualizing Higher-Layer Features of a Deep Network, Technical Report 1341, DIRO, Universite de Montreal, 2009

https://www.kaggle.com/carloalbertobarbano/convolutional-network-visualizations-deep-dream
