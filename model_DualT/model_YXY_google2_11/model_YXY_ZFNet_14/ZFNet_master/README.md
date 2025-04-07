<h1>ZFNet Model Implementation</h1>
<p><a href="https://arxiv.org/pdf/1311.2901.pdf">Visualizing and Understanding Convolutional Networks</a></p>
<h3>Dataset</h3>
<p>In the paper, the authors have trained the model on Imagenet dataset, but due to memory and processing constraints the model here has been trained on the CIFAR100 dataset</p>
<h3>Model</h3>
<p>The proposed model is a variation of the <a href="http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf">Alexnet</a> module. Since the only change is in the first convolutional layer of the model, hence here all pretrained weights, of the alexnet module have been downloaded and used, and the training has been done only on the first CNN layer and the final Dense Linear layer.</p>
<h3>Convnet Visualization</h3>
<p>For feature visualization, the top 9 activations for each given layer is projected seperately back to the input space, which reveals the various different structures that excite a given feature map. This is done by setting all other activations in the layer to zero and the successively <ol><li>unpool</li><li>rectify</li><li>filter</li></ol>to reconstruct the activations all the way back to the input pixel space.</p>
<p>The <a href="https://github.com/rahatsantosh/ZFNet/blob/master/pytorch/Convnet_visualization/deconvnet.py">deconvnet</a> function does this only for the 5<sup>th</sup> convolutional layer, and can be modified to do the same for subsequent convolutions by mapping the corresponding convolutions to transpose convolution functions.</p>
<hr>
