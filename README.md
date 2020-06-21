I built a model based on the top ranking architecture of a team that participated in the German Traffic Sign
Recognition competition with some modification to the original design.

During preprocessing, all the training samples were down-sampled or up-sampled to 48x48 pixels. Global normalization was computed to center each input image around its mean value as well as enhancing edges. Local contrast normalization with Gaussian kernels from the original architecture was discarded.
Training images were split into 90% training and 10% validation sets.

The CNN architecture has the structure (Spatial Transformer Networks) - (Convolutional block) - (Spatial Transformer Networks) - (Convolutional block) - (Spatial Transformer Networks) - (Convolutional block) with RMSprop configuration alpha = 0.99, epsilon = 1e-8, weight decay = 0, and learning rate = 1e-5. The spatial transformation networks consist of localization network, grid generator, and sampler.
The aim was to perform a geometric transformation on an input map so that CNNs can be spatially invariant to the input data in a computationally efficient manner.

The layers are as follows:
| Layer |       Type      |  # Maps & neurons  | Kernels |
|:-----:|:---------------:|:------------------:|:-------:|
|   0   |      Input      |  3 m. of 48*48 n.  |         |
|   1   |  Convolutional  | 200 m. of 46*46 n. |   7*7   |
|   2   |       ReLU      | 200 m. of 46*46 n. |         |
|   3   |   Max-Pooling   | 200 m. of 23*23 n. |   2*2   |
|   4   |  Convolutional  | 250 m. of 24*24 n. |   4*4   |
|   5   |       ReLU      | 250 m. of 24*24 n. |         |
|   6   |   Max-Pooling   | 250 m. of 12*12 n. |   2*2   |
|   7   |  Convolutional  | 350 m. of 13*13 n. |   4*4   |
|   8   |       ReLU      | 350 m. of 13*13 n. |         |
|   9   |   Max-Pooling   |  350 m. of 6*6 n.  |   2*2   |
|   10  | Fully connected |     400 neurons    |   1*1   |
|   11  |       ReLU      |     400 neurons    |         |
|   12  | Fully connected |     43 neurons     |   1*1   |
|   13  |     Soft-max    |     43 neurons     |         |

Convolutional layers’ stride is set to 1, zero padding set to 2, and max-pooling layers’ stride set to 2 with their zero padding to 0.
The implementation of Spatial Transformer Networks (STN) is based on pytorch’s tutorial:
https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

__Citation__: Arcos-García, Álvaro, et al. “Deep Neural Network for Traffic Sign Recognition Systems: An Analysis of Spatial Transformers and Stochastic Optimisation Methods.” _Neural Networks_, vol. 99, 31 Jan. 2018, pp. 158–165., doi:10.1016/j.neunet.2018.01.005.
