# Unsupervised Learning of Object Keypoints through Conditional Image Generation and Equivariance Constraint
Inspired by Jakab and Gupta et al. (2018) [Unsupervised Learning of Object Landmarks through Conditional Image Generation](https://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/unsupervised_landmarks.pdf) and Kulkarni and Gupta et al. (2019) [Unsupervised Learning of Object Keypoints for Perception and Control](https://arxiv.org/pdf/1906.11883.pdf), the objective of this dissertation is to produce a unsupervised learning model detecting keypoints of images or frames in videos. The approach of learning is similar from the two papers, through creating two models and pairs of source and target image. The first model trains to detect geometry of object from the source image while the second model trains to reconstruct source image via combination of source image and geometry of object. Aside from reimplementation of Convolutional Neural Network (CNN) models from the Jakab and Gupta's papers, this dissertation aims to explore the capability of model on keypoints detection with more complex background. Hence, the [TigDog Dataset](http://calvin-vision.net/datasets/tigdog/) is used during training and testing process.

# Abstract
Unsupervised learning of keypoints detection through image reconstruction has received a lot of
interest due to its demand in industry and application generality across datasets. A state-ofthe-art learning method from [Unsupervised Learning of Object Landmarks through Conditional
Image Generation](https://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/unsupervised_landmarks.pdf) a requires a pair of images, namely source image and target image, which differs via object deformation or viewpoint change. The model learns to detect the geometry of a
target image, combining with the appearance of a source image detected using a similar model,
to generate an image as close as the target image. The model has proven to work well in both
object deformation and viewpoint change by only adopting a perceptual loss formulation, but our
research shows that it does not guarantee to work well in raw video frames. Our empirical research shows that the model is not able to learn keypoints that precisely represent each specific
part of object when the number of keypoints and size of objects varies. Besides, we show that
the learning of keypoints is much ineffective when the background of object changes throughout
the frames, which is very common in video. We propose a new method to learn keypoint detection by adding equivariance constraint in training and adopting different types of loss instead of
only perceptual loss and adding new dimension to regressor to identify the visibility of keypoints.
We also show that non-static background will greatly affect the efficiency of model in keypoints
detection and image reconstruction, and the training could be improved with static background
training images. We evaluate our model on TigDog Dataset, which contains tiger frames sourced
from documentaries videos. We show that our method outperforms the state-of-the-art unsupervised keypoints detection via image reconstruction with video frame dataset, while preserving the
simplicity of implementations and generality across datasets.

# References
1. Tomas Jakab, Ankush Gupta, Hakan Bilen, Andrea Vedaldi (2018). Unsupervised Learning of Object Landmarks through Conditional Image Generation[online]. Available from:https://www.robots.ox.ac.uk/~vgg/research/unsupervised_landmarks/unsupervised_landmarks.pdf
2. Tejas Kulkarni, Ankush Gupta, Catalin Ionescu, Sebastian Borgeaud, Malcolm Reynolds, Andrew Zisserman, Volodymyr Mnih (2020). Unsupervised Learning of Object Keypoints for Perception and Control[online]. Available from:arXiv:1906.11883
3. Del Pero, L. and Ricco, S. and Sukthankar, R. and Ferrari, V (2015). Articulated motion discovery using pairs of trajectories[inproceedings]. Available from : Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
4. Del Pero, L. and Ricco, S. and Sukthankar, R. and Ferrari, V (2016). Behavior Discovery and Alignment of Articulated Object Classes from Unstructured Video[journal]. Available from : International Journal of Computer Vision (IJCV).
