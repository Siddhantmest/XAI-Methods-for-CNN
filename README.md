# Visual interpretation Methods for CNNs

Machine learning has made breakthroughs in the recent years for their applications in Computer vision tasks. However, deep learning models are not easily understood and are usually viewed as black boxes. Thus, there is a lack of trust among the users regarding the solutions obtained from the complex deep learning algorithms. Explainable AI attempts to explain the solutions obtained from these complicated algorithms. In this paper visual explainable AI (XAI) methods are used to investigate the local explanations of convolutional neural network (CNN) models for image classification. By visualizing the features most responsible for a model's decision, visual XAI can help gain trust of the users towards the solutions provided by these deep learning algorithms. The output of the solutions of visual XAI is an explanation heatmap which highlights the image regions that are important for the model’s decision making. This paper explores Grad-CAM (CAM based method) and the integrated gradients method (Backpropagation based method) on the 2 datasets, MNIST1D and HMT.

**XAI method implementation.ipynb** - Implementation of Grad CAM and Integrated Gradient on the 2 data sets, while comparing their performance

**xai_utils.py** - Contains code for other XAI such as SISE, RISE, etc.

**mnist1d_utils.py** - Code to transform the MNIST dataset to 1D 

**MNIST1D.pkl** - Pickel file of the MNIST data

**MNIST1D.h5** - CNN model trained on the MNIST1D data (built from scratch, code can be viewed in **MNIST1D.ipynb**, you can use any other CNN architecture of your choice)

**HMT.h5** - VGG-7 trained on the HMT data (code can be viewed in **HMT.ipynb**, you can use any other CNN architecture of your choice)

HMT dataset is available via this link https://drive.google.com/drive/folders/1R1kBdADpELCjR76D8icHRWpZvcqNZez4?usp=sharing

#### Note:

If you are using Google Colab to implement the code, I recommend you to change the runtime type to GPU as for some unknown reason Integrated gradient is giving inverted results on normal runtime

## Research Papers

**GradCAM** - Visual Explanations from Deep Networks via Gradient-based Localization, Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra

**Integrated Gradients** - Sundararajan, M., Taly, A. and Yan, Q., 2017, July. Axiomatic attribution for deep networks. In International Conference on Machine Learning (pp. 3319-3328). PMLR.
