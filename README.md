# Seeing-the-World-Through-Words-Image-Captioning---Neural-Network-
Seeing the World Through Eyes (Image Captioning - Neural Network) is a Project in which we use neural network to build a image captioning software which helps visually impaired people to listen to the caption of the image taken by camera


Image captioning is a technology that allows visually impaired individuals to understand and appreciate visual content through descriptive text. This technology uses machine learning algorithms to generate accurate and detailed captions for images, which describe the content and context of an image in natural language.

This is particularly beneficial for individuals who are visually impaired, as they are often unable to access visual information such as photos, videos, and graphics. With the use of image captioning, visually impaired individuals can experience a more inclusive and enriched media experience, which can enhance their overall quality of life.

Objective:
#The objective of an assistant for visually impaired using image captioning is to provide a technology-based solution that helps individuals with visual impairments to access visual information in their daily lives. 
#The main goal is to use computer vision and natural language processing techniques to generate accurate and descriptive captions for images, which can be read aloud by an assistant or displayed on a braille device. This can help visually impaired individuals to navigate their environment, recognize faces, read signs, and access visual media such as books and websites.

**METHODOLOGY**

Data Pre-processing: 
The first step is to prepare the dataset for training the model. The Flickr8k dataset contains images and corresponding captions in text format. We need to pre-process the images by resizing them to a fixed size and normalizing the pixel values. We also need to pre-process the captions by converting them to lowercase, removing special characters and numbers, and tokenizing them into words.

Feature Extraction: 
The EfficientNetB0 CNN model is used to extract the features of the images. EfficientNetB0 is a state-of-the-art CNN architecture that achieves high accuracy with fewer parameters than traditional CNN models. We can use a pre-trained EfficientNetB0 model and remove the last few layers to extract the features of the images. The output of the EfficientNetB0 model is a feature vector for each image, which represents the important features of the image that are useful for generating captions.


Feature Extraction: 
The EfficientNetB0 CNN model is used to extract the features of the images. EfficientNetB0 is a state-of-the-art CNN architecture that achieves high accuracy with fewer parameters than traditional CNN models. We can use a pre-trained EfficientNetB0 model and remove the last few layers to extract the features of the images. The output of the EfficientNetB0 model is a feature vector for each image, which represents the important features of the image that are useful for generating captions.

Data Augmentation: 
Data augmentation is the process of creating new variations of the existing training data by applying various transformations to the images. This is done to increase the diversity of the training data and reduce the risk of overfitting.
Horizontal flipping: Flipping the images horizontally to create mirror images and increase the diversity of the training data.
By applying these transformations to the training data, the model can be trained on a larger and more diverse dataset, which can improve its accuracy and generalization ability.

Text Pre-processing: 
We need to pre-process the text data by creating a vocabulary of words that appear in the captions. We can also add start and end tokens to each caption to indicate the beginning and end of the sentence.

Model Architecture: 
The next step is to design the model architecture. we can use a CNN model to generate.

Building the CNN Encoder:
•	Use the pre-trained EfficientNetB0 model as the CNN encoder.
•	Remove the top layer of the model to get the encoded image.
•	Freeze the weights of the model to prevent them from being updated during training.

Fine-tuning the CNN Encoder:
•	After training the model, you can fine-tune the CNN encoder to further improve the model's performance.
•	Unfreeze the last few layers of the EfficientNetB0 model and train the entire model using a smaller learning rate.
•	Fine-tuning the CNN encoder can help the model better capture the relevant features of the image.

Building the Caption Decoder:
•	Initialize the decoder with a fully connected layer that takes the encoded image as input and outputs a feature vector.
•	Add multiple fully connected layers with ReLU activation to learn the features of the image and the caption.
•	Add a final fully connected layer with SoftMax activation that generates a probability distribution over the vocabulary for each word in the caption.

Transfer Learning:
Transfer learning can be used to leverage the pre-trained weights of the EfficientNetB0 By using transfer learning, you can save time and resources required for training the model from scratch.

Model Compiling:
•	Model compiling refers to the process of specifying the optimizer, loss function, and evaluation metrics for the model. Once the model architecture is defined, the next step is to compile the model before training.
•	Optimizer specifies the algorithm used to update the weights of model during training. Here we are using Adam Optimizer which is used in deep learning that combines the advantages of Adagrad and RMSprop optimization algorithms.
•	Loss function specifies the objective that the model is trying to minimize during training. Cross-entropy loss function, also known as log loss, is commonly used in classification tasks. It measures the difference between the predicted probability distribution and the actual probability distribution.

Image Capturing:
•	Using a webcam to capture images for processing on the runtime. Resize the output to fit the video element. Wait for Capture to be clicked. Image which was just taken used for image captioning.

<img width="413" alt="image" src="https://github.com/karthikpagnis/Seeing-the-World-Through-Words-Image-Captioning---Neural-Network-/assets/91360050/2f663efc-9b55-4561-849c-7bfbebf5b9e0">
      Figure 1: Encoding and Decoding for Image Captioning
      
Training the Model:
•	Compile the model using the Adam optimizer and categorical cross-entropy loss.
•	Train the model on the pre-processed image and caption data using the fit generator function.
•	Evaluate the model using metrics such as BLEU score and perplexity.
•	Fine-tune the model by unfreezing the top layers of the CNN encoder and continuing training.


Generating Captions:
•	Use the trained model to generate captions for new images.
•	Pre-process the new image using the EfficientNetB0 pre-processor.
•	EfficientNetB0 is a pre-trained CNN that has been trained on a large dataset of images to perform image classification tasks. It is commonly used as a pre-processor for image captioning models because it can efficiently transform images into a feature representation that is suitable for further processing by the image captioning model.
<img width="319" alt="image" src="https://github.com/karthikpagnis/Seeing-the-World-Through-Words-Image-Captioning---Neural-Network-/assets/91360050/f214c2bf-0b79-4762-8105-69eaf290b3cf">

GTTS (Google Text-to-Speech):
GTTS (Google Text-to-Speech) is a Python library and command-line interface that utilizes Google's Text-to-Speech API to convert text into speech. It provides a convenient way to generate speech from text in multiple languages. gTTS offers a simple and intuitive interface for creating audio files from textual content.

<img width="249" alt="image" src="https://github.com/karthikpagnis/Seeing-the-World-Through-Words-Image-Captioning---Neural-Network-/assets/91360050/1975121b-a694-4966-a157-c616f0c075ed">


