# computer-vision-2024
Learning Pathway of Computer Vision - Started April 2024 - Featured Udacity

*Github Computer Vision Exercise Repository:* [UDACITY COMPUTER VISION](https://github.com/udacity/cd0360-Introduction-to-Computer-Vision)

## **Lesson 1: Computer Vision Pipeline**

### 1. Pattern Recognition

**Definition:** Pattern recognition in computer vision is a fundamental concept that involves the automatic detection, interpretation, and classification of patterns or objects within images or videos. It's a process through which a computer system learns to recognize patterns or features in visual data, similar to how humans perceive and interpret visual information.

Here's how it generally works:

- Feature Extraction: In this step, relevant features or characteristics are extracted from the input images or video frames. These features can be simple, such as edges, corners, or textures, or they can be more complex, like shapes or object parts.

- Pattern Representation: Once the features are extracted, they are represented in a suitable format that can be understood by a computer algorithm. This representation can be numerical descriptors, vectors, or other mathematical representations.

- Pattern Classification: The represented patterns are then classified into predefined categories or classes using machine learning algorithms. These algorithms are trained on a dataset containing examples of patterns along with their corresponding labels.

- Training and Learning: The classification algorithms learn from the labeled data during a training phase, where they adjust their internal parameters to minimize errors in classifying patterns.

- Pattern Recognition: After the training phase, the system is able to recognize patterns or objects in new, unseen images or video frames. It applies the learned classification model to classify the features extracted from the input data into one of the predefined classes.

### 2. Emotional Intelligence

**Cognitive intelligence** is the ability to reason and understand the world based on observations and facts. It's often what is measured on academic tests and what's measured to calculate a person's IQ.

**Emotional intelligence** is the ability to understand and influence human emotion. For example, observing that someone looks sad based on their facial expression, body language, and what you know about them - then acting to comfort them or asking them if they want to talk, etc. For humans, this kind of intelligence allows us to form meaningful connections and build a trustworthy network of friends and family. It's also often thought of as only a human quality and is not yet a part of traditional AI systems.

### 3. Computer Vision Pipeline

A computer vision pipeline is a series of steps that most computer vision applications will go through. Many vision applications start off by acquiring images and data, then processing that data, performing some analysis and recognition steps, then finally performing an action.

3.1. Image Acquisition: The process begins with capturing images or videos using cameras or other imaging devices. This stage involves considerations such as resolution, lighting conditions, and camera settings.

3.2. Preprocessing: The acquired images or frames may undergo preprocessing to enhance their quality and make subsequent processing more effective. Preprocessing steps may include operations like resizing, noise reduction, color correction, and image normalization.

3.3. Feature Extraction: In this stage, relevant features or characteristics are extracted from the preprocessed images. Features can include edges, corners, textures, shapes, or other visual cues that are essential for the task at hand. Feature extraction methods can range from simple techniques like edge detection to more complex methods like feature point detection or deep learning-based feature extraction.

3.4. Feature Representation: Extracted features are often represented in a suitable format for further processing. This may involve converting features into numerical vectors or other representations that can be easily manipulated and analyzed by computer algorithms.

3.5. Learning (Optional): In some cases, the extracted features are used to train machine learning models or deep neural networks. During the training phase, the models learn to recognize patterns or make predictions based on the input features and corresponding labels.

3.6. Inference: Once trained (if applicable), the models are used to perform inference on new, unseen data. This involves applying the learned models to classify objects, detect anomalies, recognize patterns, or perform other tasks based on the extracted features.

3.7. Postprocessing: The results obtained from the inference stage may undergo postprocessing to refine or interpret the output. Postprocessing steps may include filtering, smoothing, thresholding, or other operations to improve the accuracy or usability of the results.

3.8. Visualization (Optional): Finally, the processed data or results may be visualized to facilitate human understanding or further analysis. Visualization techniques can include displaying images with annotated objects or overlaying augmented reality information onto the scene.

**Standardizing Data**

> ***Pre-processing images*** is all about standardizing input images so that you can move further along the pipeline and analyze images in the same way. In machine learning tasks, the pre-processing step is often one of the most important.

For example, imagine that you've created a simple algorithm to distinguish between stop signs and other traffic lights.

### 4. Training a model

> ***Training a Neural Network***

To train a computer vision neural network, we typically provide sets of labeled images, which we can compare to the predicted output label or recognition measurements. The neural network then monitors any errors it makes (by comparing the correct label to the output label) and corrects for them by modifying how it finds and prioritizes patterns and differences among the image data. Eventually, given enough labeled data, the model should be able to characterize any new, unlabeled, image data it sees!

A training flow is pictured below. This is a convolutional neural network that learns to recognize and distinguish between images of a smile and a smirk.

**Gradient descent** is a fundamental optimization algorithm used in various fields of machine learning, including computer vision. In the context of computer vision, gradient descent is often employed to train machine learning models, such as neural networks, for tasks like object detection, image classification, or image segmentation.

**Convolutional neural networks** are a specific type of neural network that are commonly used in computer vision applications.

### 5. Image Formation

Image formation refers to the process by which an image is created or captured using an imaging system, such as a camera or an optical instrument. It involves the transformation of light rays reflecting off objects in a scene into a two-dimensional representation on a photosensitive surface (such as a camera sensor or film).

Here's a simplified overview of the image formation process:

5.1. Light Source: The image formation process begins with a source of illumination, such as natural light from the sun or artificial light from lamps or flash units. Light rays emanate from the source and illuminate objects in the scene.

5.2. Reflection and Refraction: When light strikes an object in the scene, it interacts with the object's surface in various ways. Some of the light is reflected off the surface, while some may be absorbed or transmitted through the object. The properties of the object's surface, such as its color, texture, and reflectance, determine how it interacts with light.

5.3. Lens and Optics: In imaging systems like cameras, light rays from the scene pass through a lens or a series of optical elements. The lens focuses the incoming light rays to form an image at a specific plane, known as the focal plane. The optical properties of the lens, such as its focal length and aperture size, determine the characteristics of the resulting image, such as its sharpness, depth of field, and perspective.

5.4. Projection: The focused image formed by the lens is a two-dimensional projection of the three-dimensional scene onto the focal plane. This projection preserves the spatial relationships between objects in the scene but may introduce distortions or aberrations due to the characteristics of the imaging system.

5.5 Capture Medium: In digital imaging systems, such as digital cameras, the focused image is captured by a photosensitive sensor, typically a charge-coupled device (CCD) or a complementary metal-oxide-semiconductor (CMOS) sensor. Each pixel on the sensor records the intensity of light falling on it, effectively digitizing the image.

5.6. Digital Image Representation: The captured image is stored and represented in digital format, typically as an array of numerical values representing the intensity of light at each pixel. This digital representation allows for further processing, analysis, and manipulation of the image using computational techniques.

### 6. Image as Grids and Pixels

> ***Images as Numerical Data***

Every pixel in an image is just a numerical value and, we can also change these pixel values. We can multiply every single one by a scalar to change how bright the image is, we can shift each pixel value to the right, and many more operations!

Treating images as grids of numbers is the basis for many image processing techniques.

Most color and shape transformations are done just by mathematically operating on an image and changing it pixel-by-pixel.

### Python Practice Notebook 1: Images as Grids of Pixels

Link: [Learning: Computer Vision 1]()

In this practice notebook, we will gain these new knowledge:

- Learn about the name of packages and library that we will be using while working with Computer Vision (**cv2, numpy, matplotlib**)
- Read an image from a file: *image = mpimg.imread('images/waymo_car.jpg')* (**mpimg** is used to call **matplotlib.image** package)
- Learn to change colored image to greyscale: *gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)*
- Understand the pixel values and grids (if an image is 500x760 in size, it means the height is 760 pixels and the width is 500 pixels)
- Understand the greyscale value (0 to 255, 0 for WHITE, 255 for BLACK)

### 7. Color Images

Color images are interpreted as 3D cubes of values with width, height, and depth!

*Example: image_dog.shape = [500, 800, 3] (height 500, width 800 and depth 3 for R G B)*

The depth is the number of colors. Most color images can be represented by combinations of only 3 colors: red, green, and blue values; these are known as RGB images. And for RGB images, the depth is 3!

It’s helpful to think of the depth as three stacked, 2D color layers. One layer is Red, one Green, and one Blue. Together they create a complete color image.

### 8. Color Thresholds

**Color thresholds** refer to predefined ranges or criteria used to segment or isolate regions of interest in an image based on their color. In computer vision and image processing, color thresholding is a common technique used for tasks such as object detection, image segmentation, and feature extraction, particularly when dealing with color images.

### Python Practice Notebook 2: Coding a Blue Screen

Link: [Learning: Computer Vision 2]()

> ***OpenCV***

**OpenCV** is a popular computer vision library that has many built-in tools for image analysis and understanding!

Note: In the example above and in later examples, I'm using my own Jupyter notebook and sets of images stored on my personal computer. You're encouraged to set up a similar environment and use images of your own to practice! You'll also be given some code quizzes (coming up next), with images provided, to practice these techniques.

> ***Why BGR instead of RGB?***

**OpenCV** reads in images in BGR format (instead of RGB) because when OpenCV was first being developed, BGR color format was popular among camera manufacturers and image software providers. The red channel was considered one of the least important color channels, so was listed last, and many bitmaps use BGR format for image storage. However, now the standard has changed and most image software and cameras use RGB format, which is why, in these examples, it's good practice to initially convert BGR images to RGB before analyzing or manipulating them.

> ***Masking***

In the context of computer vision and image processing, a **mask** refers to a binary image or a matrix where certain pixels are marked or highlighted to define regions of interest or to apply specific operations selectively. Masking, therefore, is the process of using such a mask to either isolate or highlight certain parts of an image for further processing or analysis.

In this practice notebook, we will gain these new knowledge:

- Using *shape* command to display the height, width and depth of an image.
- OpenCV prefers using BGR instead of RGB because *Red* is considered to be the least important color out of the three, and because BGR was popular among camera manufacturers.
- Understand Masking.
- Know how to crop background
- Know that it is required for the shape of background and the item image to be equal in order to combine (add background to item image)

### Python Practice Notebook 3: Coding a Green Screen

Link: [Learning: Computer Vision 3]()

In this practice notebook, we will gain these new knowledge:

- Know how to define the color threshold boundaries in order to select the allowed color range for the main image, the remaining will be removed to create a mask.
- Understand cropping (background and item image mas have the same size)
- Know that it is neccessary to make a copy of the image we are working with constantly in order to not losing the image processes

### 9. Color Spaces and Transforms

**HSV** stands for Hue, Saturation, and Value, and it's a color space commonly used in computer vision and image processing. Unlike the RGB (Red, Green, Blue) color space, which represents colors as combinations of red, green, and blue components, the HSV color space represents colors in terms of their perceived attributes: hue, saturation, and brightness.

Here's a brief explanation of each component in the HSV color space:

**Hue (H)**: Hue represents the dominant wavelength of light that gives a color its pure tone. It's often described as the "color" of the color. In the HSV color space, hue is typically represented as an angle around a color wheel, with red at 0°, green at 120°, and blue at 240°, wrapping back to red at 360°. This arrangement allows for intuitive understanding of color relationships, as colors close to each other on the color wheel are similar in hue.

**Saturation (S)**: Saturation refers to the intensity or purity of a color. A fully saturated color is vivid and rich, while a desaturated color appears more muted or grayscale. In the HSV color space, saturation is represented as a percentage, with 0% indicating a completely grayscale (unsaturated) color and 100% indicating a fully saturated color.

**Value (V)**: Value, also known as brightness or lightness, represents the brightness of a color. It determines how much light is emitted or reflected by the color. In the HSV color space, value is typically represented as a percentage, with 0% indicating black (minimum brightness) and 100% indicating white (maximum brightness).

The HSV color space is particularly useful in computer vision tasks that involve color-based segmentation, object detection, or image analysis, as it separates the color information (hue and saturation) from the brightness information (value). This separation makes it easier to work with color information independently of brightness, allowing for more robust color-based algorithms and techniques.

### Python Practice Notebook 4: Color Conversion

Link: [Learning: Computer Vision 4]()

In this practice notebook, we will gain these new knowledge:

- In order to separate the color and its color brightness (Case: select only the pink balloons in the picture showing balloons in all sort of colors), we need to plot the channels for RGB and HSV for the masking.
- Understand HSV defintion, and what it differs from RGB
- Understand plotting channels (RGB, HSV)

### Python Practice Notebook 5: Day and Night Classification

Link: [Learning: Computer Vision 5]()

Resources: 1700 day/night images at multiple location and multiple timing (Train - Test: 40:60)

***Currently work in progress...***

> Task: Using the day/night image from the library, try separating the day and night image based on RGB and HSV.

### 10. Labeled Data

Why do we need labels?

You can tell if an image is night or day, but a computer cannot unless we tell it explicitly with a label! This becomes especially important when we are testing the accuracy of a classification model.

A classifier takes in an image as input and should output a predicted_label that tells us the predicted class of that image. Now, when we load in data, like you’ve seen, we load in what are called the true_labels which are the correct labels for the image.

To check the accuracy of a classification model, we compare the predicted and true labels. If the true and predicted labels match, then we’ve classified the image correctly! Sometimes the labels do not match, which means we’ve misclassified an image.

> ***Accuracy***

After looking at many images, the accuracy of a classifier is defined as the number of correctly classified images (for which the predicted_label matches the true label) divided by the total number of images. So, say we tried to classify 100 images total, and we correctly classified 81 of them. We’d have **0.81** or **81%** accuracy!

We can tell a computer to check the accuracy of a classifier only when we have these predicted and true labels to compare.

> ***Numberical Labels***

It’s good practice to use numerical labels instead of strings or categorical labels. They're easier to track and compare. So, for our day and night, binary class example, instead of "day" and "night" labels we’ll use the numerical labels: 0 for night and 1 for day.

Example (Binary): 

**DAY = 1**  ||   **NIGHT = 0**  

### 11. Features

**Features** refer to distinctive and identifiable characteristics or patterns present in an image. These features are often used as a basis for various tasks such as object detection, recognition, matching, and tracking. Features are essentially local regions or points within an image that contain meaningful information, which can be used to describe and distinguish objects or patterns.

> ***Distinguishing and Measurable Traits***

When you approach a classification challenge, you may ask yourself: how can I tell these images apart? What traits do these images have that differentiate them, and how can I write code to represent their differences? Adding on to that, how can I ignore irrelevant or overly similar parts of these images?

You may have thought about a number of distinguishing features: day images are much brighter, generally than night images. Night images also have these really bright small spots, so the brightness over the whole image varies a lot more than the day images. There is a lot more of a gray/blue color palette in the day images.

There are lots of measurable traits that distinguish these images, and these measurable traits are referred to as features.

A feature is a measurable component of an image or object that is, ideally, unique and recognizable under varying conditions - like under varying light or camera angle. And we’ll learn more about features soon.

Here are some real-world examples of features in computer vision applications:

11.1. **Facial Features for Face Recognition**: In face recognition systems, features such as eyes, nose, mouth, and facial contours are detected and used to represent individual faces. These features are often described by their spatial relationships, shapes, and appearance characteristics, enabling accurate face matching and identification.

11.2. **Keypoints in Image Stitching**: In image stitching applications, keypoints are detected in overlapping images to find corresponding points between them. These keypoints are then used to align and blend the images seamlessly, creating a panoramic view. Examples of keypoints include corners, edges, and distinctive structures like building corners or tree branches.

11.3. **Traffic Sign Recognition**: In autonomous vehicles and intelligent transportation systems, features such as shapes, colors, and symbols are extracted from traffic sign images to recognize and interpret traffic signs. These features help in identifying the type of sign (e.g., stop sign, speed limit sign) and understanding its meaning for safe navigation.

11.4. **Object Detection in Robotics**: In robotics applications, features such as edges, corners, and textures are extracted from sensor data (e.g., camera images, LiDAR scans) to detect and localize objects in the environment. These features are used to identify obstacles, landmarks, and objects of interest for navigation and manipulation tasks.

11.5. **Medical Image Analysis**: In medical imaging, features such as textures, shapes, and intensity patterns are extracted from medical images (e.g., MRI, CT scans) to aid in diagnosis and treatment planning. These features help in detecting abnormalities, segmenting organs or tissues, and quantifying disease progression.

11.6. **Quality Inspection in Manufacturing**: In industrial automation, features such as surface defects, cracks, and geometric attributes are extracted from product images to perform quality inspection and defect detection. These features are used to identify flaws or irregularities in manufactured parts, ensuring product quality and compliance with specifications.

11.7. **Gesture Recognition in Human-Computer Interaction**: In gesture recognition systems, features such as hand shape, motion trajectories, and finger positions are extracted from video streams or depth sensor data to recognize and interpret human gestures. These features enable natural and intuitive interaction with computers, gaming consoles, and other devices.

> ***Standardizing and Pre-processing***

Standardizing and preprocessing are essential steps in preparing data for machine learning and data analysis tasks, including in the field of computer vision. These steps involve transforming and normalizing the data to improve its quality, consistency, and suitability for the intended analysis or modeling.

> ***Numerical vs. Categorical***

Let's learn a little more about labels. After visualizing the image data, you'll have seen that each image has an attached label: "day" or "night," and these are known as categorical values.

Categorical values are typically text values that represent various traits about an image. A couple of examples are:

- An "animal" variable with the values: "cat," "tiger," "hippopotamus," and "dog."
- A "color" variable with the values: "red," "green," and "blue."
  
Each value represents a different category, and most collected data is labeled in this way!

These labels are descriptive for us, but may be inefficient for a classification task. Many machine learning algorithms do not use categorical data; they require that all output be numerical. Numbers are easily compared and stored in memory, and for this reason, we often have to convert categorical values into numerical labels. There are two main approaches that you'll come across:

- Integer encoding
- One hot-encoding

**Integer encoding** means to assign each category value an integer value. So, day = 1 and night = 0. This is a nice way to separate binary data, and it's what we'll do for our day and night images.

**One-hot encoding** is often used when there are more than 2 values to separate. A one-hot label is a 1D list that's the length of the number of classes. Say we are looking at the animal variable with the values: "cat," "tiger," "hippopotamus," and "dog." There are 4 classes in this category and so our one-hot labels will be a list of length four. The list will be all 0's and one 1; the 1 indicates which class a certain image is.

For example, since we have four classes (cat, tiger, hippopotamus, and dog), we can make a list in that order: [cat value, tiger value, hippopotamus value, dog value]. In general, order does not matter.

If we have an image and it's one-hot label is [0, 1, 0, 0], what does that indicate?

In order of [cat value, tiger value, hippopotamus value, dog value], that label indicates that it's an image of a tiger! Let's do one more example, what about the label [0, 0, 0, 1]?

Here's how to interpret the one-hot label:

- The first value (0) represents the category "cat."
- The second value (0) represents the category "tiger."
- The third value (0) represents the category "hippopotamus."
- The fourth value (1) represents the category "dog."

Since only the fourth value is set to 1 ("on"), while the others are 0 ("off"), this indicates that the data point belongs to the category "dog" and not to any of the other categories. This encoding is commonly used in machine learning tasks, such as classification, where the model needs to predict the presence or absence of specific categories.

### Python Practice Notebook 6: Standardizing Day and Night Images

Link: [Learning: Computer Vision 6]()

Resources: 1700 day/night images at multiple location and multiple timing (Train - Test: 40:60)

***Currently work in progress...***

> Task: Using the day/night image from the library, try separating the day and night image based on RGB and HSV.

### 12. Average Brightness

Here were the steps we took to extract the average brightness of an image.

- Convert the image to HSV color space (the Value channel is an approximation for brightness)
- Sum up all the values of the pixels in the Value channel
- Divide that brightness sum by the area of the image, which is just the width times the height.

This gave us one value: the average brightness or the average Value of that image.

### Python Practice Notebook 7: Standardizing Day and Night Images

Link: [Learning: Computer Vision 7]()

Resources: 1700 day/night images at multiple location and multiple timing (Train - Test: 40:60)

***Currently work in progress...***

> Task: Using the day/night image from the library, try separating the day and night image based on RGB and HSV.

Work tasks:

- Find the average brightness of the images.
- Create a classifier model to find out the value boundary for day and night
- Test the average brightness calculation function on an image

### Python Practice Notebook 8: Accuracy and Misclassification

Link: [Learning: Computer Vision 8]()

Resources: 1700 day/night images at multiple location and multiple timing (Train - Test: 40:60)

***Currently work in progress...***

> Task: Using the day/night image from the library, try separating the day and night image based on RGB and HSV.

Work tasks:

- Iterate through the whole test set to get the prediction result
- Take the prediction result to calculate the accuracy of the day/night classification model.

### End of project

*Review and the Computer Vision Pipeline*

In this lesson, you’ve really made it through a lot of material, from learning how images are represented to programming an image classifier!

You approached the classification challenge by completing each step of the Computer Vision Pipeline step-by-step. First by looking at the classification problem, visualizing the image data you were working with, and planning out a complete approach to a solution.

The steps include pre-processing images so that they could be further analyzed in the same way, this included changing color spaces. Then we moved on to feature extraction, in which you decided on distinguishing traits in each class of image, and tried to isolate those features! You may note that skipped the pipeline step of "Selecting Areas of Interest," and this is because we focused on classifying an image as a whole and did not need to break it up into different segments, but we'll see where this step can be useful later in this course.

Finally, you created a complete classifier that output a label or a class for a given image, and analyzed your classification model to see its accuracy!

Project for this lesson: [Day and Night Classifier Model]()

## Lesson 2: Filters and Detection

### 13. Filters and Finding Edges

In computer vision, a **filter** refers to a mathematical operation or convolutional kernel applied to an image to perform various types of image processing tasks such as smoothing, sharpening, edge detection, or feature extraction. Filters are commonly used to enhance or extract certain features or characteristics from images, making them more suitable for subsequent analysis or interpretation.

Here's a brief overview of some common types of filters used in computer vision:

13.1. **Smoothing Filters**: Smoothing filters are used to reduce noise and blur images by averaging pixel values within a neighborhood. Examples include the Gaussian filter, which applies a weighted average based on a Gaussian distribution, and the median filter, which replaces each pixel value with the median value in its neighborhood.

13.2. **Edge Detection Filters**: Edge detection filters highlight the boundaries between regions of different intensity or color in an image. Examples include the Sobel filter, which calculates the gradient of the image intensity in both the horizontal and vertical directions, and the Canny edge detector, which uses multiple stages of processing to detect edges with high precision and low false positives.

13.3. **Sharpening Filters**: Sharpening filters enhance the contrast and detail in an image by emphasizing edges and fine features. Examples include the Laplacian filter, which highlights regions of rapid intensity change, and the unsharp mask filter, which subtracts a blurred version of the image from the original to enhance edges.

13.4. **Feature Extraction Filters**: Feature extraction filters are designed to detect specific patterns or structures in an image, such as corners, blobs, or texture patterns. Examples include the Harris corner detector, which identifies corner points based on local intensity variations, and the Gabor filter, which extracts texture features at different orientations and scales.

13.5. **Frequency Domain Filters**: Frequency domain filters operate on the frequency components of an image, such as its Fourier transform. Examples include high-pass filters, which emphasize high-frequency components associated with edges and fine details, and low-pass filters, which smooth the image by removing high-frequency noise.

Filters are typically applied to images using convolution, where the filter kernel is slid over the image and its values are multiplied with corresponding pixel values to compute the output at each location. Filters play a crucial role in many computer vision tasks, including image enhancement, feature detection, object recognition, and image segmentation.

### 14. Frequency in Images

We have an intuition of what frequency means when it comes to sound. High-frequency is a high pitched noise, like a bird chirp or violin. And low frequency sounds are low pitch, like a deep voice or a bass drum. For sound, frequency actually refers to how fast a sound wave is oscillating; oscillations are usually measured in cycles/s (Hz(opens in a new tab)), and high pitches and are made by high-frequency waves. Examples of low and high-frequency sound waves are pictured below. On the y-axis is amplitude, which is a measure of sound pressure that corresponds to the perceived loudness of a sound, and on the x-axis is time.

> ***High and Low Frequency***

Similarly, frequency in images is a rate of change. But, what does it means for an image to change? Well, images change in space, and a high frequency image is one where the intensity changes a lot. And the level of brightness changes quickly from one pixel to the next. A low frequency image may be one that is relatively uniform in brightness or changes very slowly. This is easiest to see in an example.

Most images have both high-frequency and low-frequency components. In the image above, on the scarf and striped shirt, we have a high-frequency image pattern; this part changes very rapidly from one brightness to another. Higher up in this same image, we see parts of the sky and background that change very gradually, which is considered a smooth, low-frequency pattern.

**High-frequency components also correspond to the edges of objects in images,** which can help us classify those objects.

> ***Fourier Transform***

The Fourier Transform (FT) is an important image processing tool that is used to decompose an image into its frequency components. The output of an FT represents the image in the frequency domain, while the input image is the spatial domain (x, y) equivalent. In the frequency domain image, each point represents a particular frequency contained in the spatial domain image. So, for images with a lot of high-frequency components (edges, corners, and stripes), there will be a number of points in the frequency domain at high frequency values.

> ***Understanding Frequency in Images***

#### Basic Concept

In the realm of image processing and computer vision, "frequency" refers to the rate of change of intensity values in an image. It's similar to the concept of frequency in sound waves, but instead of audio pitches, it deals with changes in visual details.

#### High and Low Frequency

- High Frequency: High-frequency content in images refers to the edges and fine details where the intensity changes abruptly over a small area. These are often found at the boundaries of objects or within textures.
- Low Frequency: Low-frequency content corresponds to smooth and slowly varying parts of the image, like open skies or solid-color backgrounds. These areas have minimal changes in intensity or color over large regions.

#### Importance in Image Processing

- Filtering: Understanding frequencies is crucial in image filtering. High-pass filters are used to enhance or detect edges (high-frequency components), while low-pass filters are used for blurring or noise reduction, focusing on low-frequency areas.
- Compression: Image compression techniques often exploit frequency components. For example, JPEG compression removes high-frequency content to a degree, as the human eye is less sensitive to fine details.
- Analysis: Frequency analysis is essential in many advanced image processing techniques, including feature extraction, edge detection, and texture analysis.
Frequency Domain
- The frequency content of an image can be analyzed using tools like the Fourier Transform, which converts the spatial representation of an image into a frequency representation. This is especially useful for filtering, compression, and noise reduction tasks.

#### Summary

In summary, frequency in images relates to the rate of change in intensity values. High-frequency elements represent rapid changes (like edges), while low-frequency elements correspond to gradual changes. The concept of frequency is fundamental in various image processing tasks, enabling effective manipulation and analysis of visual information.

### Python Practice Notebook 8: Fourier Transforms

Link: [Learning: Computer Vision 8]()

Resources: [OpenCV Documentary](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html)

> Task: Learn about the frequency in images, try testing the **Fourier Transforms**

In this notebook, we will experiment the Fourier Transforms with 2 images

- Image 1: Striped black and white, which image frequency transitions happen very often and sharply
- Image 2: Pink image, no frequency changes at all

### 15. High-pass Filters

**High-pass filters**, also known as high-frequency filters, are image processing filters that emphasize the high-frequency components of an image while suppressing or attenuating the low-frequency components. These filters are commonly used in image processing and computer vision for tasks such as edge detection, sharpening, and noise reduction.

> ***What are Edges***

**Edges** are areas in an image where the intensity changes very quickly, and they often indicate object boundaries.

> ***Convolution Kernels***

**Convolution kernel**, also known simply as a kernel or filter, is a small matrix of weights that is applied to an input image using the mathematical operation known as convolution. Convolution kernels are fundamental to various image processing tasks, including filtering, feature extraction, and image transformation.

> ***Edge Handling***

Kernel convolution relies on centering a pixel and looking at its surrounding neighbors. So, what do you do if there are no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are listed below. It’s most common to use padding, cropping, or extension. In extension, the border pixels of an image are copied and extended far enough to result in a filtered image of the same size as the original image.

**Extend** The nearest border pixels are conceptually extended as far as necessary to provide values for the convolution. Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.

**Padding** The image is padded with a border of 0's, black pixels.

**Crop** Any pixel in the output image which would require values from beyond the edge is skipped. This method can result in the output image being slightly smaller, with the edges having been cropped.

### 16. Gradients and Sobel Filters

> ***Gradients***

**Gradients** are a measure of intensity change in an image, and they generally mark object boundaries and changing areas of light and dark. If we think back to treating images as functions, F(x, y), we can think of the gradient as a derivative operation F ’ (x, y). Where the derivative is a measurement of intensity change.

> ***Sobel filters***

**Sobel filters** are a type of edge detection filter used in image processing and computer vision to identify edges and gradients in images. They are named after their inventor, Irwin Sobel, and are widely used for tasks such as feature extraction, object detection, and image segmentation.

> ***Magnitude***

Sobel also detects which edges are strongest. This is encapsulated by the magnitude of the gradient; the greater the magnitude, the stronger the edge is. The magnitude, or absolute value, of the gradient, is just the square root of the squares of the individual x and y gradients. For a gradient in both the x and y directions, the magnitude is the square root of the sum of the squares.

> ***Direction***

In many cases, it will be useful to look for edges in a particular orientation. For example, we may want to find lines that only angle upwards or point left. By calculating the direction of the image gradient in the x and y directions separately, we can determine the direction of that gradient!

### Python Practice Notebook 9: Finding Edges

Link: [Learning: Computer Vision 9]()

Resources: [Computer Vision Practice Notebook 9]()

> Task: Learn about custom kernels and edges.

### 17. Low-pass Filters

...
