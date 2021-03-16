# Bat_Call_Detection

The aim of the Bluebat project is to develop a sensor that identifies the calls of common pipistrelles when they call in its vicinity. The problem we are facing is a classification problem because we have to determine whether or not the sound recorded by the sensor microphone is a pipistrelle call.
To address this type of problem, we chose to use a deep learning approach. We decided to model the problem in a particular way, to make use of deep learning for image recognition. More specifically, 2D convolutional neural networks are the most suitable. We will proceed using PyTorch. 

We must then transform the sounds into images. This is possible thanks to sonograms, which give an image from a sound. The frequency is represented as a function of time, and a colour scale is used to represent the intensity: the more the colour tends towards red, the stronger the intensity; conversely, the more it tends towards blue, the weaker the intensity.

<img title="Sonogram of a pipistrelle call" src="https://user-images.githubusercontent.com/69425777/111322878-ab62d080-8669-11eb-9641-9173c760dcdb.png" alt="drawing" width="400"/>


<img title="Outdoor noise sonogram" src="https://user-images.githubusercontent.com/69425777/111322919-b4ec3880-8669-11eb-8d7e-15a04b83227d.png" alt="drawing" width="400"/>
 
Looking at the sonogram of pipistrelles' calls, we can see that they emit a rather characteristic call, called a "squeak": it is short, of high intensity, and has a particular shape that takes the form of a decreasing exponential. At this frequency, the sound is audible; it is a social call used for communication, courtship or to claim territory. This atypical call pattern can be recognised with an image recognition algorithm. The objective will be to differentiate it from "noise". Here we will call "noise" all sounds that are not bat calls: it can be the wind blowing, people talking, the noise of a car, the call of another animal, etc...

Thus, the problem was modelled as follows: the bat calls and the noise are transformed into sonograms. It is then necessary to develop a classification algorithm that will take a sonogram as input, and which will determine as output whether it is a pipistrelle sonogram or noise. This algorithm will be a 2D convolutional neural network. 

