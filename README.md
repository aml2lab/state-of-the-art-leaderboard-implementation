# state-of-the-art-leaderboard-implementation
Show better implementation of AI problems and applications so LLM and AI can be useful in real world.

requirements.txt summarized the simulation enviroment.

## mnist

The MNIST benchmark is held by models that employ ensembles of CNNs. I presented a reference implementation that can be run in a simple PC that has one AMD RX 7900 XTX gpu.

The implmentation code is in mnist folder. And the test results has reached 
99.66% accuracy.

## fashion mnist

The state-of-the-art (SOTA) for classifying the Fashion MNIST dataset, a popular benchmark for machine learning models, has reached an impressive accuracy of 99.65%. This remarkable performance was achieved in late 2024 by a Convolutional Neural Network (CNN) model designated as CNN-3-128, enhanced with data augmentation techniques.

I have used above approach simplified in my PC with amd gpu RX 7900 XTX

The implmentation code is in fashion-mnist folder. And the test results here has 92.97% accuracy.

## cifar 10

Many ViT base models has claimed SOTA of cifar 10 without external data has reached accuracy of 99.61%. 

The code in cifar10 folder here use amd comsumer gpu and reached 96.78% accuracy.

## cifar 100

The sota of cifar 100 without external data has achieved 91.7% with efficient net, while ViT reached 82.45%.

In cifar100 folder, our model uses efficient net in PC with amd gpu RX 7900 XTX has 87.09% accuracy.