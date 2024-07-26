### Real vs. Fake News Classifier Project

#### Project Overview
The goal of this project was to develop a robust model to classify news articles as real or fake. This task involved dealing with textual data, which presents unique challenges such as handling large vocabularies and varying text lengths. The project utilized various deep learning techniques to build and fine-tune a model that can accurately distinguish between real and fake news.

#### Dataset
The dataset used for this project consisted of news articles labeled as either real or fake. The data was preprocessed to convert the text into a suitable format for training the model. This included tokenizing the text, padding sequences to ensure uniform input lengths, and splitting the data into training, validation, and test sets.

#### Model Architecture
The best performing model in this project had the following architecture:

1. **Embedding Layer**:
   - Converts the input text sequences into dense vectors of fixed size.
   - Input dimension: Vocabulary size
   - Output dimension: 128

2. **Bidirectional LSTM Layer**:
   - A bidirectional LSTM (Long Short-Term Memory) layer to capture dependencies in both forward and backward directions in the text data.
   - Output dimension: 128

3. **Dropout Layer**:
   - Applied dropout to prevent overfitting.
   - Dropout rate: 0.5

4. **Conv1D Layer**:
   - A 1D convolutional layer to extract local features from the text.
   - Number of filters: 64
   - Kernel size: 5

5. **Global Max Pooling Layer**:
   - Applied global max pooling to reduce the dimensionality of the feature maps.

6. **Dense Layer**:
   - A fully connected layer with 64 units and ReLU activation function.

7. **Dropout Layer**:
   - Another dropout layer to prevent overfitting.
   - Dropout rate: 0.5

8. **Output Layer**:
   - A single unit with a sigmoid activation function for binary classification (real vs. fake).

#### Model Summary
```
 Model: "sequential_6"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_6 (Embedding)     (None, 1000, 128)         1280000   
                                                                 
 bidirectional (Bidirection  (None, 1000, 128)         98816     
 al)                                                             
                                                                 
 dropout_2 (Dropout)         (None, 1000, 128)         0         
                                                                 
 conv1d (Conv1D)             (None, 996, 64)           41024     
                                                                 
 global_max_pooling1d (Glob  (None, 64)                0         
 alMaxPooling1D)                                                 
                                                                 
 dense_14 (Dense)            (None, 64)                4160      
                                                                 
 dropout_3 (Dropout)         (None, 64)                0         
                                                                 
 dense_15 (Dense)            (None, 1)                 65        
                                                                 
=================================================================
Total params: 1424065 (5.43 MB)
Trainable params: 1424065 (5.43 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

#### Model Training and Performance
The model was trained using the training dataset, with a portion of the data used for validation to monitor the model's performance during training. The Adam optimizer and binary cross-entropy loss function were used for training. Dropout layers were included to prevent overfitting.

The model achieved impressive results:
- **Training Loss**: 0.0238
- **Training Accuracy**: 99.44%
- **Validation Loss**: 0.1625
- **Validation Accuracy**: 96.66%

The performance on the test dataset was also outstanding:
- **Test Loss**: 0.0857
- **Test Accuracy**: 99.05%

#### Conclusion
This project successfully demonstrated the use of advanced deep learning techniques to build an effective real vs. fake news classifier. The combination of an embedding layer, bidirectional LSTM, Conv1D, and global max pooling, along with dropout layers, resulted in a model with high accuracy and low loss on both training and test datasets. The achieved performance metrics highlight the model's capability to accurately classify news articles, making it a valuable tool for identifying fake news in real-world applications.


You can Download glove.6b from Browser
