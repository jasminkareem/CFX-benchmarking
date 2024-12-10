import tensorflow as tf
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, Dense, Input, Conv2D, MaxPooling2D, Flatten, Layer
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from time import time
from alibi.explainers import Counterfactual,  CounterfactualProto, CEM
import pandas as pd
import os
import os.path
import argparse
import torch
import pathlib



def ae_model():
    # encoder
    x_in = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x_in)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    encoder = Model(x_in, encoded)

    # decoder
    dec_in = Input(shape=(14, 14, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(dec_in)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    decoder = Model(dec_in, decoded)

    # autoencoder = encoder + decoder
    x_out = decoder(encoder(x_in))
    autoencoder = Model(x_in, x_out)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder, decoder


"""
### Code for training and saving autoencoder model (taken from Alibi package)
ae, enc, dec = ae_model()
ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
ae.save('mnist_ae.h5', save_format='h5')
enc.save('mnist_enc.h5', save_format='h5')

"""

def main():
    # Add arguments
     parser = argparse.ArgumentParser(description="Experiments for alibi implementation of Counterfactual and ProtoCounterfactual")
     parser.add_argument('--path_classifier', help='Path to the image classifier file', default='mnist_resnet8_weights.h5')
     parser.add_argument('--dataset', help='dataset that you would like to use. Default: MNIST', default='mnist')
     parser.add_argument('--n_samples', help='Number of samples/instances you want to create explanations for', default=100) 
     parser.add_argument('--path_explanations', help='Path to output explanations and original instance', default='mnist_resnet8_output/')
     #parser.add_argument('--target_classes', help='File with target classes for each instances, if only specific classes are of interest', default='target_classes_1.csv')
     args = parser.parse_args()
     # Define arguments into variables
     path_classifier = args.path_classifier 
     dataset = args.dataset 
     n_samples = args.n_samples 
     path_explanations = args.path_explanations
     #target_classes = args.target_classes

     save_path = '/ivi/ilps/personal/jkareem/reproducing-CFX-methods-data'
     #pathlib.Path(save_path + '/mnist_alibi_cnn_output').mkdir(parents=True, exist_ok=True)


     if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = np.reshape(x_train, x_train.shape + (1,))
        x_test = np.reshape(x_test, x_test.shape + (1,))
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        xmin, xmax = -1, 1
        x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
        x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin

        n_classes = 10
        #ae, enc, dec = ae_model()
        #ae.fit(x_train, x_train, batch_size=128, epochs=4, validation_data=(x_test, x_test), verbose=0)
        #ae.save('mnist_ae_neg.h5', save_format='h5')
        #enc.save('mnist_enc_neg.h5', save_format='h5')

       

        ae = load_model(os.path.join(save_path, 'mnist_ae.h5'))
        enc = load_model(os.path.join(save_path, 'mnist_enc.h5'), compile=False)


     else:
        print("add alternative dataset here")
        quit()
         
     
    

     if path_classifier == 'mnist_cnn_6_128.h5':
        # Load model
        model = load_model(os.path.join(save_path, path_classifier))
        model.predict(x_test)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy: ', score[1])

     elif path_classifier == 'mnist_relu_4_1024.h5':
        # Load model
        model = load_model(os.path.join(save_path, path_classifier))
        x_reshaped = x_test.reshape(10000, 1, 28, 28)
        model.predict(x_reshaped)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        score = model.evaluate(x_reshaped, y_test, verbose=0)
        print('Test accuracy: ', score[1])

         # Adapt and rename model
        config = model.get_config()
        new_input = Input(shape=(28, 28, 1))
        x = new_input
        for layer in model.layers[2:]:
            x = layer(x)
            
        model = Model(new_input, x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 


     elif path_classifier == 'mnist_alibi_cnn.h5':
        # Load model
        model = load_model(os.path.join(save_path, path_classifier))
        x_reshaped = x_test.reshape(10000, 1, 28, 28)
        model.predict(x_test)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy: ', score[1])

     elif path_classifier == 'mnist_lenet.h5':
        # Load model
        model = load_model(os.path.join(save_path, path_classifier))
        x_reshaped = x_test.reshape(10000, 1, 28, 28)
        model.predict(x_test)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy: ', score[1])

     elif path_classifier == 'mnist_schut.h5':
        # Load model
        model = load_model(os.path.join(save_path, path_classifier))
        x_reshaped = x_test.reshape(10000, 1, 28, 28)
        model.predict(x_test)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy: ', score[1])

     elif path_classifier == 'mnist_resnet8_weights.h5':
        # Basic Block
        class BasicBlock(Layer):
            expansion = 1

            def __init__(self, in_channels, out_channels, stride=1, downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = Conv2D(out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)
                self.bn1 = BatchNormalization()
                self.relu = Activation('relu')
                self.conv2 = Conv2D(out_channels, kernel_size=3, strides=1, padding="same", use_bias=False)
                self.bn2 = BatchNormalization()
                self.downsample = downsample

            def call(self, inputs, training=False):
                identity = inputs
                if self.downsample is not None:
                    identity = self.downsample(inputs)

                x = self.conv1(inputs)
                x = self.bn1(x, training=training)
                x = self.relu(x)

                x = self.conv2(x)
                x = self.bn2(x, training=training)

                x += identity
                x = self.relu(x)

                return x


        # ResNet
        class ResNet(Model):
            def __init__(self, block, layers, num_classes=10):
                super(ResNet, self).__init__()
                self.in_channels = 16

                # Initial convolution
                self.conv1 = Conv2D(16, kernel_size=3, strides=1, padding="same", use_bias=False)
                self.bn1 = BatchNormalization()
                self.relu = Activation('relu')

                # Residual layers
                self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
                self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

                # Classification head
                self.avgpool = GlobalAveragePooling2D()
                self.fc = Dense(num_classes, activation='softmax')

            def _make_layer(self, block, out_channels, blocks, stride):
                downsample = None
                if stride != 1 or self.in_channels != out_channels:
                    downsample = tf.keras.Sequential([
                        Conv2D(out_channels, kernel_size=1, strides=stride, use_bias=False),
                        BatchNormalization()
                    ])

                layers_list = [block(self.in_channels, out_channels, stride, downsample)]
                self.in_channels = out_channels
                for _ in range(1, blocks):
                    layers_list.append(block(self.in_channels, out_channels))

                return tf.keras.Sequential(layers_list)

            def call(self, inputs, training=False):
                x = self.conv1(inputs)
                x = self.bn1(x, training=training)
                x = self.relu(x)

                x = self.layer1(x, training=training)
                x = self.layer2(x, training=training)
                x = self.layer3(x, training=training)

                x = self.avgpool(x)
                x = self.fc(x)

                return x


        # Instantiate the Model
        model = ResNet(BasicBlock, [1, 1, 1])  # Using a shallow ResNet for MNIST
        model.build(input_shape=(None, 28, 28, 1))  # MNIST input size
        model.load_weights(os.path.join(save_path, path_classifier))

        # test model
        x_reshaped = x_test.reshape(10000, 1, 28, 28)
        model.predict(x_test)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy: ', score[1])

    
        
     
    # Explanation
     
     target_classes = range(n_classes)
     feature_range = (x_test.min(),x_test.max())

     experiment_data = pd.DataFrame(columns=['Instance', 'Name', 'optim_time','Original Label', 'Original Prediction', 'Target Label', 'New Prediction'])



     for sample in range(n_samples):
         X = x_test[sample].reshape((1,) + x_test[sample].shape)
         shape = (1,) + x_train.shape[1:]
         original_label = np.argmax(y_test[sample])
         original_prediction = np.argmax(model.predict(X))
         print("original pred:", original_prediction)

         for target_class in target_classes:
             print('target class:', target_class)
             if original_prediction == target_class:
                 continue
             for name in ['alibi-CF', 'alibi-Proto-CF']:
                 print(" ")
                 print("-------------------------------")
                 print(sample, name)
                 print("-------------------------------")
                 temp_data = pd.DataFrame()
                 start_time = time()
                 if name == 'alibi-CF':
                    # Parameters CF
                    target_proba = 1.0
                    tol = 0.1
                    max_iter = 1000 # Good choice because PIECE runs for max 1000 iterations
                    lam_init = 1e-1 
                    max_lam_steps = 10
                    learning_rate_init = 0.1
                    cf = Counterfactual(model, shape=shape, target_proba=target_proba, tol=tol, target_class=target_class, max_iter=max_iter, lam_init=lam_init, max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init, feature_range=feature_range)
                    explanation = cf.explain(X)
    

                 elif name == 'alibi-Proto-CF':
                     gamma = 100.
                     theta = 100.
                     c_init = 1.
                     c_steps = 2
                     k = 5
                     max_iterations = 1000
                     cf = CounterfactualProto(model, shape, gamma=gamma, theta=theta, ae_model=ae, enc_model=enc, max_iterations=max_iterations, feature_range=feature_range, c_init=c_init, c_steps=c_steps)
                     cf.fit(x_train)
                     explanation = cf.explain(X, k=k, k_type='mean', target_class=[target_class])


                     
                 optim_time = time() - start_time
                 # Save explanation as txt file
                 if explanation.cf == None:
                    temp_data = pd.concat([temp_data, pd.DataFrame([{'Instance': sample, 'Name': name, 'optim_time': optim_time, 'Original Label': original_label, 'Original Prediction': original_prediction, 'Target Label': target_class, 'New Prediction': None}])], ignore_index=True)


                 else:
                     perturbed_image = explanation.cf['X'].reshape(1,1,28,28)
                     
                     

                     #with open(path_explanations + name + '/instance_' + str(sample) + '_target_' + str(target_class) +'.txt', 'w') as outfile:
                     #   np.savetxt(outfile, perturbed_image)
                     image_tensor = torch.from_numpy(perturbed_image)
                     print(image_tensor.shape)
                     outfile = path_explanations + name + '/instance_' + str(sample) + '_target_' + str(target_class) + '.pt'
                     new_outfile = os.path.join(save_path, outfile)
                     pathlib.Path(os.path.join(save_path, path_explanations + name)).mkdir(parents=True, exist_ok=True)
                     torch.save(image_tensor, new_outfile)
                     
                     new_prediction = np.argmax(model.predict(explanation.cf['X']))
                     temp_data = pd.concat([temp_data, pd.DataFrame([{'Instance': sample, 'Name': name, 'optim_time': optim_time, 'Original Label': original_label, 'Original Prediction': original_prediction, 'Target Label': target_class, 'New Prediction': new_prediction}])], ignore_index=True)



        
                 print(temp_data.head())
                 experiment_data = pd.concat([experiment_data, temp_data])

     print(experiment_data.head())
     csv_file_name = path_explanations +'output_Alibi_'+ dataset +'_data.csv'
     new_path_csv = os.path.join(save_path,csv_file_name)
     print(new_path_csv)
     experiment_data.to_csv(new_path_csv, index=False) 





if __name__ == "__main__":
     main()

            



            





