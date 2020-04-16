""" This class inherits a network architecture and performs various functions on a define architecture like training
 and predicting"""

import os

import pickle
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import concatenate, int_shape
import tensorflow.keras.backend as K
import tensorflow as tf
from provided_code.general_functions import get_paths, make_directory_and_return_path, sparse_vector_function
from provided_code.network_architectures import DefineDoseFromCT
from provided_code.dropout import CustomSpatialDropout3D
from provided_code.network_architectures import MultiplyByScalar
from provided_code.spectral_normalization import apply_spectral_normalization, DenseSN, ConvSN3D, ConvSN3DTranspose
from provided_code.Cones import Cones3D, cone3D_aperture_regularization, cone3D_direction_regularization, ConeCoordinatesRegularization, cone3D_coordinates_normalization
from scipy.ndimage import zoom

custom_objects = {"CustomSpatialDropout3D":CustomSpatialDropout3D ,
                  "MultiplyByScalar":MultiplyByScalar,
                  "DenseSN":DenseSN,
                  "ConvSN3D":ConvSN3D,
                  "ConvSN3DTranspose":ConvSN3DTranspose,
                  "Cones3D":Cones3D,
                  "cone3D_coordinates_normalization":cone3D_coordinates_normalization,
                  "cone3D_aperture_regularization":cone3D_aperture_regularization,
                  "cone3D_direction_regularization":cone3D_direction_regularization,
                  "ConeCoordinatesRegularization":ConeCoordinatesRegularization}

class PredictionModel(DefineDoseFromCT):

    def __init__(self, data_loader, results_patent_path, model_name, stage='training'):
        """
        Initialize the Prediction model class
        :param data_loader: An object that loads batches of image data
        :param results_patent_path: The path at which all results and generated models will be saved
        :param model_name: The name of your model, used when saving and loading data
        """
        super(PredictionModel, self).__init__()

        # set attributes for data shape from data loader
        self.data_loader = data_loader
        self.patient_shape = data_loader.patient_shape
        self.full_roi_list = data_loader.full_roi_list
        self.model_name = model_name

        # Define training parameters
        self.epoch_start = 0  # Minimum epoch (overwritten during initialization if a newer model exists)
        self.epoch_last = 200  # When training will stop

        # Define image sizes
        self.dose_shape = (*self.patient_shape, 1)
        self.ct_shape = (*self.patient_shape, 1)
        self.roi_masks_shape = (*self.patient_shape, len(self.full_roi_list))

        # Define filter and stride lengths
        self.filter_size = (4, 4, 4)
        self.stride_size = (2, 2, 2)

        # Define the initial number of filters in the model (first layer)
        self.initial_number_of_filters = 8

        # Define model optimizer
        self.gen_optimizer = Adam(lr=0.00002, decay=0.001, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(lr=0.0002, decay=0.001, beta_1=0.5, beta_2=0.999)

        # Make directories for data and models
        model_results_path = '{}/{}'.format(results_patent_path, model_name)
        self.model_dir = make_directory_and_return_path('{}/models'.format(model_results_path))
        self.prediction_dir = '{}/{}-predictions'.format(model_results_path, stage)

        # Make template for model path
        self.generator_path_template = '{}/generator_epoch_'.format(self.model_dir)
        self.discriminator_path_template  = '{}/discriminator_epoch_'.format(self.model_dir)
        
        # ajout de l'historique de training
        self.training_history_path = '{}/training_history.hist'.format(self.model_dir)
        self.d_loss_history = []
        self.g_loss_history = []
        self.d_acc_history = []

    def train_adversarial_model(self, epochs=200, epochs_discriminator = 0, save_frequency=5, keep_model_history=2):
        """
        Train the model over several epochs
        :param epochs: the number of epochs the model will be trained over
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models back are kept (anything older than
        save_frequency*keep_model_history epochs)
        :return: None
        """
        # Define new models, or load most recent model if model already exists
        self.epoch_last = epochs
        begin_training = time.clock()

        # Check if training has already finished
        if self.epoch_start == epochs:
            return

        else:
            # Start training GAN
            num_batches = self.data_loader.number_of_batches()
            for e in range(self.epoch_start, epochs):
                # Begin a new epoch
                begin_epoch = time.clock()
                print('\n'+80*'='+'\n')
                print('Epoch number {} out of {}'.format(e+1, epochs))
                print('\n'+80*'='+'\n')

                self.data_loader.on_epoch_end()  # Shuffle the data after each epoch
                
                d_loss = 0
                d_acc = 0
                g_loss = np.zeros(3)
                for i in range(num_batches):
                    print("EPOCH #{}/{} - BATCH #{}/{}".format(e+1, epochs, i+1, num_batches))
                    # Load a subset of the data and train the network with the data
                    begin_batch = time.clock()
                    print("Loading the data...")
                    loading_start = time.clock()
                    image_batch = self.data_loader.get_batch(i)
                    loading_time = time.clock() - loading_start
                    print("Data loaded in {:.3f}".format(loading_time))

                    print("Discriminator training...")

                    begin_discriminator = time.clock()
                    disc_history = self.train_discriminator_on_batch(i, image_batch, e)
                    d_loss += disc_history[0]
                    d_acc += disc_history[1]
                    batch_time = time.clock() - begin_batch
                    discriminator_time = time.clock() - begin_discriminator
                    print("Discriminator trained in {:.3f}".format(discriminator_time))
                    
                    if e < epochs_discriminator:
                        print("Only training discriminator for now.")
                    else:
                        print("Generator training...")
                        begin_generator = time.clock()
                        g_loss += self.train_generator_on_batch(i, image_batch,e)
                        generator_time = time.clock() - begin_generator
                        print("Generator trained in {:.3f}".format(generator_time))
                        print("Attention weight sigma ", self.get_sigma())
                    batch_time = time.clock() - begin_batch
                    print("Total training time : {:.3f}s".format(batch_time))

                    print('\n'+80*'-'+'\n')
                self.d_loss_history.append(d_loss/num_batches)
                self.d_acc_history.append(d_acc/num_batches)
                self.g_loss_history.append(g_loss/num_batches)

                # Plotting the losses
                self.plot_history()
                self.plot_predictions(1)
                if self.gen_use_attention_layer:
                    self.visualize_attention(self.data_loader.get_batch(0))

                epoch_time = time.clock() - begin_epoch
                print("Epoch {} performed in {:.3f}s\n".format(e+1, epoch_time))

                # Create epoch label and save models at the specified save frequency
                current_epoch = e + 1
                if 0 == np.mod(current_epoch, save_frequency):
                    self.save_model_and_delete_older_models(current_epoch, save_frequency, keep_model_history)

            training_time = time.clock() - begin_training
            print("Training done in {:.3f}s\n".format(training_time))

    def train_generator_model(self, epochs=200, epochs_discriminator = 0, save_frequency=5, keep_model_history=2):
            """
            Train the model over several epochs
            :param epochs: the number of epochs the model will be trained over
            :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
            :param keep_model_history: how many models back are kept (anything older than
            save_frequency*keep_model_history epochs)
            :return: None
            """
            # Define new models, or load most recent model if model already exists
            self.epoch_last = epochs
            begin_training = time.clock()

            # Check if training has already finished
            if self.epoch_start == epochs:
                return

            else:
                # Start training GAN
                history = []
                num_batches = self.data_loader.number_of_batches()
                for e in range(self.epoch_start, epochs):
                    # Begin a new epoch
                    begin_epoch = time.clock()
                    print('\n'+80*'='+'\n')
                    print('Epoch number {} out of {}'.format(e+1, epochs))
                    print('\n'+80*'='+'\n')

                    self.data_loader.on_epoch_end()  # Shuffle the data after each epoch
                    

                    g_loss = 0
                    for i in range(num_batches):
                        print("EPOCH #{}/{} - BATCH #{}/{}".format(e+1, epochs, i+1, num_batches))
                        # Load a subset of the data and train the network with the data
                        begin_batch = time.clock()
                        print("Loading the data...")
                        loading_start = time.clock()
                        image_batch = self.data_loader.get_batch(i)
                        loading_time = time.clock() - loading_start
                        print("Data loaded in {:.3f}".format(loading_time))
                        
                        print("Generator training...")
                        begin_generator = time.clock()
                        g_loss += self.train_generator_alone_on_batch(i, image_batch,e)
                        generator_time = time.clock() - begin_generator
                        print("Generator trained in {:.3f}".format(generator_time))
                        print("Attention weight sigma ", self.get_sigma())

                        batch_time = time.clock() - begin_batch
                        print("Total training time : {:.3f}s".format(batch_time))

                        print('\n'+80*'-'+'\n')
                    history.append(g_loss/num_batches)

                    # Plotting the losses
                    plt.plot(history, '+')
                    self.plot_predictions(5)
                    if self.gen_use_attention_layer:
                        self.visualize_attention(self.data_loader.get_batch(0))

                    epoch_time = time.clock() - begin_epoch
                    print("Epoch {} performed in {:.3f}s\n".format(e+1, epoch_time))

                training_time = time.clock() - begin_training
                print("Training done in {:.3f}s\n".format(training_time))

    def save_model_and_delete_older_models(self, current_epoch, save_frequency, keep_model_history):
        """
        Save the current model and delete old models, based on how many models the user has asked to keep. We overwrite
        files (rather than deleting them) to ensure the user's trash doesn't fill up.
        :param current_epoch: the current epoch number that is being saved
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models back are kept (anything older than
        save_frequency*keep_model_history epochs)
        """

        # Save the model to a temporary path
        temporary_generator_path = '{}_temp.h5'.format(self.generator_path_template)
        temporary_discriminator_path = '{}_temp.h5'.format(self.discriminator_path_template)
        self.generator_.save(temporary_generator_path)
        self.discriminator_.save(temporary_discriminator_path)
        # Define the epoch that should be over written
        epoch_to_overwrite = current_epoch - keep_model_history * save_frequency
        # Make appropriate path to save model at
        if epoch_to_overwrite > 0:
            generator_to_delete_path = '{}{}.h5'.format(self.generator_path_template, epoch_to_overwrite)
            discriminator_to_delete_path = '{}{}.h5'.format(self.discriminator_path_template, epoch_to_overwrite)
        else:
            generator_to_delete_path = '{}{}.h5'.format(self.generator_path_template, current_epoch)
            discriminator_to_delete_path = '{}{}.h5'.format(self.discriminator_path_template, current_epoch)

        # Save model
        os.rename(temporary_generator_path, generator_to_delete_path)
        os.rename(temporary_discriminator_path, discriminator_to_delete_path)
        
        # We systematically save the training history
        history_dict = {'d_loss_history':self.d_loss_history, 'g_loss_history':self.g_loss_history,\
                        'd_acc_history':self.d_acc_history}
        history_file = open(self.training_history_path,'wb')
        pickle.dump(history_dict, history_file)
        history_file.close()
        
        # The code below is a hack to ensure the Google Drive trash doesn't fill up
        if epoch_to_overwrite > 0:
            final_save_discriminator_path = '{}{}.h5'.format(self.discriminator_path_template, current_epoch)
            final_save_generator_path = '{}{}.h5'.format(self.generator_path_template, current_epoch)
            os.rename(discriminator_to_delete_path, final_save_discriminator_path)
            os.rename(generator_to_delete_path, final_save_generator_path)

    def initialize_networks(self):
        """
        Load the newest model, or if no model exists with the appropriate name a new model will be created.
        :return:
        """
        # Initialize variables for models
        all_models = get_paths(self.model_dir, ext='h5')

        # Get last epoch of existing models if they exist
        for model_name in all_models:
            model_epoch_number = model_name.split('_epoch_')[-1].split('.h5')[0]
            if model_epoch_number.isdigit():
                self.epoch_start = max(self.epoch_start, int(model_epoch_number))

        # Build new models or load most recent old model if one exists
        if self.epoch_start >= self.epoch_last:
            print('Model fully trained, loading model from epoch {}'.format(self.epoch_last))
            return 0, 0, 0, self.epoch_last

        elif self.epoch_start >= 1:
            # If models exist then load them
            self.generator_ = load_model('{}{}.h5'.format(self.generator_path_template, self.epoch_start), compile=False, custom_objects=custom_objects)
            self.discriminator_ = load_model('{}{}.h5'.format(self.discriminator_path_template, self.epoch_start), compile=False, custom_objects=custom_objects)
            
            # Load the training history as well
            history_file = open(self.training_history_path,'rb')
            history_dict = pickle.load(history_file)
            history_file.close()
            self.d_loss_history = history_dict['d_loss_history']
            self.g_loss_history = history_dict['g_loss_history']
            self.d_acc_history = history_dict['d_acc_history']
        else:
            # If models don't exist then define them
            self.generator()
            self.discriminator()

        # set the attention getter ; needed for visualize_attention
        self.get_attention_activation = K.function([self.generator_.get_layer('ct_input').input,\
                                                    self.generator_.get_layer('roi_input').input,\
                                                    self.generator_.get_layer('possible_dose_input').input],\
                                                   [self.generator_.get_layer('attention_activation_1').output])
 
        # set getters ; useful for analysis
        self.get_cone_coordinates = K.function([self.generator_.get_layer('ct_input').input,\
                                                self.generator_.get_layer('roi_input').input,\
                                                self.generator_.get_layer('possible_dose_input').input],\
                                               [self.generator_.get_layer('normalize_cone_coordinates').output])

        self.get_cone_coefficients = K.function([self.generator_.get_layer('ct_input').input,\
                                                 self.generator_.get_layer('roi_input').input,\
                                                 self.generator_.get_layer('possible_dose_input').input],\
                                                [self.generator_.get_layer('reshape_cone_coefficients').output])

        # Build adversarial and discriminator optimization pielines from generator and discriminator        
        # on doit avoir déjà appelé generator() et discriminator()
        self.adversarial_model()
        self.discriminator_model()

    def train_discriminator_on_batch(self, batch_index, image_batch, epoch_number):
        """Loads a sample of data and uses it to train the model
        :param batch_index: The batch index
        :param epoch_number: The epoch
        """

        batch_size = int_shape(image_batch['dose'])[0]

        # Generate fake images
        images_fake = self.generator_.predict([image_batch['ct'], image_batch['structure_masks'], image_batch['possible_dose_mask']])

        labels_true = np.ones((batch_size, 1))
        labels_false = np.zeros((batch_size, 1))
        discriminator_loss_1 = self.discriminator_model_.train_on_batch([image_batch['dose'], image_batch['ct'], image_batch['structure_masks']], labels_true)
        apply_spectral_normalization(self.discriminator_)
        discriminator_loss_2 = self.discriminator_model_.train_on_batch([images_fake, image_batch['ct'], image_batch['structure_masks']], labels_false)
        apply_spectral_normalization(self.discriminator_)
        discriminator_loss = [0.5*(dl1+dl2) for dl1, dl2 in zip(discriminator_loss_1, discriminator_loss_2)]
        

        print('Discriminator loss : {:.3f}'.format(discriminator_loss[0]))
        print('Discriminator accuracy : {:.3f}'.format(discriminator_loss[1]))

        return discriminator_loss

    def train_generator_on_batch(self, batch_index, image_batch, epoch_number):
        """Loads a sample of data and uses it to train the model
        :param batch_index: The batch index
        :param epoch_number: The epoch
        """

        batch_size = int_shape(image_batch['dose'])[0]

        # Generate outputs
        desired_scores = np.ones([batch_size, 1])

        # Train the generator model with the batch
        adversarial_loss = self.adversarial_model_.train_on_batch([image_batch['ct'], image_batch['structure_masks'], image_batch['possible_dose_mask']], [image_batch['dose'], desired_scores])
        apply_spectral_normalization(self.generator_)

        print('Adversarial loss : {:.3f}'.format(adversarial_loss[2]))
        print('Mean absolute error : {:.3f}'.format(adversarial_loss[1]))
        print('Total loss : {:.3f}'.format(adversarial_loss[0]))

        # retourne la loss moyenne du batch
        return np.array(adversarial_loss)

    def train_generator_alone_on_batch(self, batch_index, image_batch, epoch_number):
        """Loads a sample of data and uses it to train the model
        :param batch_index: The batch index
        :param epoch_number: The epoch
        """

        batch_size = int_shape(image_batch['dose'])[0]

        # Generate outputs
        # Train the generator model with the batch
        generator_loss = self.generator_.train_on_batch([image_batch['ct'], image_batch['structure_masks'], image_batch['possible_dose_mask']], image_batch['dose'])
        apply_spectral_normalization(self.generator_)

        print('Mean absolute error : {:.3f}'.format(generator_loss))

        # retourne la loss moyenne du batch
        return generator_loss

    def predict_dose(self, epoch=1):
        """Predicts the dose for the given epoch number, this will only work if the batch size of the data loader
        is set to 1.
        :param epoch: The epoch that should be loaded to make predictions
        """
        # Define new models, or load most recent model if model already exists
        self.generator_ = load_model('{}{}.h5'.format(self.generator_path_template, epoch),compile=False, custom_objects=custom_objects)
        os.makedirs(self.prediction_dir, exist_ok=True)
        # Use generator to predict dose
        number_of_batches = self.data_loader.number_of_batches()

        print('Predicting dose')
        for idx in range(number_of_batches):
            print("Image {} out of {}.".format(idx, number_of_batches))
            image_batch = self.data_loader.get_batch(idx)

            # Get patient ID and make a prediction
            pat_id = image_batch['patient_list'][0]
            dose_pred_gy = self.generator().predict([image_batch['ct'], image_batch['structure_masks'], image_batch['possible_dose_mask']])[:][0]
            dose_pred_gy = dose_pred_gy * image_batch['possible_dose_mask'] * 100
            print(np.amax(dose_pred_gy))
            # Prepare the dose to save
            dose_pred_gy = np.squeeze(dose_pred_gy)
            dose_to_save = sparse_vector_function(dose_pred_gy)
            dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                                   columns=['data'])
            dose_df.to_csv('{}/{}.csv'.format(self.prediction_dir, pat_id))

        print("Done.")
        
    def plot_history(self):
        arrayed_g_loss_history = np.array(self.g_loss_history)
        N = range(len(self.d_loss_history))
        
        plt.plot(N, self.d_loss_history, '+-')
        plt.title('Discriminator loss')
        plt.show()

        plt.plot(N, self.d_acc_history, '+-')
        plt.title('Discriminator accuracy')
        plt.ylim(-0.05, 1.05)
        plt.plot([0,max(N)], [0.5, 0.5], 'r-')
        plt.show()

        arrayed_g_loss_history[:,1] = self.l1_lambda*arrayed_g_loss_history[:,1]

        arrayed_g_loss_history[:,0] = arrayed_g_loss_history[:,0] - arrayed_g_loss_history[:,1] - arrayed_g_loss_history[:,2]
        arrayed_g_loss_history_backup  = arrayed_g_loss_history.copy()
        arrayed_g_loss_history[:,1] = arrayed_g_loss_history[:,0] + arrayed_g_loss_history[:,1]
        arrayed_g_loss_history[:,2] = arrayed_g_loss_history[:,1] + arrayed_g_loss_history[:,2]
        
        plt.fill_between(N, 0, arrayed_g_loss_history[:,0], facecolor='blue', where=np.full_like(N,True))
        plt.fill_between(N, arrayed_g_loss_history[:,0], arrayed_g_loss_history[:,1], facecolor='red')
        plt.fill_between(N, arrayed_g_loss_history[:,1], arrayed_g_loss_history[:,2], facecolor='green')


        blue_patch = mpatches.Patch(color='blue', label='regularization')
        red_patch = mpatches.Patch(color='red', label='L1 loss')
        green_patch = mpatches.Patch(color='green', label='adversarial loss')
        plt.legend(handles=[blue_patch, red_patch, green_patch], loc='lower left')

        plt.title("Generator loss")

        plt.show()

        arrayed_g_loss_history = arrayed_g_loss_history_backup
        arrayed_g_loss_history = arrayed_g_loss_history/arrayed_g_loss_history.sum(axis=-1, keepdims=True)
        arrayed_g_loss_history[:,1] = arrayed_g_loss_history[:,0] + arrayed_g_loss_history[:,1]
        arrayed_g_loss_history[:,2] = arrayed_g_loss_history[:,1] + arrayed_g_loss_history[:,2]
        
        plt.fill_between(N, 0, arrayed_g_loss_history[:,0], facecolor='blue', where=np.full_like(N,True))
        plt.fill_between(N, arrayed_g_loss_history[:,0], arrayed_g_loss_history[:,1], facecolor='red')
        plt.fill_between(N, arrayed_g_loss_history[:,1], arrayed_g_loss_history[:,2], facecolor='green')


        blue_patch = mpatches.Patch(color='blue', label='regularization')
        red_patch = mpatches.Patch(color='red', label='L1 loss')
        green_patch = mpatches.Patch(color='green', label='adversarial loss')
        plt.legend(handles=[blue_patch, red_patch, green_patch], loc='lower left')

        plt.title("Composition of the loss")

        plt.show()

    def plot_predictions(self, num_to_plot):
        num_batches = self.data_loader.number_of_batches()
        batch_size = self.data_loader.batch_size
        batches_to_plot = np.random.choice(num_batches, num_to_plot, replace=True)
        slicing_axis = np.random.choice(3, (num_to_plot,batch_size))
        slice_to_look_at = np.random.randint(40, 89, (num_batches, batch_size))

        for i, batch_idx in enumerate(batches_to_plot):
            image_batch = self.data_loader.get_batch(batch_idx)
            prediction = self.generator_.predict([image_batch['ct'], image_batch['structure_masks'], image_batch['possible_dose_mask']])
            for image_idx in range(batch_size):
                k = slice_to_look_at[i, image_idx]
                if slicing_axis[i, image_idx] == 0:
                    prediction_sliced = prediction[image_idx,k,:,:,0]
                    dose_sliced = image_batch['dose'][image_idx,k,:,:,0]
                elif slicing_axis[i, image_idx] == 1:
                    prediction_sliced = prediction[image_idx,:,k,:,0]
                    dose_sliced = image_batch['dose'][image_idx,:,k,:,0]
                else:
                    prediction_sliced = prediction[image_idx,:,:,k,0]
                    dose_sliced = image_batch['dose'][image_idx,:,:,k,0]

                f, ax = plt.subplots(1,3, figsize=(20, 60))

                ax[0].imshow(prediction_sliced)
                ax[0].get_xaxis().set_visible(False)
                ax[0].get_yaxis().set_visible(False)
                ax[0].set_title("Predicted dose")

                ax[1].imshow(dose_sliced)
                ax[1].get_xaxis().set_visible(False)
                ax[1].get_yaxis().set_visible(False)
                ax[1].set_title("Target dose")

                ax[2].imshow(prediction_sliced-dose_sliced)
                ax[2].get_xaxis().set_visible(False)
                ax[2].get_yaxis().set_visible(False)
                ax[2].set_title("Difference")
                plt.show()
                
    def get_sigma(self):
        weights = []
        for layer in self.generator_.layers:
            if "attention_multiplier_" in layer.name:
                weights.append(layer.get_weights())

        for layer in self.discriminator_.layers:
            if "attention_multiplier_" in layer.name:
                weights.append(layer.get_weights())

        return weights
    
    def visualize_attention(self, image_batch):
        
        activations = self.get_attention_activation([image_batch['ct'], image_batch['structure_masks'], image_batch['possible_dose_mask']])
        dim = int(np.rint(len(activations[0][0][0])**(1/3)))
        upsampling_factor = 128//dim
        ct_scan = image_batch['ct'][:,:,:,:,0]
        
        plt.figure()
        for j in range(len(activations)):
            center_pix = dim*dim*(dim//2)+dim*(dim//2)+(dim//2)
            attention = activations[j][0][center_pix].reshape((dim,dim,dim))
            attention = zoom(attention, upsampling_factor)
            f, ax = plt.subplots(1,3, figsize=(20, 60))
            ax[0].imshow(ct_scan[j,64,:,:], cmap='Greys')
            ax[0].imshow(attention[64,:,:], cmap='gray', alpha=0.9)
            ax[0].get_xaxis().set_visible(False)
            ax[0].get_yaxis().set_visible(False)
            
            ax[1].imshow(ct_scan[j,:,64,:], cmap='Greys')
            ax[1].imshow(attention[:,64,:], cmap='gray', alpha=0.9)
            ax[1].get_xaxis().set_visible(False)
            ax[1].get_yaxis().set_visible(False)
            
            ax[2].imshow(ct_scan[j,:,:,64], cmap='Greys')
            ax[2].imshow(attention[:,:,64], cmap='gray', alpha=0.9)
            ax[2].get_xaxis().set_visible(False)
            ax[2].get_yaxis().set_visible(False)        
            plt.show()