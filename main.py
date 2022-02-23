import datetime
import os.path as op
import glob
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# EEGNet-specific imports
from tensorflow.keras.callbacks import TensorBoard
from keras.utils import np_utils
from EEGNet_updated import EEGNet

# tools for plotting confusion matrices
from matplotlib import pyplot as plt


def create_import_list(filepath, file_ext):
    print(f"Creating a list for {file_ext} files in directory {filepath}.")
    fullpath = op.join(filepath, file_ext)
    List = glob.glob(fullpath)

    return List


def eeg_import(filename, data_format, epoched=False, filepath=None):
    if filepath is None:
        fullname = filename
    else:
        fullname = filepath + filename

    if data_format == "bva":
        if epoched:
            raise ValueError(
                "Brainvision Analyzer data should be imported as continuous data."
            )
        else:
            raw = mne.io.read_raw_brainvision(fullname, preload=True, verbose=False)
    elif data_format == "eeglab":
        if epoched:
            # raw = mne.io.read_epochs_eeglab(fullname, eog='auto', verbose=False)
            raise Warning(
                "Please load in un-epoched data. MNE-Python seems to have trouble parsing events "
                "from already epoched data."
            )
        else:
            raw = mne.io.read_raw_eeglab(
                fullname, eog="auto", preload=True, verbose=False
            )
    else:
        raise ValueError(
            "MNE allows either Brainvision .vhdr or EEGLAB .set read-in of data."
        )

    return raw


def eeg_preprocessing(data, epoch_limits, event_id, eog=True):
    events_from_annot, event_dict = mne.events_from_annotations(raw, event_id=event_id)

    if not eog:
        # picks only EEG channels.
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
    else:
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=True)

    epoched_data = mne.Epochs(
        data,
        events_from_annot,
        event_id,
        epoch_limits[0],
        epoch_limits[1],
        proj=False,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False,
    )
    labels = epoched_data.events[:, -1]

    return epoched_data, labels


if __name__ == "__main__":
    # set paths for tensorboard logs
    log_dir = (
        "/Volumes/diskAshur2/Deep_Learning_Nigel/logs/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # set random seed for replicability of data selection
    np.random.seed(1028)

    # set event_id for specific markers we want to extract
    event_id = dict(Misophonia=1, OCD=2)

    # set epoch limits for the duration for each epoch. Currently trying 20s epochs.
    epoch_limits = (0, 20)

    # Setup for reading the raw data, data is stored in one huge file. Can also do raw.append() in for loop.
    # Appending is fast in MATLAB, unsure in Python.
    data_list = create_import_list(
        "/Volumes/diskAshur2/Deep_Learning_Nigel/EEGClassifier/EEGLAB/Processed_Data/Train_10sEpochs/",
        "*10s*.set",
    )

    for count, subject in enumerate(data_list):
        print(f"Currently importing subject {count+1}.")

        # MNE import functions somehow don't function when loading individual files and concatenating them later.
        # Do these steps in MATLAB first or figure this out. Currently using one huge dataset.
        if count == 0:
            raw = eeg_import(filename=subject, data_format="eeglab", epoched=False)
        else:
            raw_new = eeg_import(filename=subject, data_format="eeglab", epoched=False)
            raw.append([raw_new])

    epoched_data, events = eeg_preprocessing(
        raw, epoch_limits=epoch_limits, event_id=event_id, eog=True
    )

    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    X = epoched_data.get_data() * 1000  # format is in (trials, channels, samples)
    y = events

    # set amount of kernels, channels and samples per epoch.
    kernels, chans, samples = (
        1,
        66,
        128 * 20 + 1,
    )  # Sampling rate is 128 Hz, epoch length is 20s.

    # take 70/15/15 percent of the data to train/validate/test.
    # Randomize later for proper conduction of the model.
    # Need to somehow shuffle the data. Either in MATLAB or Python.
    # And make sure that epochs do not overlap in training, val, test test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    ######################## EEGNET PORTION #######################
    # convert labels to one-hot encodings for later classification of each epoch
    y_train = np_utils.to_categorical(y_train - 1, num_classes=2)
    y_test = np_utils.to_categorical(y_test - 1, num_classes=2)

    # convert data to NCHW (trials, kernels, channels, samples) format. Data
    # contains 66 channels and 128*20+1 time-points. Set the number of kernels to 1.
    X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples)
    X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)

    print("X_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(
        n_classes=2,
        Chans=chans,
        Samples=samples,
        lr=0.0001,
        kernLength=32,
        dropoutRate=0.5,
        dropoutType="SpatialDropout2D",
    )

    # count number of parameters in the model
    numParams = model.count_params()

    # set a valid path for your system to record model checkpoints
    # checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
    #                               save_best_only=True)

    # use class_weights = class_weights = {0:1, 1:1, 2:1, 3:1} if significantly fewer epochs in one category.
    fittedModel = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=300,
        verbose=2,
        validation_split=0.15,
        callbacks=[tensorboard_callback],
    )

    # calculate predictions for test data for plotting a confusion matrix
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    acc = np.mean(preds == y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % acc)

    names = ["Misophonia", "OCD"]
    plt.figure(0)
    confusion_matrix(preds, y_test.argmax(axis=-1), names, title="EEGNet-8,2")
