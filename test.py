from tensorflow import keras

from utilities import csv_readnstack, csv_to_onehot
from training import scalings


def main():
    # Caricamento dei dati dal file csv e trasformazione in tensori
    test_dataset_values = csv_readnstack(["test_gesture_x.csv", "test_gesture_y.csv", "test_gesture_z.csv"])
    test_dataset_labels = csv_to_onehot("test_label.csv")

    # Scaling dei parametri
    datasets = [test_dataset_values]
    test_dataset_values = scalings(datasets)

    best_model = keras.models.load_model("bestModel.h5")

    test_loss, test_accuracy = best_model.evaluate(test_dataset_values, test_dataset_labels)
    print("\n\n -----------> loss su TEST: {} <----------".format(test_loss))
    print("\n\n -------> accuracy su TEST: {} <----------".format(test_accuracy))


if __name__ == '__main__':
    main()
