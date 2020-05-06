import matplotlib.pyplot as plt
import keras


class FeatureLogger(keras.callbacks.Callback):
    def __init__(self, epochs_to_log):
        self.epochs_to_log = epochs_to_log
        super(FeatureLogger, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')

        if epoch not in self.epochs_to_log:
            return

        ixs = [0, 2]  # conv layer indices
        outputs = [self.model.layers[i].output for i in ixs]
        mdl = keras.models.Model(inputs=self.model.inputs, outputs=outputs)
        img = self.validation_data[0]
        feature_maps = mdl.predict(img)
        a = b = 4
        layer_number = iter(ixs)
        for fmap in feature_maps:
            fltr_number = 1
            for _ in range(a):
                for _ in range(b):
                    # specify subplot
                    ax = plt.subplot(a, b, fltr_number)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(fmap[0, :, :, fltr_number-1], cmap='gray')
                    plt.title("Filter #" + str(fltr_number))
                    fltr_number += 1
            # saving feature maps
            plt.subplots_adjust(hspace=0.4)
            plt.savefig("layer-" + str(next(layer_number)) + "_filters-(1-" + str(fltr_number-1) + ")_epoch-" +
                        str(epoch) + ".png")
            a = 2
