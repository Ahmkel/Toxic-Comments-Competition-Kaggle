from base.base_model import BaseModel
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Concatenate
from tensorflow.python.keras.callbacks import ModelCheckpoint


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()
        this.callbacks = []

    def n_grams_channel(inputs, n):
        channel = Conv2D(1, kernel_size=(n, embeddings_length), activation='relu')(inputs)
        channel_mp = MaxPool2D(pool_size=(channel.shape[1], 1))(channel)
        channel_final = Flatten()(channel_mp)        
        return channel_final
        
    def build_model(self):
        self.inputs = Input(shape=(max_sequence_length, embeddings_length, 1))
        self.channel1_final = n_grams_channel(inputs, 2)
        self.channel2_final = n_grams_channel(inputs, 3)
        self.channel3_final = n_grams_channel(inputs, 4)
        self.channels_final = Concatenate()([channel1_final, channel2_final, channel3_final])
        self.predictions = Dense(1, 'sigmoid')(channel1_final)
        
        self.model = Model(inputs=inputs, outputs=predictions)
        
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
        )

    def init_saver(self):
        this.callbacks.push(
            ModelCheckpoint(
            filepath='weights-improvement-{epoch:02d}-{loss:.2f}.hdf5',
            monitor='loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=True,
        )
