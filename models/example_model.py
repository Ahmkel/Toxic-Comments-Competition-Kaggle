from base.base_model import BaseModel
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Concatenate, Reshape
from tensorflow.python.keras.callbacks import ModelCheckpoint


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()
        self.max_sequence_length = config.max_sequence_length
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.callbacks = []

    def get_embedding_matrix():
        # Todo: implement by gammal
        return None
        
    def embedding_layer():
        embedding_matrix = get_embedding_matrix()
        
        return Embedding(
            input_dim=self.vocab_size + 1,
            output_dim=self.embedding_dim,
            weights=[embedding_matrix],
            input_length=self.max_sequence_length,
            trainable=False
            )
        
    def n_grams_channel(inputs, n):
        channel = Conv2D(1, kernel_size=(n, self.embedding_dim), activation='relu')(inputs)
        channel_mp = MaxPool2D(pool_size=(channel.shape[1], 1))(channel)
        channel_final = Flatten()(channel_mp)        
        return channel_final
        
    def build_model(self):
        self.inputs = Input(shape=(max_sequence_length,))
        self.embedding = embedding_layer()(self.inputs)
        self.channel_inputs = Reshape(target_shape=(self.max_sequence_length, self.embeddings_dim, 1))(self.embedding)
        
        self.channel1_final = n_grams_channel(self.inputs, 2)
        self.channel2_final = n_grams_channel(self.inputs, 3)
        self.channel3_final = n_grams_channel(self.inputs, 4)
        self.channels_final = Concatenate()([self.channel1_final, self.channel2_final, self.channel3_final])
        self.predictions = Dense(1, 'sigmoid')(self.channels_final)
        
        self.model = Model(inputs=self.inputs, outputs=self.predictions)
        
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
        )

    def init_saver(self):
        self.callbacks.append(
            ModelCheckpoint(
            filepath='weights-improvement-{epoch:02d}-{loss:.2f}.hdf5',
            monitor='loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=True,
        )
