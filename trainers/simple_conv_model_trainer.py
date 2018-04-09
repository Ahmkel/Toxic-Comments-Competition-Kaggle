from base.base_train import BaseTrain
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import numpy as np


class SimpleConvModelTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        self.callbacks = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        super(SimpleConvModelTrainer, self).__init__(sess, model, data, config, logger)
        self.init_saver()

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
        )
    
    def train(self):
        history = model.fit(
            data.[0], data.[1],
            epochs = self.config.num_epochs,
            verbose = True,
            batch_size = self.config.batch_size,
            validation_split = self.config.validation_split
            callbacks = self.callbacks
        )
        self.loss = history.history['loss']
        self.accuracy = history.history['acc']
        self.val_loss = history.history['val_loss']
        self.val_acc = history.history['val_acc']

    def train_epoch(self):
        #Todo: ka7la
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        #Todo: ka7la
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
