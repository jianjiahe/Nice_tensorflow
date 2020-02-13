import logging
import os
import sys
import numpy as np
import tensorflow as tf
from time import time
from tensorflow.python.client import timeline

from tensorflow.examples.tutorials.mnist import input_data


from utils.basis_util import basic_config, check_param_num, check_restore_params, StandardLogistic
from utils.batch import MiniBatch


from models.nice import Nice
from config import TrainBasic, DataBasic, ModelBasic


flags = tf.flags

flags.DEFINE_boolean('trace_time', False, 'whether trace time or not')
flags.DEFINE_float('l2', 1e-6, 'L2 regularization factor')
flags.DEFINE_string('optimizer', TrainBasic.optimizer, 'which optimizer to use {momentum/adam}')
flags.DEFINE_float('momentum', 0.99, 'momentum factor for the SGD')
flags.DEFINE_float('lr_start', 0.0005, 'learning rate decay starts with this value')
flags.DEFINE_float('lr_end', 0.00001, 'learning rate decay ends up with this value')
flags.DEFINE_float('lr_const', 0.00001, 'learning rate keeps this value at the last 5 epoch')
flags.DEFINE_integer('warming_steps', 10000, 'step number for warming learning rate')
flags.DEFINE_integer('summary_stride', 50, 'write summary every this epoch')
flags.DEFINE_integer('corpus_size', 55000, 'the number of all items in the train corpus')  # 107104(8w, 7144,19960)
flags.DEFINE_integer('batch_size', TrainBasic.batch_size, 'batch_size')
flags.DEFINE_integer('epoch_num', 2000, 'epoch number')
flags.DEFINE_string('corpus_name', TrainBasic.dataset, 'corpus name')
# flags.DEFINE_integer('mel_filters', AudioBasic.num_mels, 'the number of mel-filters')
# flags.DEFINE_integer('n_fft', AudioBasic.n_fft, 'the number of fft')
flags.DEFINE_string('run_name',
                    # 'th30-adam0.001-l20.001-rec_loss1e-5-not_accumulate-norm-log-mel_center',
                    TrainBasic.runname,
                    'run name for this train')

FLAGS = flags.FLAGS

class Trainer:
    def __init__(self):
        self.l2 = FLAGS.l2
        self.optimizer_name = FLAGS.optimizer
        self.lr_start = FLAGS.lr_start
        self.lr_end = FLAGS.lr_end
        self.lr_const = FLAGS.lr_const
        self.warming_steps = FLAGS.warming_steps
        self.summary_stride = FLAGS.summary_stride
        self.corpus_name = FLAGS.corpus_name
        self.corpus_size = FLAGS.corpus_size
        self.epoch_num = FLAGS.epoch_num
        self.run_name = FLAGS.run_name
        self.batch_size = FLAGS.batch_size
        self.epoch_size = self.corpus_size // self.batch_size
        self._training = True

        self.mini_batch = MiniBatch(batch_size=self.batch_size, training=self._training)

        self.loss, self.regularization_loss = None, None

        self.has_built = False
        self.describe()

    def describe(self):
        logging.info('epoch size: {0}'.format(self.epoch_size))
        logging.info('epoch num: {0}'.format(self.epoch_num))
        logging.info('batch size: {0}'.format(self.batch_size))
        # logging.info('n_fft: {0}'.format(self.n_fft))
        # logging.info('mel-filters: {0}'.format(self.mel_filters))

    @staticmethod
    def check_param():
        check_param_num()

    def _config_step(self):
        with tf.variable_scope('Step'):
            self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
            self.global_epoch = self.global_step // self.epoch_size + 1

    def const_lr(self):
        return tf.convert_to_tensor(self.lr_const, dtype=tf.float32)

    def decay_lr(self):
        # 学习率多项式方式衰减，cycle参数是决定lr是否在下降后重新上升
        decayed = lambda: tf.train.polynomial_decay(self.lr_start, global_step=self.global_epoch - 1,
                                                    decay_steps=self.epoch_num, end_learning_rate=self.lr_end, power=2,
                                                    cycle=False)

        return tf.cond(tf.less(self.global_epoch, self.epoch_num - 5), decayed, self.const_lr, name='decay_lr')

    def warming_lr(self):
        return tf.train.polynomial_decay(0., global_step=self.global_step, decay_steps=self.warming_steps,
                                         end_learning_rate=self.lr_start, power=1, name='warming_lr')

    def _config_lr(self):
        with tf.variable_scope('LearningRate'):
            self.lr = tf.cond(tf.less(self.global_step, self.warming_steps), self.warming_lr, self.decay_lr)
            # self.lr = self.const_lr()
            tf.summary.scalar('lr', self.lr)

    def _gen(self):
        while True:
            yield self.mini_batch.next_batch()

    def _get_input_iterator(self):
        inputs_shape = tf.TensorShape([None, 784])

        train_data_set = tf.data.Dataset.from_generator(
            self._gen,
            (tf.float32),
            (inputs_shape)
        ).prefetch(buffer_size=2)

        self.train_iterator = train_data_set.make_initializable_iterator()
        return self.train_iterator.get_next()

    def _create_loss(self):
        with tf.variable_scope('Loss'):
            self.loss = -tf.reduce_mean(self.log_prob)

            if self._training:
                tf.summary.scalar('loss_train', self.loss, collections=['loss'])
            else:
                tf.summary.scalar('loss_validation', self.loss, collections=['loss'])

            self.regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                                 if not ('bias' in v.name or 'Bias' in v.name)]) * self.l2
            tf.summary.scalar('l2reg_loss', self.regularization_loss, collections=['loss'])
            self.loss = self.loss + self.regularization_loss

    def _create_summary(self):
        with tf.variable_scope('summaries'):
            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
            self.loss_summary = tf.summary.merge_all(key='loss')
            self.writer = tf.summary.FileWriter(os.path.join(TrainBasic.checkpoint_dir, self.corpus_name, self.run_name))

    def _config_optimizer(self):
        if self.optimizer_name == TrainBasic.Momentum:
            return tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=FLAGS.momentum, use_nesterov=True)
        elif self.optimizer_name == TrainBasic.Adam:
            return tf.train.AdamOptimizer(learning_rate=self.lr)

    def _create_optimizer(self):
        with tf.variable_scope('optimizer'):
            optimizer = self._config_optimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            print('len(gradients):', len(gradients))
            with tf.variable_scope('grad_norms'):
                grad_norms = []
                for index, grad in enumerate(gradients):
                    grad_norms.append(tf.norm(grad))
                    # print(index, ' grad is: ', grad)
                    # if grad is not None:
                    #     grad_norms.append(tf.norm(grad))
                tf.summary.histogram('grad_norms', grad_norms)

            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            with tf.control_dependencies(update_ops):
                # self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step)
                self.optimizer = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                           global_step=self.global_step)


    def _build_graph(self):
        self._config_step()
        self._config_lr()
        with tf.variable_scope('input'):
            self.inputs = self._get_input_iterator()

        self.log_prob= Nice(prior=StandardLogistic(),
                            batch_size=self.batch_size,
                            couple_layers = ModelBasic.couple_layers,
                            in_out_dim = ModelBasic.in_out_dim,
                            mid_dim = ModelBasic.mid_dim,
                            hidden_dim=ModelBasic.hidden_dim,
                            mask_config=ModelBasic.mask_config,
                            training=True).infer(self.inputs)

        self._create_loss()
        self._create_optimizer()
        self._create_summary()
        self.check_param()

        self.has_built = True

    def _train_init(self, sess: tf.Session):
        self.saver = tf.train.Saver()
        self.writer.add_graph(sess.graph)
        sess.run(self.train_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        check_restore_params(self.saver, sess, self.run_name, corpus_name=self.corpus_name)

    def train_epoch(self):
        assert not self.has_built
        tf.reset_default_graph()
        logging.info('Building graph...')
        self._build_graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            self._train_init(sess)
            logging.info('Train start.')

            # if time line: options run_metadata
            options = None
            run_metadata = None
            if FLAGS.trace_time:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            while True:
                try:
                    start_time = time()
                    if FLAGS.trace_time:
                        loss, step, epoch, ms, ls, _ = sess.run(
                            [self.loss, self.global_step, self.global_epoch,
                             self.merged_summary, self.loss_summary, self.optimizer],
                            options=options, run_metadata=run_metadata
                        )
                        fetched_time_line = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_time_line.generate_chrome_trace_format()
                        if not os.path.exists('timeline/'):
                            os.makedirs('timeline')
                        with open('timeline/time_line_step_%d.json' % step, 'w') as f:
                            f.write(chrome_trace)

                    else:
                        loss, step, epoch, ms, ls, _ = sess.run(
                            [self.loss, self.global_step, self.global_epoch,
                             self.merged_summary, self.loss_summary, self.optimizer]
                        )

                    self.writer.add_summary(ls, global_step=step)
                    if step % self.summary_stride == 0:
                        self.writer.add_summary(ms, global_step=step)

                    if step % self.epoch_size == 0:
                        bit_per_dim = (loss + np.log(256.) * ModelBasic.in_out_dim) \
                                      / (ModelBasic.in_out_dim * np.log(2.))
                        logging.info('Epoch {0} step {1}, Processed in {2:.2f}s, loss: {3}, bit_per_dim: {4}'.format(epoch,
                                                                                                     step % self.epoch_size,
                                                                                                     time() - start_time,
                                                                                                     loss,
                                                                                                     bit_per_dim))

                    if step % self.epoch_size == 0 and epoch % 50 == 0:
                        # if step % 1 == 0 and epoch % 1 == 0:
                        # # validation
                        # self._training = False
                        # loss_val, step, epoch, ms, ls = sess.run([self.loss, self.global_step, self.global_epoch, self.merged_summary, self.loss_summary])
                        # self._training = True
                        # bit_per_dim = (loss_val + np.log(256.) * ModelBasic.in_out_dim) \
                        #               / (ModelBasic.in_out_dim * np.log(2.))
                        # logging.info(
                        #     'Val: loss: {0}, bit_per_dim: {1}'.format(loss_val, bit_per_dim))

                        self._save_checkpoint(sess)
                        self._train_finalize()
                        return step, epoch

                except KeyboardInterrupt:
                    logging.info('You terminate the program manually.')
                    self._save_checkpoint(sess)
                    sys.exit(0)

    def _save_checkpoint(self, sess: tf.Session):
        logging.info('Saving checkpoints...')
        self.saver.save(sess, os.path.join(TrainBasic.checkpoint_dir, self.corpus_name, self.run_name, 'ckp'),
                        global_step=self.global_step)
        logging.info('Checkpoint saves.')

    def _train_finalize(self):
        tf.get_default_graph().finalize()
        self.has_built = False

    def evaluate(self, epoch, step):
        pass
        # validition_mini_batch = MiniBatch(corpus_name=self.corpus_name, datacsv_name=DatadirBasic.validitiondata_csv, batch_size=self.batch_size)
        #
        # infer = Infer(corpus_name=self.corpus_name, run_name=self.run_name,
        #               mel_filters=self.mel_filters, n_fft=self.n_fft)
        #
        # # rest = list(set(range(1, 101)))
        # # infer.synthesis_by_pinyinlist(rest)
        #
        # infer.synthesis('绿是阳春烟景,大块文章的底色, 四月的林峦,更是绿的鲜活,秀嫩.',
        #                 wav_path=os.path.join(DatadirBasic.eval_dir, '{0}/epoch{1:0>3d}-step{2:0>6d}.wav'.format(self.run_name, epoch, step)))


    def main(self):
        while True:
            step, epoch = self.train_epoch()
            logging.info('Epoch: {0}. to do [evaluation]'.format(epoch))

            self.evaluate(epoch, step)
            logging.info('Epoch {0} complete.'.format(epoch))
            if epoch > self.epoch_num:
                logging.info('Train complete')
                break


if __name__ == '__main__':
    basic_config()
    trainer = Trainer()
    trainer.main()