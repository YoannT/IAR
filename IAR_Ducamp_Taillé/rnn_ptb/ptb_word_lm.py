#!/usr/bin/python
# -*-coding:Utf-8 -*

"""
Pour lancer une exécution sur Penn Treebank (ptb):

python ptb_word_lm.py --data_path=simple-examples/ptb/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn.ptb import reader

from scrnn import SCRNNCell


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
  "model", "small",
  "Taille du modele: smaller, small, medium ou large, voir les paramètres de chacun définis en dessous"
)
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string(
  "cell", "mikolov",
  "Type de neurone: simple, lstm, gru ou mikolov(scrn)"
)

FLAGS = flags.FLAGS

"""
Hyper-paramètres du modèle:
- init_scale - échelle initiale des poids
- learning_rate - valeur initiale de la vitesse d'apprentissage
- max_grad_norm - norme du gradient maximale
- num_layers - nombre de couches pour LSTM
- num_steps - nombre de cycles
- hidden_size - taille de la couche cachée
- max_epoch - nombre de periodes sur lesquelles est appliqué la vitesse initiale
- max_max_epoch - nombre total de périodes pendant l'apprentissage
- keep_prob - probabilité d'oublier des neurones
- lr_decay - coefficient de diminution de la vitesse d'apprentissage après max_epoch périodes
- batch_size - taille des échantillons
"""
class SmallerConfig(object):
  """Petit réseau"""
  init_scale = 0.05
  learning_rate = 0.25
  max_grad_norm = 5
  num_layers = 1
  num_steps = 50
  hidden_size = 40
  context_size = 10
  alpha = 0.95
  max_epoch = 6
  max_max_epoch = 20
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000

class SmallConfig(object):
  """Petit réseau"""
  init_scale = 0.05
  learning_rate = 0.25
  max_grad_norm = 5
  num_layers = 1
  num_steps = 50
  hidden_size = 90
  context_size = 10
  alpha = 0.95
  max_epoch = 6
  max_max_epoch = 20
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000

class MediumConfig(object):
  """Réseau moyen"""
  init_scale = 0.05
  learning_rate = 0.25
  max_grad_norm = 5
  num_layers = 1
  num_steps = 50
  hidden_size = 100
  context_size = 40
  alpha = 0.95
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000

class LargeConfig(object):
  """Grand réseau"""
  init_scale = 0.05
  learning_rate = 0.25
  max_grad_norm = 5
  num_layers = 1
  num_steps = 50
  hidden_size = 300
  context_size = 40
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 32
  vocab_size = 10000
  alpha = 0.95

class TestConfig(object):
  """Réseau de test"""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  context_size = 80
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  alpha = 0.95


class PTBModel(object):
  """Modèle de traitement de PTB"""
  """
  Itinialise un modèle avec les paramètres spécifiés
  """
  def __init__(self, is_training, config, cell_type):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    
    # Input et targets modifiés pendant le run
    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Type de neurone
    if cell_type == 'simple':
      rnn_cell = tf.nn.rnn_cell.BasicRNNCell(size)
    elif cell_type == 'gru':
      rnn_cell = tf.nn.rnn_cell.GRUCell(size)
    elif cell_type == 'lstm':
      rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    elif cell_type == 'mikolov':
      rnn_cell = SCRNNCell(batch_size, size, config.hidden_size, config.context_size, config.alpha)
    else:
      raise Exception("Type de neurone invalide: {}".format(cell_type))

    if is_training and config.keep_prob < 1:
      rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
          rnn_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output_size = rnn_cell.output_size

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    
    # Calcul de la perplexité
    output = tf.reshape(tf.concat(1, outputs), [-1, output_size])
    softmax_w = tf.get_variable("softmax_w", [output_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return
    
    # Descente du gradient
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grad_values = tf.gradients(cost, tvars)
    grads, _ = tf.clip_by_global_norm(grad_values, config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

def run_epoch(session, m, data, eval_op, verbose=False):
  """Lancement du modèle"""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()

  # Sélection d'un batch de données d'apprentissage et de validation
  # Les données de validation sont les mots suivant ceux d'apprentissage
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexité: %.3f vitesse de traitement: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  elif FLAGS.model == "smaller":
    return SmallerConfig()
  else:
    raise ValueError("Taille du modèle invalide: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Paramétrer --datapath à l'emplacement du dossier de test")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  start_time = time.time()

  print("Type de neurones: {}".format(FLAGS.cell))

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config, cell_type=FLAGS.cell)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config, cell_type=FLAGS.cell)
      mtest = PTBModel(is_training=False, config=eval_config, cell_type=FLAGS.cell)

    tf.initialize_all_variables().run()

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Période: %d Vitesse d'apprentissage: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)

      print("Période: %d Perplexité sur données d'apprentissage: %.3f" % (i + 1, train_perplexity))
      
      # Test si la perplexité de validation a augmenté ou stagné, si oui on divise le lr par 1.5
      # Non utilisé mais correspond mieux à l'article
      
      '''
      valid_old = np.inf
      if i > 0:
        valid_old = valid_perplexity
      '''
      
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Période: %d Perplexité sur données de validation: %.3f" % (i + 1, valid_perplexity))
      
      '''
      if i > 0 and valid_old <= valid_perplexity:
        print('Diminution du LR: %.4f' % m.lr)
        m.lr_decay /=1.5
      '''
      
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Perplexité sur données de test: %.3f" % test_perplexity)
    print("Temps d'exécution: %.3f" % (time.time()-start_time))

if __name__ == "__main__":
  tf.app.run()
