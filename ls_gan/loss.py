
import tensorflow as tf
from keras import backend as K


def dc_gan(disc_fake, disc_real):
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, \
                                        labels=tf.ones_like(disc_fake)))
    
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, \
                                        labels=tf.zeros_like(disc_fake)))
    
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, \
                                        labels=tf.ones_like(disc_real)))
    
    disc_cost /= 2.
    
    return gen_cost, disc_cost