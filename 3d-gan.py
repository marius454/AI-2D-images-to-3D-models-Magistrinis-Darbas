from numpy import dtype
import tensorflow as tf
import variables as var
import file_processing as fp

def create_3dgan_model():
    inputs = tf.keras.layers.Input((var.imageWidth, var.imageHeight, 3))
    generator = generator_model()
    discriminator = discriminator_model()

def generator_model(z, batch_size, weights, batchNorm=True, train=True):
    strides    = [1,2,2,2,2]

    z = tf.reshape(z, (batch_size, 1, 1, 1, var.z_size))
    g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,512), var.generator_strides, "VALID")
    if (batchNorm):
        g_1 = tf.nn.batch_normalization(g_1)
    g_1 = tf.nn.relu(g_1)

    g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,8,8,8,256), var.generator_strides, "SAME")
    if (batchNorm):
         g_2 = tf.nn.batch_normalization(g_2)
    g_2 = tf.nn.relu(g_2)

    g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,16,16,16,128), var.generator_strides, "SAME")
    if (batchNorm):
        g_3 = tf.nn.batch_normalization(g_3)
    g_3 = tf.nn.relu(g_3)

    g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,32,32,32,64), var.generator_strides, "SAME")
    if (batchNorm):
        g_4 = tf.nn.batch_normalization(g_4)
    g_4 = tf.nn.relu(g_4)
    
    g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size,64,64,64,1), var.generator_strides, "SAME")
    g_5 = tf.nn.sigmoid(g_5)
    # g_5 = tf.nn.tanh(g_5)

    print (g_1, 'g1')
    print (g_2, 'g2')
    print (g_3, 'g3')
    print (g_4, 'g4')
    print (g_5, 'g5')
    
    return g_5


def discriminator_model(input, batch_size, weights, batchNorm=True, train=True):
    strides = [2,2,2,2,1]

    d_1 = tf.nn.conv3d(input, weights['wd1'], var.discriminator_strides, "SAME")
    if (batchNorm):
        d_1 = tf.nn.batch_normalization(d_1)                              
    d_1 = tf.nn.leaky_relu(d_1, var.relu_leak_value)

    d_2 = tf.nn.conv3d(d_1, weights['wd2'], var.discriminator_strides, "SAME") 
    if (batchNorm):
        d_2 = tf.nn.batch_normalization(d_2)
    d_2 = tf.nn.leaky_relu(d_2, var.relu_leak_value)
    
    d_3 = tf.nn.conv3d(d_2, weights['wd3'], var.discriminator_strides, "SAME")  
    if (batchNorm):
        d_3 = tf.nn.batch_normalization(d_3)
    d_3 = tf.nn.leaky_relu(d_3, var.relu_leak_value) 

    d_4 = tf.nn.conv3d(d_3, weights['wd4'], var.discriminator_strides, "SAME")     
    if (batchNorm):
        d_4 = tf.nn.batch_normalization(d_4)
    d_4 = tf.nn.leaky_relu(d_4, var.relu_leak_value)

    d_5 = tf.nn.conv3d(d_4, weights['wd5'], var.discriminator_strides, "VALID")     
    # d_5_no_sigmoid = d_5
    d_5 = tf.nn.sigmoid(d_5)

    print (d_1, 'd1')
    print (d_2, 'd2')
    print (d_3, 'd3')
    print (d_4, 'd4')
    print (d_5, 'd5')

    return d_5#, d_5_no_sigmoid