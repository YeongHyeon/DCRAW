import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

class DRAW(object):

    def __init__(self, height, width, channel=1, sequence_length=10, learning_rate=1e-3, attention=False):

        self.height, self.width, self.channel = height, width, channel
        self.sequence_length, self.learning_rate = sequence_length, learning_rate

        self.attention_n = 3
        # self.n_z = 10
        self.n_z = self.sequence_length
        self.share_parameters = False

        self.feature = [32, 64]

        self.c = [0] * self.sequence_length

        self.mu, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length

        self.x = tf.placeholder(tf.float32, [None, self.height*self.width]) # input (batch_size * img_size)
        self.x_img = tf.reshape(self.x, [-1, self.height, self.width, self.channel])

        self.enconv_1, self.enc1_h, self.enc1_w, self.enc1_c, self.enc1_hwc = self.conv2d(inputs=self.x_img, num_inputs=self.channel, num_outputs=self.feature[0], kernel_size=3, stride=2, padding='SAME', activation='sigmoid')
        self.enconv_2, self.enc2_h, self.enc2_w, self.enc2_c, self.enc2_hwc = self.conv2d(inputs=self.enconv_1, num_inputs=self.feature[0], num_outputs=self.feature[1], kernel_size=3, stride=2, padding='SAME', activation='sigmoid')

        self.c_shape = self.enc2_hwc
        self.c_h, self.c_w, self.c_c = self.enc2_h, self.enc2_w, self.enc2_c

        self.conv_enc_flat = tf.reshape(self.enconv_2, [-1, self.c_h*self.c_w])

        self.z_noise = tf.random_normal((tf.shape(self.x)[0]*self.c_c, self.n_z), mean=0, stddev=1) # Qsampler noise

        self.n_hidden = 256
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(num_units=self.n_hidden, state_is_tuple=True) # encoder Op
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(num_units=self.n_hidden, state_is_tuple=True) # decoder Op

        self.h_prev_dec = tf.zeros((tf.shape(self.x)[0]*self.c_c, self.n_hidden))
        self.h_prev_enc = self.lstm_enc.zero_state(tf.shape(self.x)[0]*self.c_c, tf.float32)
        self.h_dec_state = self.lstm_dec.zero_state(tf.shape(self.x)[0]*self.c_c, tf.float32)

        for t in range(self.sequence_length):

            # Equation 3.
            # x_t_hat = x = sigmoid(c_(t-1))
            if(t==0): c_prev = tf.zeros((tf.shape(self.x)[0]*self.c_c, self.c_h*self.c_w))
            else: c_prev = self.c[t-1]
            x_t_hat = self.conv_enc_flat - tf.nn.sigmoid(c_prev)

            # Equation 4.
            # r_t = read(x_t, x_t_hat)
            if(attention): r_t = self.attention_read(self.conv_enc_flat, x_t_hat, self.h_prev_dec)
            else: r_t = self.basic_read(self.conv_enc_flat, x_t_hat)

            # Equation 5.
            # h_t_enc = RNN_encoder(self.h_prev_enc, [r_t, self.h_prev_dec])
            self.mu[t], self.sigma[t], h_t_enc, self.h_prev_enc = self.encode(self.h_prev_enc, tf.concat([r_t, self.h_prev_dec], 1))

            # Equation 6.
            # z_t
            z_t = self.sample_latent(self.mu[t], self.sigma[t])

            # Equation 7.
            # h_t_dec = RNN_decoder(self.h_prev_dec, z_t)
            h_t_dec, self.h_dec_state = self.decode(self.h_dec_state, z_t)

            # Equation 8.
            if(attention): self.c[t] = c_prev + self.attention_write(h_t_dec)
            else: self.c[t] = c_prev + self.basic_write(h_t_dec)

            # Replace self.h_prev_dec as h_t_dec
            self.h_prev_dec = h_t_dec

            self.share_parameters = True

        self.conv_dec_img = tf.reshape(self.c[t], [-1, self.c_h, self.c_w, self.c_c])

        self.deconv_1 = self.conv2d_transpose(inputs=self.conv_dec_img, num_inputs=self.feature[1], num_outputs=self.feature[0], output_shape=tf.shape(self.enconv_1), kernel_size=3, stride=2, padding='SAME', activation='sigmoid')
        self.deconv_2 = self.conv2d_transpose(inputs=self.deconv_1, num_inputs=self.feature[0], num_outputs=self.channel, output_shape=tf.shape(self.x_img), kernel_size=3, stride=2, padding='SAME', activation=None)

        print("Input", self.x_img.shape)
        print("Conv 1", self.enconv_1.shape)
        print("Conv 2", self.enconv_2.shape)
        print("Flat", self.conv_enc_flat.shape)

        print("Read", r_t.shape)
        print("Encode", h_t_enc.shape)
        print("Latent Space", z_t.shape)
        print("Decode", h_t_dec.shape)
        print("Write", self.c[t].shape)

        print("Reshape", self.conv_dec_img.shape)
        print("Conv_Tr 1", self.deconv_1.shape)
        print("Conv_Tr 2", self.deconv_2.shape)

        # Equation 9.
        # Reconstruction error: Negative log probability.
        # L^x = -log D(x|c_T)
        # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        # self.loss_recon = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x_img, logits=self.deconv_2), 1))
        self.recon = tf.nn.sigmoid(self.deconv_2)
        self.recon = tf.clip_by_value(self.recon, 1e-8, 1 - 1e-8)
        self.loss_recon = tf.reduce_mean(tf.reduce_sum(self.x_img * tf.log(self.recon) + (1 - self.x_img) * tf.log(1 - self.recon), [1, 2]))

        # Equation 10 & 11.
        # Regularizer: Kullback-Leibler divergence of latent prior.
        # L^z = (1/2) * {Sum_(t=1)^(T) mu^2 + sigma^2 - log(sigma^2)} - (T/2)
        kl_list = [0]*self.sequence_length
        for t in range(self.sequence_length): kl_list[t] = 0.5 * tf.reduce_sum(tf.square(self.mu[t]) + tf.square(self.sigma[t]) - tf.log(tf.square(self.sigma[t]) + 1e-12), 1) - 0.5
        self.loss_kl = tf.reduce_mean(tf.add_n(kl_list)) # element wise sum using tf.add_n

        self.loss_total = self.loss_recon + self.loss_kl

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_total)

        tf.summary.scalar('loss_recon', self.loss_recon)
        tf.summary.scalar('loss_kl', self.loss_kl)
        tf.summary.scalar('loss_total', self.loss_total)
        self.summaries = tf.summary.merge_all()

    def binary_crossentropy(self, t,o): return -(t*tf.log(o+1e-12) + (1.0-t)*tf.log(1.0-o+1e-12))

    def sample_latent(self, mu, sigma): return mu + sigma * self.z_noise

    def basic_read(self, inputs, x_hat): return tf.concat([inputs, x_hat], 1)

    def extract_shape(self, inputs): return inputs.shape[1].value, inputs.shape[2].value, inputs.shape[3].value, (inputs.shape[1] * inputs.shape[2] * inputs.shape[3]).value

    def basic_write(self, hidden_state):

        with tf.variable_scope("write", reuse=self.share_parameters):
            decoded_image_portion = self.fully_connected(hidden_state, self.n_hidden, self.c_h*self.c_w)

        return decoded_image_portion

    def attention_read(self, inputs, x_hat, h_prev_dec):

        Fx, Fy, gamma = self.attn_window("read", h_prev_dec)

        x = self.filter_img(inputs, Fx, Fy, gamma)
        x_hat = self.filter_img(x_hat, Fx, Fy, gamma)

        return self.basic_read(x, x_hat)

    def attention_write(self, hidden_state):

        with tf.variable_scope("writeW", reuse=self.share_parameters):
            w = self.fully_connected(hidden_state, self.n_hidden, self.attention_n**2)
        w = tf.reshape(w, [-1, self.attention_n, self.attention_n])
        Fx, Fy, gamma = self.attn_window("write", hidden_state)
        Fyt = tf.transpose(Fy, perm=[0,2,1])

        wr = tf.matmul(Fyt, tf.matmul(w, Fx))
        wr = tf.reshape(wr, [-1, self.c_h*self.c_w])

        tmpout = wr * tf.reshape(1.0/gamma, [-1, 1])
        return tf.reshape(tmpout, [-1, self.c_h*self.c_w])

    def attn_window(self, scope, h_dec):

        with tf.variable_scope(scope,reuse=self.share_parameters):
            params_tmp = self.fully_connected(h_dec, self.n_hidden, 5) # make parameters by fully connencted layer.

        gx_, gy_, log_sigma_sq, log_delta_, log_gamma = tf.split(params_tmp, 5, 1)
        gx = ((self.c_w + 1) / 2) * (gx_ + 1)
        gy = ((self.c_h + 1) / 2) * (gy_ + 1)

        sigma_sq = tf.exp(log_sigma_sq)
        delta = ((max(self.c_w, self.c_h) - 1) / (self.attention_n-1)) * tf.exp(log_delta_)

        Fx, Fy = self.filterbank(gx, gy, sigma_sq, delta)
        return Fx, Fy, tf.exp(log_gamma)

    def filter_img(self, inputs, Fx, Fy, gamma): # apply parameters for patch of gaussian filters

        Fxt = tf.transpose(Fx, perm=[0,2,1])
        img = tf.reshape(inputs, [-1, self.c_h, self.c_w])

        glimpse = tf.matmul(Fy, tf.matmul(img, Fxt)) # gaussian patches
        glimpse = tf.reshape(glimpse, [-1, self.attention_n**2])

        return glimpse * tf.reshape(gamma, [-1, 1]) # rescale

    def filterbank(self, gx, gy, sigma_sq, delta):

        grid_c = tf.ones((tf.shape(self.x)[0]*self.c_c, 1))
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), tf.float32), [1, -1])
        grid_i = tf.matmul(grid_c, grid_i)

        # Cordination is moved to (0, 0) by Equation 19 & 20.
        # Equation 19.
        mu_x = gx + (grid_i - (self.attention_n / 2) - 0.5) * delta
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])
        # Equation 20.
        mu_y = gy + (grid_i - (self.attention_n / 2) - 0.5) * delta
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])

        # (i, j) of F is a point in the attention patch.
        # (a, b) is a point in the input image.
        a = tf.reshape(tf.cast(tf.range(self.c_w), tf.float32), [1, 1, -1])
        b = tf.reshape(tf.cast(tf.range(self.c_h), tf.float32), [1, 1, -1])

        sigma_sq = tf.reshape(sigma_sq, [-1, 1, 1])

        # Equation 25.
        Fx = tf.exp(-(tf.square(a - mu_x) / (2*sigma_sq)))
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True),1e-12)
        # Equation 26.
        Fy = tf.exp(-(tf.square(b - mu_y) / (2*sigma_sq)))
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True),1e-12)

        return Fx, Fy

    def xavier_std(self, in_dim): return 1. / tf.sqrt(in_dim / 2.)

    def fully_connected(self, inputs, in_dim, out_dim, scope=None):

        with tf.variable_scope(scope or "Linear"):
            W = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=self.xavier_std(in_dim)))
            b = tf.Variable(tf.zeros(shape=[out_dim]))

            return tf.matmul(inputs, W) + b

    def encode(self, prev_state, inputs):

        with tf.variable_scope("encoder", reuse=self.share_parameters):
            hidden_state, next_state = self.lstm_enc(inputs, prev_state)

        # Equation 1.
        # mu_t = h_t_enc * W + b
        with tf.variable_scope("mu", reuse=self.share_parameters):
            mu = self.fully_connected(hidden_state, self.n_hidden, self.n_z)

        # Equation 2.
        # sigma_t = exp(h_t_enc * W + b)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            sigma = tf.exp(self.fully_connected(hidden_state, self.n_hidden, self.n_z))

        return mu, sigma, hidden_state, next_state

    def decode(self, prev_state, latents):

        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_state, next_state = self.lstm_dec(latents, prev_state)

        return hidden_state, next_state

    def conv2d(self, inputs, num_inputs, num_outputs, kernel_size=5, stride=1, padding='SAME', activation='sigmoid', scope=None):

        with tf.variable_scope("Conv_Enc", reuse=self.share_parameters):
            weight = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_inputs, num_outputs], stddev=self.xavier_std(in_dim=num_inputs)))
            bias = tf.Variable(tf.random_normal([num_outputs], stddev=self.xavier_std(in_dim=num_inputs)))

            out_conv = tf.nn.conv2d(input=inputs, filter=weight, strides=[1, stride, stride, 1], padding=padding, use_cudnn_on_gpu=True, data_format='NHWC', name=None)
            out_bias = tf.add(out_conv, bias)

            # print(out_bias.shape)
            h, w, c, hwc = self.extract_shape(out_bias)
            if(activation is 'sigmoid'): return tf.nn.sigmoid(out_bias), h, w, c, hwc
            elif(activation is 'tanh'): return tf.nn.tanh(out_bias), h, w, c, hwc
            elif(activation is 'relu'): return tf.nn.relu(out_bias), h, w, c, hwc
            else: return tf.nn.sigmoid(out_bias), h, w, c, hwc # default

    def conv2d_transpose(self, inputs, num_inputs, num_outputs, output_shape, kernel_size=5, stride=1, padding='SAME', activation='sigmoid', scope=None):

        with tf.variable_scope("Conv_Dec", reuse=self.share_parameters):
            weight = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_outputs, num_inputs], stddev=self.xavier_std(in_dim=num_inputs)))
            bias = tf.Variable(tf.random_normal([num_outputs], stddev=self.xavier_std(in_dim=num_inputs)))

            out_conv = tf.nn.conv2d_transpose(value=inputs, filter=weight, output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding, data_format='NHWC', name=None)
            out_bias = tf.add(out_conv, bias)

            # print(out_bias.shape)
            if(activation is 'sigmoid'): return tf.nn.sigmoid(out_bias)
            elif(activation is 'tanh'): return tf.nn.tanh(out_bias)
            elif(activation is 'relu'): return tf.nn.relu(out_bias)
            else: return tf.nn.sigmoid(out_bias) # default
