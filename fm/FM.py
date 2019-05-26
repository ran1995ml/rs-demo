from data_loader import vectorize_dic
import pandas as pd
import tensorflow as tf
import numpy as np

class FM(object):

    def __init__(self,config):
        self.lr = config['lr']
        self.num_batches = config['num_batches']
        self.p = config['feature_length']
        self.k = config['k']
        self.l2 = config['l2']
        self.epochs = config['epochs']



    def add_placeholders(self):
        self.X = tf.placeholder('float',shape=[None,self.p])
        self.y = tf.placeholder('float',shape=[None,1])

    def inference(self):

        with tf.variable_scope('linear_layer'):
            self.w0 = tf.get_variable('bias',shape=[1],initializer=tf.zeros_initializer())
            self.w = tf.get_variable('w1',shape=[self.p],initializer=tf.truncated_normal_initializer())
            self.linear_terms = tf.add(self.w0,tf.reduce_sum(tf.multiply(self.w,self.X),1,keep_dims=True))

        with tf.variable_scope('interaction_layer'):
            self.v = tf.get_variable('v',shape=[self.p,self.k],initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))

            self.interaction_terms = tf.multiply(0.5,tf.reduce_sum(
                tf.subtract(tf.pow(tf.matmul(self.X,self.v),2),
                            tf.matmul(tf.pow(self.X,2),self.v))))

        self.y_out = tf.add(self.linear_terms,self.interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)


    def add_loss(self):
        l2_norm = self.l2*tf.reduce_sum(tf.pow(self.w0,2)) + self.l2*tf.reduce_sum(
            tf.pow(self.w,2)) + self.l2*tf.reduce_sum(tf.pow(self.v,2))
        error = tf.reduce_mean(tf.square(tf.subtract(self.y,self.y_out)))

        self.loss = tf.add(l2_norm,error)

    def train(self):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for epoch in range(self.epochs):
            perm = np.random.permutation(X_train.shape[0])
            X_ = X_train[perm]
            y_ = y_train[perm]

            n_samples = X_train.shape[0]
            losses = []
            for i in range(0,X_train.shape[0],self.num_batches):
                if i+self.num_batches>=n_samples:
                    batch_X = X_[i-self.num_batches:i]
                    batch_y = y_[i-self.num_batches:i]
                else:
                    batch_X = X_[i:i+self.num_batches]
                    batch_y = y_[i:i+self.num_batches]

                sess.run(self.optimizer,feed_dict={
                    self.X:batch_X.reshape(-1,self.p),
                    self.y:batch_y.reshape(-1,1)})
                losses.append(sess.run(self.loss,feed_dict={
                    self.X:batch_X.reshape(-1,self.p),
                    self.y:batch_y.reshape(-1,1)}))
            RMSE = np.sqrt(np.array(losses).mean())
            if epoch%2==0:
                print('Epoch [{}/{}], RMSE loss:{:.4f}'
                      .format(epoch+1,self.epochs,RMSE))

    def build_graph(self):
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.train()

if __name__ == "__main__":
    cols = ['user', 'item', 'rating', 'timestamp']
    train = pd.read_csv('../../data/ml-100k/u1.base', delimiter='\t', names=cols)
    test = pd.read_csv('../../data/ml-100k/u1.test', delimiter='\t', names=cols)

    train_dic = {'users': train['user'].values, 'items': train['item'].values}
    test_dic = {'users': test['user'].values, 'items': test['item'].values}

    X_train, ix = vectorize_dic(train_dic)
    X_test, ix = vectorize_dic(test_dic, ix, X_train.shape[1])

    y_train = train.rating.values
    y_test = train.rating.values

    X_train = X_train.todense()
    X_test = X_test.todense()


    config = {'lr':0.001, 'num_batches':100, 'l2':0.001,
              'k':50, 'feature_length':X_train.shape[1],
              'epochs':10}
    model = FM(config)
    model.build_graph()

