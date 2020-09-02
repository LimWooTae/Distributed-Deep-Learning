import numpy as np
import tensorflow as tf

def addone(x):
    # print(type(x)
    return x + 1

def addone_grad(op, grad):
    x = op.inputs[0]
    return x

from tensorflow.python.framework import ops
import numpy as np

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def pyfunc_test():

    # create data
    x_data = tf.placeholder(dtype=tf.float32, shape=[None])
    y_data = tf.placeholder(dtype=tf.float32, shape=[None])

    w = tf.Variable(tf.constant([0.5]))
    b = tf.Variable(tf.zeros([1]))

    y1 = tf.multiply(w, x_data, name='y1')
    y2 = py_func(addone, [y1], [tf.float32], grad=addone_grad)[0]
    y = tf.add(y2, b)

    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    print("Pyfunc grad", ops.get_gradient_function(y2.op))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10):
            #            ran = np.random.rand(115).astype(np.float32)
            ran = np.ones((115)).astype(np.float32)
            ans = ran * 1.5 + 3
            dic = {x_data: ran, y_data: ans}
            tt, yy, yy1= sess.run([train, y1, y2], feed_dict=dic)
            if step % 1 == 0:
                print('step {}'.format(step))
                print('{}, {}'.format(w.eval(), b.eval()))

        test = sess.run(y, feed_dict={x_data:[1]})
        print('test = {}'.format(test))


if __name__ == '__main__':
    pyfunc_test()
