import tensorflow as tf 
import numpy as np

"""
[[0,0], -> [1, 0, 0] 기타
 [1,0], -> [0, 1, 0] 포유류
 [1,1], -> [0, 0, 1] 조류
 [0,0], -> [1, 0, 0] 기타
 [0,0], -> [1, 0, 0] 기타
 [0,1]  -> [0, 0, 1] 조류
"""

class Mammal:
    @staticmethod
    def execute():
        # [털, 날개] -> 기타, 포유류, 조류
        x_data = np.array(
            [[0, 0],
             [1, 0],
             [1, 1],
             [0, 0],
             [0, 0],
             [0, 1]
             ]
        )
        y_data = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 0, 0],
             [1, 0, 0],
             [0, 0, 1]
             ]
        )
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        W = tf.Variable(tf.random_uniform([2,3], -1, 1.))
        # -1 all
        # nn 은 2차원으로 [입력층(특성), 출력층(레이블)] -> [2,3]
        b = tf.Variable(tf.zeros([3]))
        # b 는 편향 bias
        # b 는 각 레이어의 아웃풋 갯수로 결정함
        L = tf.add(tf.matmul(X, W), b) # y = WX + b
        L = tf.nn.relu(L)
        model = tf.nn.softmax(L)
        '''
        softmax 함수는 다음 처럼 결과값을 전체 합이 1인 확률로 만들어 주는 함수
        예) [8.04, 2.76, -6.52] -> [0.53, 0.24, 0.23]
        '''
        print(f'모델 내부 보기 : {model}')

        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis = 1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
        train_op = optimizer.minimize(cost)

        # 비용함수를 최소화 시키면 (= 경사도를 0으로 만들면) 그 값이 최적화 된다

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for step in range(100):
            sess.run(train_op, {X: x_data, Y: y_data})
            if (step + 1) % 10 == 0:
                print(step + 1, sess.run(cost, {X: x_data, Y: y_data}))

        prediction = tf.argmax(model, 1)
        target = tf.argmax(Y, 1)
        print('예측값:', sess.run(prediction, feed_dict={X: x_data})) 
        print('실제값:', sess.run(target, feed_dict={Y: y_data}))

        is_correct = tf.equal(prediction, target)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print('정확도 : %2.f' % sess.run(accuracy * 100, {X: x_data, Y: y_data}))

mammal = Mammal()
mammal.execute()
