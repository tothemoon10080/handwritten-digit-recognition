# handwritten digit recognition
 CNN implements mnist handwritten digit recognition

遇到的问题
1.  问题：在尝试from tensorflow.keras import datasets, layers, models遇到了无法解析导入“tensorflow.keras” 
    原因：在TensorFlow 2.4版本之后，tensorflow.keras已被弃用。
    解决办法：直接导入keras from keras import datasets, layers, models

2.  问题：在运行时遇到ValueError: Shapes (32, 1) and (32, 10) are incompatible.
    尝试的步骤：检查最后一层的神经元数量是否与问题中的类别数量相匹配，发现符合。
    原因：标签的形状与模型输出层的形状不匹配导致的，标签应该是一个10维的向量，而不是一个单一的数字。
    解决办法：使用keras.utils.to_categorical函数将标签转换为one-hot编码的形式。
    train_labels = keras.utils.to_categorical(train_labels, 10) 

    额外发现：loss=tf.keras.losses.CategoricalCrossentropy() 与loss=tf.keras.losses.SparseCategoricalCrossentropy()是针对不同形式的标签（label）而设计的。如果您的标签是one-hot编码的，也就是说每个样本只属于一个类别，并且用一个向量来表示，例如[0, 0, 1]表示第三个类别，那么您可以使用categorical_crossentropy作为损失函数。如果您的标签是整数编码的，也就是说每个样本只用一个数字来表示类别，例如2表示第三个类别，那么您可以使用SparseCategoricalCrossentropy作为损失函数。这两种损失函数在数学上是等价的，只是在实现上有一些细节上的差别。

    也就是说在这个情况下更改模型中的loss不需要转换标签，可以直接使用整数编码的形式。这样可以节省一些空间和时间。

一些总结
1.  在个项目中使用了Keras框架中的顺序模型（Sequential Model）在最后的输出层使用了softmax激活函数，适用于多类别分类问题。
    反向传播算法在代码中是有应用的，但是它是由keras框架自动完成的

    优化器（optimizer）：优化器决定了模型如何更新和调整参数以最小化损失函数。常见的优化器包括随机梯度下降（SGD）、Adam、RMSprop等。根据问题的特点和需求，可以选择适当的优化器。
    损失函数（loss function）：损失函数衡量模型的输出与实际标签之间的差异，用于指导模型的训练和优化。对于不同的问题类型，可以选择合适的损失函数，如均方误差（Mean Squared Error，MSE）用于回归问题，分类交叉熵（Categorical Crossentropy）用于分类问题等。
    评估指标（metrics）：评估指标用于衡量模型在训练和测试过程中的性能。常见的评估指标包括准确率、精确率、召回率等。根据具体任务的需求，可以选择适当的评估指标。