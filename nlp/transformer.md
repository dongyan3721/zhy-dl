# 输入处理

## 一些特殊符号、独热编码和词嵌入





## 位置编码

![873B98F54AFD1FF6D64FB23E19CD5185.png](../notebook_image_dir/873B98F54AFD1FF6D64FB23E19CD5185.png)
保证位置编码信息是正的且长度是$d$的向量





# 注意力机制

## 自注意力


![FD0D4CD8C4B951EBCF40E96E7CB48216.png](../notebook_image_dir/FD0D4CD8C4B951EBCF40E96E7CB48216.png)
$W$是可学习的参数，自身和自身的关联程度最高
![64F3A0567714C04236D79369D6C4C398.png](../notebook_image_dir/64F3A0567714C04236D79369D6C4C398.png)


$b$是新的$q$值


多头自注意力机制

![EE43F3E617824E00E4E2099334D9876F.png](../notebook_image_dir/EE43F3E617824E00E4E2099334D9876F.png)

$softmax(QK^T/\sqrt{d_k})$是注意力分数

类比到卷积的multi channel，$W_O$是dense层的参数

![A51785FCA2FB46B3431FEF5AD9B1D3CF.png](../notebook_image_dir/A51785FCA2FB46B3431FEF5AD9B1D3CF.png)
掩码注意力：目的是为了对齐批次数量


## 归一化层和前馈层

![FDFB79882AF0E52A0002E306CD128EF5.png](../notebook_image_dir/FDFB79882AF0E52A0002E306CD128EF5.png)

## transformer的训练过程
![transformer_train.png](../notebook_image_dir/transformer_train.png)
一次性将输入读入，用带掩码的真实标签做批量训练。

## transformer训练过程中的因果掩码
![mask_train.png](../notebook_image_dir/mask_train.png)

## transformer推理过程
![mask_eval.png](../notebook_image_dir/mask_eval.png)

## 交叉注意力机制
编码器的输出作为K/V输入到交叉注意力机制，解码器因果掩码自注意力机制的输出作为Q

