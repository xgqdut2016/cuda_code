softmax_v1.cu：先计算M，然后计算S

softmax_v2.cu：先计算M，然后计算S，速度比V1快

softmax_v3.cu：一个kernel

softmax_v4.cu：全局max

softmax_v5.cu：算子融合

softmax_v6.cu：先max，然后sum，手动规约

softmax_cub.cu：用BlockReduce函数规约