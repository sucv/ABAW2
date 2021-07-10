# Multi-task exp fork from [main branch](https://github.com/Chrisa142857/ABAW2/tree/main)

> Using pretrained ResNet50 features.
> Using TCN to classfy the sequence of features.
> Three tasks: Predicting Action Units (**AU**), Expression (**EXPR**), and Valence (**V**) and Arousal (**A**).

## Results on the validation set:

| tag | Best/Epochs | Total (AU) | Total (EXPR) | CCC (V) | CCC (A) |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |
| Unimodal |  |  |  |  |  |
| V | 41/53 | --- | --- | **<font color=red>0.4252</font>** | --- |
| A | 29/49 | --- | --- | --- | 0.6472 |
| AU | 36/38 | **<font color=red>0.5787</font>** | --- | --- | --- |
| EXPR | 3/12 | --- | 0.3313 | --- | --- |
| A + AU | 5/24 | 0.3737 | --- | --- | 0.6138 |
| A + AU + Attn | 24/44 | 0.3940 | --- | --- | **<font color=red>0.6530</font>** |
| V + AU + Attn | 6/26 | 0.3944 | --- | 0.3642 | --- |
| A + EXPR + Attn | 7/27 | --- | 0.3418 | --- | 0.6262 |
| V + EXPR + Attn | 9/25 | --- | 0.3466 | 0.3742 | --- |
| A + AU + EXPR + Attn | 3/18 | 0.3567 | 0.3039 | --- | 0.6226 |
| AU + EXPR + Attn | 32/38 | 0.4279 | **<font color=red>0.4077</font>** | --- | --- |
|  |  |  |  |  |  |
| Multimodal |  |  |  |  |  |
| AU | 12/32 | 0.5723 | --- | --- | --- |
| EXPR | 7/27 | --- | 0.3395 | --- | --- |
| AU + EXPR + Attn | 7/27 | 0.4223 | 0.3390 | --- | --- |

> *Attn: Multi-task Attention.

# Methods

1. Directly stacking one Linear layer (FC) for each task after the shared TCN with every task. 

2. For multi-task attention, two stages are stacked as the attention module before output layers. 

  i. In stage one, with `N` tasks, one FC operation _a_ has `1` output channel for the first task and another _b_ has `N-1` channels for other tasks are activated by `nn.Tanh()` and `nn.Sigmoid`, respectively. Then, `a = a.mul(b[..., n])`, where `n` indicates each channel of _b_ and `n <= N-1`. 

  ii. In stage two, with one FC `out` has `N` output channels, `attn = Softmax(out(a))`. The dot-product is used between every frame of input embedded features with the task attention, `x = attn[:, n]*x`, where `n <= N`.

# Conclusion

Briefly, some simple Linear layers are rather powerless for new challenging tasks, even Attn improved some performances.
