# Multi-task exp fork from [main branch](https://github.com/Chrisa142857/ABAW2/tree/main)

> Using pretrained ResNet50 features.
> Using TCN to classfy the sequence of features.
> Three tasks: Predicting Action Units (**AU**), Expression (**EXPR**), and Valence (**V**) and Arousal (**A**).

## Results on the validation set:

| tag | Best/Epochs | Total (AU) | Total (EXPR) | CCC (V) | CCC (A) |
| --- | --- | --- | --- | --- | --- |
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

| --- | --- | --- | --- | --- | --- |
| fusion features |  |  |  |  |  |
| AU | running | running | --- | --- | --- |

> **Attn**: Multi-task Attention.
