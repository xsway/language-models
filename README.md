# language-models
Various flavours of language models in PyTorch: tied, bidirectional etc.  
Extended and modified from [PyTorch language model tutorial](https://github.com/pytorch/examples/tree/master/word_language_model).


*Requires*: PyTorch 0.4+

## Arguments

You can run three types of language models by specifying the flag `--mode`: 
- `forward`: predict <img src="/svgs/40e217e3e6b01852275fa14991484e3d.svg" align=middle width=148.71961499999998pt height=24.56552999999997pt/>
- `backward`: predict <img src="/svgs/b1132e818d1a8a2cf1ea188558a0ba97.svg" align=middle width=148.71961499999998pt height=24.56552999999997pt/>
- `bidir` (bi-directional): predict <img src="/svgs/7ad32f76e3d869bf7731997c6ed06097.svg" align=middle width=256.002945pt height=24.56552999999997pt/>

Additionally, you can:
- share input and output embeddings with the flag `--tied standard` or `--tied plusL`
- use pre-trained embeddings (fasttext-style embeddings are supported) using the flag `--pretrained_embs path_to_embs` 

To see all arguments run `python train.py --help`.

## Example

Using the standard wikitext-2 data (in a folder with train.txt/valid.txt/test.txt split) you can run:

`python train.py --data data/wikitext-2/ --cuda --mode bidir --tied standard`
