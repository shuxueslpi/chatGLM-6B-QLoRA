# ChatGLM-6B-QLoRA

## 介绍

本项目使用peft库，实现了ChatGLM-6B模型4bit的QLoRA高效微调，可以在一张RTX3060上完成全部微调过程。

## 环境配置

### 环境依赖

- CUDA >= 11.7

- python依赖项

    ```text
    transformers==4.30.0.dev0
    peft==0.4.0.dev0
    accelerate==0.20.0.dev0
    datasets==2.12.0
    tqdm==4.65.0
    loguru==0.7.0
    bitsandbytes==0.39.0
    cpm_kernels==1.0.11
    sentencepiece==0.1.99
    ```
  注意：其中带有dev0后缀的包，当前在各个pip源还无法索引到，可先安装低版本后再用以下命令升级，比如安装transformers==4.29.1，再执行以下第一条命令，peft和accelerate类似处理：
  ```shell
  pip install -q -U git+https://github.com/huggingface/transformers.git
  pip install -q -U git+https://github.com/huggingface/peft.git
  pip install -q -U git+https://github.com/huggingface/accelerate.git
  ```

### 推荐环境

推荐使用docker镜像，如下：

```shell
docker pull huggingface/transformers-pytorch-gpu:4.29.1
docker run -tid -v /your/data/path:/home -p 58323:58323 --gpus all huggingface/transformers-pytorch-gpu:4.29.1

# 上面命令中的 -p 58323:58323 如果你想用tensorboard查看训练过程，需要映射一个端口出来

# 进入容器
docker exec -ti container_name /bin/bash
```

进入容器后，执行：

```shell
python3 -m pip install --upgrade pip

pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

## 使用方法

### 数据集说明

数据集使用ADGEN广告数据集，任务为根据instruction生成一段广告词，见本项目data文件夹，每条样本为一行，形式为：

```json
{
  "instruction": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤", 
  "output": "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。毕竟好穿时尚，谁都能穿出腿长2米的效果宽松的裤腿，当然是遮肉小能手啊。上身随性自然不拘束，面料亲肤舒适贴身体验感棒棒哒。系带部分增加设计看点，还让单品的设计感更强。腿部线条若隐若现的，性感撩人。颜色敲温柔的，与裤子本身所呈现的风格有点反差萌。"
}
```

其中训练数据`train.jsonl`共计114599条，验证数据`dev.jsonl`共计1070条。

### 启动训练

进入本项目目录，训练启动命令如下：

```shell
python3 train_qlora.py \
--train_args_json chatGLM_6B_QLoRA.json \
--model_name_or_path THUDM/chatglm-6b \
--train_data_path data/train.jsonl \
--eval_data_path data/dev.jsonl \
--lora_rank 4 \
--lora_dropout 0.05 \
--compute_dtype fp32
```

其中`chatGLM_6B_QLoRA.json`文件为所有transformers框架支持的TrainingArguments，参考：https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

默认如下，可根据实际情况自行修改：

```json
{
    "output_dir": "saved_files/chatGLM_6B_QLoRA_t32",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "per_device_eval_batch_size": 4,
    "learning_rate": 1e-3,
    "num_train_epochs": 1.0,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "logging_steps": 100,
    "save_strategy": "steps",
    "save_steps": 500,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "optim": "adamw_torch",
    "fp16": false,
    "remove_unused_columns": false,
    "ddp_find_unused_parameters": false,
    "seed": 42
}
```


对参数`compute_type`，可选`fp16`, `bf16`和`fp32`，实测使用`fp16`, `bf16`这两种计算速度有明显提升，相同的epoch只需要大约一半的时间，但出现loss收敛较慢的情况，默认选择`fp32`.

关于这个参数的选择，可能需要根据数据集做不同的尝试。

### 训练截图

- 显存占用，batch_size = 4

    ![img.png](pics/nvidia-smi.png)

- loss曲线，训练一个epoch，可以看到loss还在下降
    
  ![img.png](pics/train_loss.png)
  ![img.png](pics/eval_loss.png)

默认会在`saved_files/chatGLM_6B_QLoRA_t32`文件夹中生成一个`runs`的文件夹，可进入`saved_files/chatGLM_6B_QLoRA_t32`文件夹，用以下命令启动tensorboard，查看训练曲线：

```shell
tensorboard --logdir runs --port 'your port' --bind_all
```

## 推理示例

训练过程会保存adapter的checkpoint及最终的adapter文件，默认配置下每个文件的情况如下：

```text
-rw-r--r-- 1 root root  417 Jun  2 21:14 adapter_config.json
-rw-r--r-- 1 root root 7.1M Jun  2 21:14 adapter_model.bin
-rw-r--r-- 1 root root  15M Jun  2 21:14 optimizer.pt
-rw-r--r-- 1 root root  15K Jun  2 21:14 rng_state.pth
-rw-r--r-- 1 root root  627 Jun  2 21:14 scheduler.pt
-rw-r--r-- 1 root root 2.0K Jun  2 21:14 trainer_state.json
-rw-r--r-- 1 root root 3.9K Jun  2 21:14 training_args.bin
```

保存的adapter只有约7M的大小。

### 推理代码

使用如下代码进行推理：

```python
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


peft_model_path = 'saved_files/chatGLM_6B_QLoRA_t32'

config = PeftConfig.from_pretrained(peft_model_path)
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float32)

base_model = AutoModel.from_pretrained(config.base_model_name_or_path,
                                       quantization_config=q_config,
                                       trust_remote_code=True,
                                       device_map='auto')

input_text = '类型#裙*版型#显瘦*风格#文艺*风格#简约*图案#印花*图案#撞色*裙下摆#压褶*裙长#连衣裙*裙领型#圆领'
print(f'输入：\n{input_text}')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

response, history = base_model.chat(tokenizer=tokenizer, query=input_text)
print(f'微调前：\n{response}')

model = PeftModel.from_pretrained(base_model, peft_model_path)
response, history = model.chat(tokenizer=tokenizer, query=input_text)
print(f'微调后: \n{response}')
```

### 微调前后推理对比

```text
输入：

类型#裙*版型#显瘦*风格#文艺*风格#简约*图案#印花*图案#撞色*裙下摆#压褶*裙长#连衣裙*裙领型#圆领

微调前输出：

* 版型：修身
* 显瘦：True
* 风格：文艺
* 简约：True
* 图案：印花
* 撞色：True
* 裙下摆：直筒或微喇
* 裙长：中长裙
* 连衣裙：True

微调后输出：

一款简约而不简单的连衣裙，采用撞色的印花点缀，打造文艺气息，简约的圆领，修饰脸型。衣袖和裙摆的压褶，增添设计感，修身的版型，勾勒出窈窕的身材曲线。
```

### 关于灾难性遗忘

看到很多朋友表示微调后的模型出现灾难性疑问的问题，无法对微调数据集以外的问题进行正常反应，我对微调后的模型进行了一些其他问题的测试。

多次测试中发现遗忘情况出现较少，但有时会出现输出<unk>的异常情况。

以下是一些正常响应和异常响应的例子：

正常响应：

```text
输入：输出一份黄山旅游的计划

输出：

黄山作为中国著名的风景区之一，吸引着众多游客前来游览。以下是一份黄山旅游的计划，供您参考。

第一天：抵达黄山市，然后前往黄山风景区，游览黄山风景区，可以游览黄山的“四绝”，即云海、云松、雾海、雾松。
第二天：游览黄山松林，可以游览黄山的松林，可以体验黄山松林的秀丽。
第三天：游览黄山市区，可以游览黄山市区，可以游览黄山市区的景点。
```

```text
输入：大众汽车怎么样？

输出：

大众是一辆全球都受欢迎的品牌，它拥有众多经典款式，无论是城市驾驶还是城市购物，它都是一把好手。它拥有出色的操控性能，同时拥有舒适的驾驶体验。它拥有良好的燃油性能，拥有良好的耐用性。
```

```text
输入：下雨天骑电瓶车有哪些需要注意的问题？

输出：

骑电瓶车的时候需要注意以下几个方面的问题，这样保证能够安全出行：

1. 下雨天，骑电瓶车需要防止滑倒，因为下雨天，电瓶车会很容易滑倒，所以骑电瓶车时需要注意保持平衡，并且不要骑得过快，这样可以保证安全。

2. 下雨天，骑电瓶车时需要防止被雨淋湿，因为被雨淋湿之后会对身体造成伤害，所以骑电瓶车时需要防止淋雨，并且要穿防水衣。

3. 骑电瓶车时需要注意保护自己的脖子，因为脖子比较脆弱，容易被压到，所以骑电瓶车时需要注意保护脖子。```
```

异常响应，和以上正常响应一样的问题，多运行几次，会出现以下异常：

```text
输入：大众汽车怎么样？

输出：

大众是德国品牌，在汽车界有着深厚的历史，拥有德国车严谨的工艺精神，是<UNK>的<UNK>。作为<UNK>，它的设计简单大方，没有多余的装饰，但经典的造型却能够让你<UNK>其中，它的性能也非常出色，具有<UNK>的能力。
```

关于这部分异常，将做更多尝试和验证，比如在训练时统一添加同样的前缀。

## 参考

- https://github.com/THUDM/ChatGLM-6B
- https://huggingface.co/blog/4bit-transformers-bitsandbytes
- https://github.com/mymusise/ChatGLM-Tuning
