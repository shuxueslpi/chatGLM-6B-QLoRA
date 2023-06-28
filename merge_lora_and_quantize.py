# -*- coding: utf-8 -*-
# time: 2023/6/7 13:44
# file: merge_lora_and_quantize.py
# author: zmfy
# email: shuxueslpi@163.com

import argparse
import os
from shutil import copyfile

import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig


def merge_lora(lora_path, device_map=None):
    """合并lora模型和base模型"""
    if device_map is None:
        device_map = {'': 'cpu'}
    config = PeftConfig.from_pretrained(lora_path)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path,
                                           load_in_8bit=False,
                                           trust_remote_code=True, torch_dtype=torch.float32,
                                           device_map=device_map)
    # check_weight = base_model.transformer.layers[0].attention.query_key_value.weight
    # check_weight_old = check_weight.clone()
    model = PeftModel.from_pretrained(base_model, lora_path, device_map=device_map)
    model = model.merge_and_unload()
    return model, config


def quantize(model, qbits=4):
    """量化模型，qbit为4或8"""
    qmodel = model.quantize(qbits).half().cuda()
    qmodel = qmodel.eval()
    return qmodel


def save_model_and_tokenizer(model, base_model_path, output_path, remote_path):
    """保存模型和tokenizer相关配置"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    for fp in os.listdir(remote_path):  # 拷贝remote_scripts中的官方脚本到最终输出的文件夹中，供load模型时使用
        if fp.split('.')[-1] == 'py':
            copyfile(os.path.join(remote_path, fp),
                     os.path.join(output_path, fp))


def main(lora_path, output_path, remote_path, qbits=4, device_map=None):
    if device_map is None:
        device_map = {'': 'cpu'}
    merged_model, lora_config = merge_lora(lora_path, device_map)
    if qbits in [4, 8]:
        quantized_model = quantize(merged_model, qbits)
        save_model_and_tokenizer(quantized_model, lora_config.base_model_name_or_path, output_path, remote_path)
        logger.info(f'''Lora model和base model成功merge, 并量化为{qbits}bits, 保存在{output_path}''')
    else:
        save_model_and_tokenizer(merged_model, lora_config.base_model_name_or_path, output_path, remote_path)
        logger.info(f'''Lora model和base model成功merge, 保存在{output_path}''')


def parse_args():
    parser = argparse.ArgumentParser(description='ChatGLM-6B merge lora and quantize.')
    parser.add_argument('--lora_path', type=str, required=True, help='QLoRA训练后保存模型的目录')
    parser.add_argument('--output_path', type=str, default='/tmp/merged_qlora_model', help='最终保存合并，量化后的模型目录')
    parser.add_argument('--qbits', type=int, default=4, help='模型量化位数')
    parser.add_argument('--device', type=str, default='auto', help='device_map')
    parser.add_argument('--remote_scripts_dir', type=str, default='remote_scripts/chatglm2-6b', help='官方脚本目录')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device != 'auto':
        device_map = {'': args.device}
    else:
        device_map = 'auto'
    main(lora_path=args.lora_path,
         output_path=args.output_path,
         remote_path=args.remote_scripts_dir,
         qbits=args.qbits,
         device_map=device_map)

