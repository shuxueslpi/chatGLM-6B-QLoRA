# -*- coding: utf-8 -*-
# time: 2023/6/9 15:59
# file: inference_test.py
# author: zmfy
# email: shuxueslpi@163.com

import argparse

from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def main(model_path,
         loop_number=100,
         max_time=1,
         instruction_text='写1000字的文章：\n'):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    # 先预热一次模型
    response, history = model.chat(tokenizer=tokenizer,
                                   query=instruction_text,
                                   do_sample=False,
                                   temperature=0, max_time=5)

    total_n = 0
    print('开始测试...')
    for _ in tqdm(range(loop_number)):
        response, history = model.chat(tokenizer=tokenizer,
                                       query=instruction_text,
                                       do_sample=False,
                                       temperature=0, max_time=max_time)  # 每次最多运行max_time秒
        total_n += len(tokenizer(response)['input_ids'])  # 这里统计max_time秒内生成的token数量
    print('+' + '-' * 40 + '+')
    print(f'模型路径：{model_path}')
    print(f'运行次数：{loop_number}')
    print(f'每次运行时长：{max_time}s')
    print(f'平均每次运行输出token数量：{total_n / loop_number}')
    print('+' + '-' * 40 + '+')


def parse_args():
    parser = argparse.ArgumentParser(description='ChatGLM-6B inference test.')
    parser.add_argument('--model_path', type=str, required=True, help='待测试的模型保存路径')
    parser.add_argument('--loop_number', type=int, default=100, help='测试模型推理的运行次数')
    parser.add_argument('--max_time', type=int, default=1, help='每次测试模型运行的最长时间，单位秒')
    parser.add_argument('--instruction_text', type=str, default='写1000字的文章：\n', help='测试输入的指令')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(model_path=args.model_path,
         loop_number=args.loop_number,
         max_time=args.max_time,
         instruction_text=args.instruction_text)





