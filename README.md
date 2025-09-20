# POINTS-Reader Deploy and Fine-tuning

## 模型架构

https://huggingface.co/tencent/POINTS-Reader

- llm
    - model
        - embed_tokens
        - layers
            - 0
                - input_layernorm
                - mlp
                    - down_proj
                    - gate_proj
                    - up_proj
                - post_attention_layernorm
                - self_attn
                    - o_proj
                    - q_proj
                    - k_proj
                    - v_proj
        - norm
    - lm_head
- vision_encoder
    - patch_embed.proj
    - blocks
        - 0
            - attn
                - proj
                - qkv
            - mlp.fc1
            - mlp.fc2
            - norm1
            - norm2
    - merger
        - ln_q
        - mlp
            - 0
            - 2
- vision_projector
    - ln_q
    - mlp
        - 0
        - 2


## 部署

```bash
# export HF_ENDPOINT=https://hf-mirror.com
hf download tencent/POINTS-Reader --local-dir models/POINTS-Reader

cd WePOINTS
pip install -e .
cd ../
```

transformer

```bash
python inference.py
```

配置支持 few-shot，但是只能用在很特殊的场景（数据同质化严重）
配置2个Prompt，默认的Prompt效果就不错。

gradio app

```bash
pip install gradio
python app.py
```

sglang (TODO)
```bash
conda create -n sglang python=3.12
cd sglang/
conda activate sglang
pip install --upgrade pip
pip install -e "python[all]"

cd ../
python3 -m sglang.launch_server \
--model-path models/POINTS-Reader \
--served-model-name POINTS-Reader \
--tp-size 1 \
--dp-size 1 \
--chat-template points-v15-chat \
--trust-remote-code \
--port 8081
```

## 微调

下载 OmniDocBench 数据集
```bash
hf download --repo-type dataset opendatalab/OmniDocBench --local-dir models/OmniDocBench
```

```bash
pip install accelerate trl deepspeed
```