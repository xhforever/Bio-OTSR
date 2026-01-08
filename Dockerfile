FROM continuumio/miniconda3

WORKDIR /workspace/SKEL-CF

# 1. 创建 conda 环境
COPY conda.yml .
RUN conda env create -f conda.yml \
 && conda clean -afy

# 2. 激活 skelvit 环境
SHELL ["conda", "run", "-n", "skelvit", "/bin/bash", "-c"]

# 3. 安装 pip 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 拷贝 SKEL-CF 全部代码
COPY . .

# 5. 默认入口（可按需改）
CMD ["bash", "scripts/train.sh"]
