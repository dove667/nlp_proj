# 如何合成 RULER 数据集

我没有服务器上的 docker 权限，而 NeMo Skills 默认使用 docker，所以不能走 NS，直接用 RULER 原仓库的代码。

```bash
git clone https://github.com/NVIDIA/RULER.git
```

忽略仓库中的 docker/，只关注 scripts/，下面有三个目录：data（合成数据），eval，pred。

data/ 中的代码可以合成 RULER 中各种数据。有些数据集的合成需要以来网络数据的爬取，如果服务器没法联网，可以现在本地爬好上传到服务器。

比如我们的实验 A 需要的数据：

```bash
TASKS=(
  niah_single_1
  niah_multikey_1
)
```
niah_multikey_1 需要先爬取 PaulGrahamEssays.json 数据。

另外 nltk 的 的 punkt 和 punkt_tab 也需要先下载好上传到服务器，下载代码（以 punkt_tab 为例）：

```bash
curl -L -o punkt_tab.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip
unzip -q punkt_tab.zip
```

放在服务器上：

```bash
/nltk_data
    /tokenizers
        /punkt_tab
        /punkt
```

在服务器上设置环境变量：

```bash
export NLTK_DATA=/data1/zsh/nltk_data
```

另外，目标模型的 tokenizer 也要准备好。

都上传好后就可以开始合成实验 A 需要的数据。

示例脚本（Llama-3.1-8B-Instruct）：[ruler_a.sh](ruler_a.sh)。注意脚本要放在 RULER/scripts/data/下面。

示例结果：
datasets/ruler
  /llama31_8b_instruct
    /4096
      /niah_single_1
      /niah_multikey_1
    /8192
      /niah_single_1
      /niah_multikey_1
    ...