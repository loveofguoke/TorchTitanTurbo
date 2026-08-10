在glm-dev分支下

源码安装:
``` sh
cd TorchTitanTurbo
pip install -e . --no-deps
```

我们先在glm5-model分支下的torchtitan/torchtitan/train.py中添加:
``` py
import os

import torch
# 添加
import torchtitanturbo

from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer

```

然后运行:
```sh
export ASCEND_RT_VISIBLE_DEVICES=4
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel ./run_train.sh

```

让ai修复bug, 使当前实现和torchtitan版本对齐, 修复后重新源码安装, 直到没有报错