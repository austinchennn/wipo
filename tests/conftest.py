"""pytest 公共配置 —— 保证 `import src.xxx` 能从项目根目录解析。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
