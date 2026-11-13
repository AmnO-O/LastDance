# zaic_tools

Thư viện chạy tool local của ZAC 2026, mỗi split một bản. BTC chấm bằng cách replay `tool_calls` trong file nộp trên `state.json` ban đầu bằng đúng gói này, rồi so state với đáp án và tính F1 trên call.

```python
import json, sys
sys.path.insert(0, "public_test")                  # chạy từ thư mục gói dữ liệu (chứa public_test/)
from zaic_tools import ToolEnv

SPLIT = "public_test"

def infer(item):
    env = ToolEnv(SPLIT, item["id"], root=".")     # mỗi mẫu đúng một phiên, nạp <split>_environment/<id>/state.json + catalog
    # ASR -> LLM -> env.call(tool, arguments) -> LLM -> ...
    # env.validate("sms.send", {"content": "hi"})          # chỉ kiểm schema -> {"ok": False, "error": "..."}
    # env.call("smart_home.get_status", {})                # -> {"ok": True, "devices": [...]}; state cập nhật trong bộ nhớ
    # env.call("smart_home.light_set", {"device_id": "light_kitchen", "power": "off"})
    # env.call("sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]})   # luôn có sẵn, chỉ ghi nhận
    return env.get_tool_calls(), env.get_state_hash()   # mọi call đã gọi (kể cả call bị từ chối) và hash của state cuối

with open(f"{SPLIT}/{SPLIT}.jsonl", encoding="utf-8") as fin, open("submission.jsonl", "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        tool_calls, state_hash = infer(item)
        fout.write(json.dumps({"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash}, ensure_ascii=False) + "\n")
```

- `ToolEnv(split, question_id, root)`: `root` là thư mục chứa các thư mục split.
- `env.validate(tool, args)` → `{"ok", "error"}`, không chạy. `env.call(tool, args)` → kết quả tool, state đổi trong bộ nhớ. `env.get_tool_calls()`, `env.get_state_hash()`: hai trường của dòng nộp. `env.get_state()`: state hiện tại của mọi app, để debug hoặc so với `label.final_state` của mẫu train.
- Call bị từ chối (`{"ok": false, "error": ...}`, state không đổi) **chỉ khi sai schema**: tool không có trong catalog hoặc không thuộc app đã cài, thiếu `required`, sai `type`/`enum`/`minimum`/`maximum`/`format`/`items`, thiếu cả nhóm `x-one-of`. ID không tồn tại, số dư không đủ, tìm không thấy... không phải lỗi: tool chạy, không đổi state, trả `{"ok": true, "done": false, "message": ...}` hoặc danh sách rỗng.
- App thông tin (weather, gold_price, fuel_price...) chỉ đọc; dữ liệu của chúng không thuộc phần state được so khi chấm.
- Dòng lệnh (từ thư mục gói dữ liệu, `PYTHONPATH=public_test`): gọi một tool `python -m zaic_tools --split public_test --root . --id public_0001 --tool smart_home.get_status --args '{}'`; kiểm tra schema cả file nộp `python -m zaic_tools --split public_test --root . --validate submission.jsonl`; liệt kê app/tool `--apps`.

## Cấu trúc

| File | Nội dung |
|---|---|
| `apps/<app>.py` | mỗi app một module, mỗi tool một method (`smart_home.get_status` → `apps/smart_home.py`, `SmartHome.get_status`); `@read`/`@write` đánh dấu loại tool. `apps/_*.py` là helper dùng chung. |
| `apps/base.py` | lớp `App`: `read`, `create`, `patch`, `delete` dùng chung trên phần state của app. |
| `executor.py` | `Session`: kiểm tra schema rồi gọi method tương ứng, ghi lại call; `Executor` nạp cả split. |
| `validate.py` | `schema_error` / `schema_validate`: luật từ chối duy nhất, executor, `env.validate`, CLI và scorer dùng chung. |
| `utils.py` | `canon(state)` bỏ `_meta`, app thông tin, trường text tự do; `state_hash(state)` = sha256 của `canon`. |

Sample nào gọi được tool nào do `installed_apps` của sample và catalog của split quyết định. Không sửa gói này hay `*_environment/`: server luôn dùng bản gốc. Python 3.10+, không cần thư viện ngoài.
