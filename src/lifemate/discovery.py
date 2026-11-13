"""Search installed tool metadata by capability."""
from collections import Counter
import math
import re

from .text import fold

SEARCH_TOOL = "sys.search_tools"
E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SEARCH_LIMIT = 6
_STOP_VI = {"có", "cho", "của", "một", "nào", "cần", "để", "thế", "là", "và", "với", "các", "những", "cái", "a", "an", "the", "to", "for", "tool", "công", "cụ"}
_STOP_UNACCENTED = {"co", "cho", "mot", "nao", "can", "de", "the", "a", "to", "for", "tool", "cong", "cu"}

# 24 expanded Vietnamese descriptions for high-impact tools
EXPANSIONS = {
    "contacts.search": "Tra cứu danh bạ bằng tên hoặc số điện thoại. Lấy thông tin liên hệ và mã liên hệ.",
    "contacts.update": "Chỉnh sửa thông tin của liên hệ đã có, chẳng hạn số điện thoại hoặc tên.",
    "bill_payment.list_bills": "Kiểm tra công nợ và hạn đóng của các hóa đơn. Liệt kê hóa đơn theo dịch vụ hoặc trạng thái. Tiền mạng internet điện nước.",
    "bill_payment.pay": "Trả tiền thanh toán cho hóa đơn đã xác định bằng mã hóa đơn và nguồn tiền. Đóng tiền mạng cước internet điện nước.",
    "chat_ott.list_chats": "Liệt kê các cuộc trò chuyện hoặc nhóm chat để xác định phòng hội thoại zalo.",
    "chat_ott.read_chat": "Đọc nội dung và tin nhắn của một cuộc trò chuyện cụ thể zalo.",
    "chat_ott.send_message": "Gửi nội dung tin nhắn đến cuộc trò chuyện hoặc người nhận nhắn tin zalo.",
    "banking.get_balance": "Kiểm tra tiền khả dụng, số dư và thông tin các tài khoản ngân hàng.",
    "banking.list_beneficiaries": "Liệt kê danh bạ thụ hưởng, tìm người nhận chuyển khoản đã lưu.",
    "banking.transfer": "Thực hiện giao dịch chuyển một số tiền từ tài khoản ngân hàng.",
    "calendar.list_events": "Tra cứu lịch hẹn và các sự kiện đã có theo thời gian.",
    "calendar.update_event": "Đổi thông tin hoặc thời điểm của một sự kiện đã có.",
    "door_lock.list_locks": "Liệt kê ổ khóa cửa thông minh và thông tin của từng khóa.",
    "door_lock.set_lock": "Điều khiển một ổ khóa cụ thể để khóa hoặc mở cửa.",
    "household_chores.assign": "Giao một việc nhà đã xác định cho thành viên phụ trách.",
    "household_chores.list_chores": "Tra cứu các công việc nhà hiện có và người đang phụ trách.",
    "package_tracking.schedule_pickup": "Hẹn thời gian và địa điểm cho đơn vị vận chuyển đến lấy hàng.",
    "package_tracking.track": "Tra cứu trạng thái vận chuyển theo mã theo dõi bưu kiện.",
    "smart_home.get_status": "Đọc tình trạng thiết bị trong nhà, ví dụ nguồn điện, chế độ và giá trị hiện tại.",
    "calculator.calculate": "Tính toán biểu thức số học và trả kết quả.",
    "exchange_rate.get_rate": "Tra cứu tỷ giá hiện tại của cặp đồng tiền.",
    "exchange_rate.convert": "Quy đổi một số tiền cụ thể từ đồng tiền này sang đồng tiền khác.",
    "loyalty.list_vouchers": "Xem danh sách phiếu thưởng có sẵn và số điểm cần để đổi.",
    "loyalty.redeem": "Dùng điểm tích lũy để đổi một phần thưởng đã chọn.",
}


def _terms(value):
    raw = [w for w in re.findall(r"\w+", str(value).lower()) if w not in _STOP_VI and len(w) > 1]
    folded = [w for w in re.findall(r"[a-z0-9]+", fold(value)) if w not in _STOP_UNACCENTED and len(w) > 1]
    return raw + folded


def eligible_tools(catalog, installed_apps, opened=()):
    opened = set(opened)
    return [tool for tool in catalog.installed_tools(installed_apps)
            if tool.name not in opened and tool.app != "sys" and tool.kind in {"read", "write"}]


def rank_bm25(catalog, installed_apps, query, *, opened=(), expansions=EXPANSIONS):
    query_terms = set(_terms(query))
    if not query_terms:
        return []
    documents = []
    for tool in eligible_tools(catalog, installed_apps, opened):
        app = catalog.apps[tool.app]
        properties = " ".join(f"{name} {prop.get('description', '')}" for name, prop in tool.properties.items())
        words = (_terms(tool.name) * 3 + _terms(tool.description) * 2
                 + _terms(f"{app.name_vi} {app.description} {tool.kind} {properties}"))
        if expansions and tool.name in expansions:
            words.extend(_terms(expansions[tool.name]) * 2)
        documents.append((tool, Counter(words), len(words)))
    if not documents:
        return []
    average = sum(length for _, _, length in documents) / len(documents) or 1
    df = Counter(term for _, counts, _ in documents for term in query_terms & counts.keys())
    ranked = []
    for tool, counts, length in documents:
        norm = 1.2 * (.25 + .75 * length / average)
        score = sum(
            math.log1p((len(documents) - df[t] + .5) / (df[t] + .5))
            * 2.2 * counts[t] / (counts[t] + norm)
            for t in sorted(query_terms & counts.keys())
        )
        if score > 0:
            ranked.append((score, tool))
    ranked.sort(key=lambda pair: (-pair[0], pair[1].name))
    return ranked


_E5_MODEL = None
_E5_TOOLS = None
_E5_EMBEDDINGS = None
_E5_FAILED = False


def _get_e5_dense_ranking(catalog, installed_apps, query, opened, threshold=0.82):
    global _E5_MODEL, _E5_TOOLS, _E5_EMBEDDINGS, _E5_FAILED
    if _E5_FAILED:
        return []
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if _E5_MODEL is None:
            _E5_MODEL = SentenceTransformer(E5_MODEL_ID, revision=E5_REVISION, device="cpu")
            _E5_TOOLS = [t for t in catalog.tools.values() if t.kind in {"read", "write"} and t.app != "sys"]
            descriptions = [f"{t.name}. {catalog.apps[t.app].name_vi}. {t.description}" for t in _E5_TOOLS]
            expanded = [f"passage: {desc} {EXPANSIONS.get(t.name, '')}".strip()
                        for desc, t in zip(descriptions, _E5_TOOLS)]
            _E5_EMBEDDINGS = _E5_MODEL.encode(expanded, normalize_embeddings=True, show_progress_bar=False)

        eligible = {t.name for t in eligible_tools(catalog, installed_apps, opened)}
        indices = [i for i, t in enumerate(_E5_TOOLS) if t.name in eligible]
        if not indices:
            return []
        q_vec = _E5_MODEL.encode("query: " + query, normalize_embeddings=True, show_progress_bar=False)
        sims = _E5_EMBEDDINGS[indices] @ q_vec
        sorted_indices = np.argsort(-sims)
        if sims[sorted_indices[0]] < threshold:
            return []
        return [_E5_TOOLS[indices[i]].name for i in sorted_indices]
    except Exception:
        _E5_FAILED = True
        return []


def fuse_rrf(lexical_names, semantic_names, k=10, depth=10):
    scores = Counter()
    for rank, name in enumerate(lexical_names[:depth], 1):
        scores[name] += 1.0 / (k + rank)
    for rank, name in enumerate(semantic_names[:depth], 1):
        scores[name] += 1.0 / (k + rank)
    return sorted(scores, key=lambda name: (-scores[name], name))


def search_tools(catalog, installed_apps, query, *, opened=(), limit=SEARCH_LIMIT):
    bm25_ranked = [tool.name for _, tool in rank_bm25(catalog, installed_apps, query, opened=opened)]
    dense_ranked = _get_e5_dense_ranking(catalog, installed_apps, query, opened, threshold=0.82)
    if dense_ranked and bm25_ranked:
        fused = fuse_rrf(bm25_ranked, dense_ranked, k=10, depth=10)
        return [catalog.tools[name] for name in fused[:limit] if name in catalog.tools]
    elif dense_ranked:
        return [catalog.tools[name] for name in dense_ranked[:limit] if name in catalog.tools]
    return [catalog.tools[name] for name in bm25_ranked[:limit] if name in catalog.tools]
