
import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from urllib import error, request

TOOLS = {"base", "database", "retrieval"}


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)


def parse_plan(raw):
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) != {"tasks"}:
        raise ValueError("Cần một đối tượng chỉ chứa tasks")
    tasks = obj["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks phải là danh sách có ít nhất một tác vụ")
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"tool", "question"}:
            raise ValueError("Mỗi tác vụ cần tool và question")
        if not isinstance(task["tool"], str) or task["tool"] not in TOOLS:
            raise ValueError("Tên công cụ không hợp lệ")
        if not isinstance(task["question"], str) or not task["question"].strip():
            raise ValueError("Nội dung tác vụ trống hoặc sai kiểu")
    return tasks


def load_cases(path):
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            user, gold = json.loads(line)["messages"]
            if (user["role"], gold["role"]) != ("user", "assistant"):
                raise ValueError("Sai thứ tự vai trò")
            inp = json.loads(user["content"])
            cases.append((inp["history"], inp["question"], parse_plan(gold["content"])))
        except (ValueError, TypeError, KeyError) as exc:
            raise SystemExit(f"Dữ liệu không hợp lệ tại dòng {line_no}: {exc}")
    return cases


def build_prompt(history, question):
    roles = {"user": "User", "assistant": "Assistant"}
    lines = [
        f"{roles[msg['role']]}: {msg['content']}"
        for msg in history[-10:] if msg.get("role") in roles
    ]
    history_text = "\n".join(lines) or "Không có lịch sử hội thoại."
    return f"""Bạn là Analyzer trong hệ thống Agentic Chatbot hỗ trợ học tập trên Moodle.

NHIỆM VỤ:
Đọc câu hỏi hiện tại và lịch sử hội thoại, xác định ý định hiện tại của người dùng
và tạo kế hoạch để Executor thực thi. Chỉ lập kế hoạch, không thực hiện tác vụ
và không trả lời trực tiếp câu hỏi.

========================
LỊCH SỬ HỘI THOẠI GẦN NHẤT
========================

{history_text}

========================
CÂU HỎI HIỆN TẠI
========================

{question}

========================
CÁC CÔNG CỤ
========================

database:
Tra dữ liệu có cấu trúc được lưu trên Moodle, như thông tin tài khoản, khóa học,
ghi danh, điểm, lượt làm quiz, hạn nộp, trạng thái nộp bài và nhận xét của giảng viên.
Giữ đúng người, môn học và hoạt động mà người dùng muốn tra.

Ví dụ:
- Điểm môn Python của tôi là bao nhiêu?
- Khoa đã tham gia những khóa học nào?
- Giảng viên nhận xét gì về bài báo cáo của tôi?

retrieval:
Tìm hoặc trích nội dung từ tài liệu nội bộ đã được lập chỉ mục, như slide, PDF,
giáo trình, đề cương và tài liệu thực hành. Dùng retrieval khi người dùng cần
thông tin theo một nguồn cụ thể, kể cả khi có thể trả lời chủ đề đó bằng kiến thức chung.

Ví dụ:
- Theo đề cương môn Python, bài nộp muộn được tính điểm thế nào?
- Trong slide Hệ điều hành, ví dụ về lập lịch ưu tiên nằm ở phần nào?

base:
Xử lý yêu cầu bằng kiến thức chung hoặc thông tin đã có, như giải thích,
lập trình, viết, dịch, tính toán, so sánh, tóm tắt hoặc chào hỏi.

Ví dụ:
- Giải thích sự khác nhau giữa tiến trình và luồng.
- Giả sử điểm giữa kỳ là 7 và cuối kỳ là 9, tính điểm trung bình với trọng số bằng nhau.

Chọn công cụ theo thông tin thực sự cần để hoàn thành từng tác vụ.
Việc câu hỏi nhắc đến “điểm”, “Moodle” hoặc “slide” chưa đủ để chọn công cụ.
Chẳng hạn, nếu người dùng đã cung cấp đủ điểm và chỉ yêu cầu tính toán, hãy dùng base.

========================
QUY TẮC LẬP KẾ HOẠCH
========================

1. Ưu tiên yêu cầu hiện tại. Dùng lịch sử để làm rõ các cụm như “môn đó”,
   “môn thứ hai”, “bài kia”, “điểm này” hoặc “còn nó thì sao”.
   Nếu người dùng sửa ý hoặc thu hẹp phạm vi, dùng yêu cầu mới nhất.
   Không biến yêu cầu cũ hoặc phương án đã bị loại bỏ thành tác vụ mới.

2. Giữ đủ các điều kiện làm thay đổi kết quả: đối tượng, môn học, mốc thời gian,
   phạm vi, nguồn tài liệu và các từ như “chưa”, “chỉ”, “trừ”, “lần đầu”, “lần cuối”.
   Không tự bổ sung tên, số liệu hoặc điều kiện mà câu hỏi và lịch sử chưa cung cấp.

3. Mỗi tác vụ dùng đúng một công cụ. Tách các yêu cầu có mục tiêu riêng.
   Giữ các điều kiện lọc của cùng một yêu cầu trong cùng tác vụ.
   Ví dụ, “bài đã nộp muộn và chưa được chấm” là một yêu cầu tra cứu có hai điều kiện.

4. Với các tác vụ độc lập, giữ thứ tự người dùng nêu. Nếu tác vụ sau cần kết quả
   của tác vụ trước, đặt bước cung cấp dữ liệu lên trước và mô tả rõ dữ liệu nào
   sẽ được dùng tiếp. Không giả định trước kết quả tra cứu.

5. Viết trường question thành yêu cầu cụ thể gửi cho công cụ. Nếu lịch sử đã xác định
   được tên môn hoặc tên bài, ghi tên đó trực tiếp. Không thêm bước giải thích,
   tìm tài liệu hay tra Moodle nếu yêu cầu của người dùng đã được đáp ứng đầy đủ.

========================
JSON ĐẦU RA BẮT BUỘC
========================

Chỉ trả về một đối tượng JSON có khóa "tasks". Giá trị của "tasks" là một mảng
không rỗng. Mỗi phần tử có đúng hai trường: "tool" và "question".
"tool" chỉ nhận một trong ba giá trị: "database", "retrieval", "base".
"question" là chuỗi mô tả tác vụ cần thực hiện.

Ví dụ 1:
Câu hỏi: "Điểm môn Cơ sở an toàn thông tin của Đạt là bao nhiêu?"
Đầu ra:
{{
  "tasks": [
    {{
      "tool": "database",
      "question": "Tra điểm môn Cơ sở an toàn thông tin của Đạt."
    }}
  ]
}}

Ví dụ 2:
Lịch sử: Người dùng đang trao đổi về môn Kinh tế vi mô.
Câu hỏi: "Môn đó trong giáo trình có ví dụ về chi phí cơ hội không? Cho mình thêm một ví dụ tự nghĩ."
Đầu ra:
{{
  "tasks": [
    {{
      "tool": "retrieval",
      "question": "Tìm ví dụ về chi phí cơ hội trong giáo trình môn Kinh tế vi mô."
    }},
    {{
      "tool": "base",
      "question": "Tự đưa ra một ví dụ khác về chi phí cơ hội."
    }}
  ]
}}

QUAN TRỌNG:
Chỉ xuất JSON hợp lệ. Không Markdown, không giải thích quá trình suy luận,
không viết nội dung ngoài JSON. Ký tự đầu tiên là "{{" và ký tự cuối cùng là "}}".
"""


def generate(model, history, question):
    body = {"model": model, "stream": False, "options": {"temperature": 0, "seed": 42},
            "messages": [{"role": "user", "content": build_prompt(history, question)}]}
    req = request.Request("http://localhost:11434/api/chat", data=dump(body).encode(),
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=600) as response:
            return json.load(response)["message"]["content"]
    except error.HTTPError as exc:
        raise SystemExit(f"Ollama HTTP {exc.code}: {exc.read().decode(errors='replace')}")
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"Lỗi khi gọi Ollama: {exc}")


ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def score_tasks(encoder, gold, predicted):
    texts = [task["question"] for task in gold + predicted]
    vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    count = len(gold)
    return [max(-1.0, min(1.0, float(a @ b)))
            for a, b in zip(vectors[:count], vectors[count:])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("test.jsonl"))
    parser.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--limit", type=int, help="Chạy N mẫu đầu")
    parser.add_argument("--out", type=Path, help="Tên CSV mới")
    args = parser.parse_args()
    cases = load_cases(args.data)
    if args.limit is None and len(cases) != 200:
        parser.error(f"Chờ 200 mẫu, đọc được {len(cases)}")
    if args.limit is not None and not 1 <= args.limit <= len(cases):
        parser.error("--limit nằm ngoài phạm vi dữ liệu")
    cases = cases[:args.limit]
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ENCODER, device="cpu")
    out = args.out or Path(datetime.now().strftime("qwen_goc_%Y%m%d_%H%M%S.csv"))
    valid = routed = 0
    sample_scores = []
    with out.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "history", "question", "gold_tasks", "raw_output", "schema_ok",
                         "tool_sequence_ok", "task_cosines", "mean_cosine", "min_cosine", "parse_error"])
        for i, (history, question, gold) in enumerate(cases, 1):
            raw = generate(args.model, history, question)
            problem = ""
            try:
                predicted = parse_plan(raw)
            except (ValueError, TypeError) as exc:
                predicted, problem = [], str(exc)
            route_ok = [t["tool"] for t in predicted] == [t["tool"] for t in gold]
            valid += int(not problem)
            routed += int(route_ok)
            scores = score_tasks(encoder, gold, predicted) if route_ok else []
            mean = statistics.mean(scores) if scores else None
            if mean is not None:
                sample_scores.append(mean)
            writer.writerow([i, dump(history), question, dump(gold), raw, int(not problem),
                             int(route_ok), dump(scores), mean, min(scores, default=None), problem])
            f.flush()
            print(f"[{i}/{len(cases)}] đúng chuỗi công cụ: {routed}/{i}", flush=True)
    print(f"CSV: {out.resolve()}")
    print(f"Đúng schema: {valid}/{len(cases)} ({valid/len(cases):.1%})")
    print(f"Đúng chuỗi công cụ / đủ điều kiện đo cosine: {routed}/{len(cases)} ({routed/len(cases):.1%})")
    if sample_scores:
        print(f"Cosine trung bình: {statistics.mean(sample_scores):.6f}")
        print(f"Độ lệch chuẩn tổng thể (ddof=0): {statistics.pstdev(sample_scores):.6f}")
    else:
        print("Không có mẫu đủ điều kiện để tính trung bình và độ lệch chuẩn.")


if __name__ == "__main__":
    main()
