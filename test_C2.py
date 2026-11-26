
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
    return f"""Bạn là Analyzer hỗ trợ học tập trên Moodle. Dựa vào lịch sử và câu hỏi hiện tại, lập kế hoạch cho Executor.

LỊCH SỬ HỘI THOẠI GẦN NHẤT
{history_text}

CÂU HỎI HIỆN TẠI
{question}

Chọn công cụ theo nguồn thông tin cần dùng:
- database: tra dữ liệu Moodle như điểm, hạn nộp, lượt làm quiz, trạng thái nộp bài hoặc nhận xét của giảng viên.
- retrieval: tìm, trích hoặc đối chiếu nội dung trong slide, giáo trình, đề cương và tài liệu nội bộ.
- base: giải thích kiến thức, tính toán, viết, dịch hoặc suy luận từ thông tin đã có.

Dùng lịch sử để làm rõ đối tượng được nhắc đến và ưu tiên yêu cầu hiện tại. Giữ đúng đối tượng, thời gian và điều kiện lọc. Mỗi tác vụ dùng một công cụ. Giữ các điều kiện của cùng một yêu cầu trong một tác vụ. Tách các yêu cầu có kết quả riêng theo thứ tự người dùng hỏi. Nếu tác vụ sau cần kết quả từ tác vụ trước, đặt bước cung cấp dữ liệu lên trước.

Chỉ xuất JSON hợp lệ theo mẫu:
{{"tasks":[{{"tool":"<tên công cụ>","question":"<yêu cầu cụ thể gửi cho công cụ>"}}]}}
Mỗi trường tool nhận đúng một giá trị: base, retrieval hoặc database. Không trả lời trực tiếp câu hỏi, không kèm Markdown hay giải thích."""


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
