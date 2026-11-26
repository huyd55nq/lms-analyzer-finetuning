# Tinh chỉnh Qwen2.5-7B-Instruct cho Analyzer trên LMS

Analyzer tiếp nhận câu hỏi tiếng Việt và lịch sử hội thoại, sau đó tạo kế hoạch JSON cho bộ thực thi của chatbot. Mỗi tác vụ xác định một công cụ và nội dung yêu cầu. Ba công cụ gồm `base` cho kiến thức chung, `retrieval` cho học liệu nội bộ và `database` cho dữ liệu học tập trên LMS.

## Dữ liệu và mã nguồn

| Tệp | Nội dung |
| --- | --- |
| `data_3420.jsonl` | Bộ dữ liệu dùng cho quá trình tinh chỉnh. Bài báo ghi nhận 3.430 mẫu, chia thành 3.087 mẫu huấn luyện và 343 mẫu xác thực. |
| `Finetunellm .ipynb` | Notebook huấn luyện trên Google Colab, chọn checkpoint và xuất mô hình GGUF. |
| [test_200.jsonl](test_200.jsonl) | 200 câu hỏi kiểm thử cùng kế hoạch tham chiếu. |
| [prompt_daydu.txt](prompt_daydu.txt), [prompt_rutgon.txt](prompt_rutgon.txt) | Nội dung chỉ dẫn đầy đủ và rút gọn. |
| [test_C1.py](test_C1.py), [test_C2.py](test_C2.py), [test_C3.py](test_C3.py) | Mã tạo đầu vào, gọi mô hình và đánh giá từng cấu hình. |

Ví dụ kế hoạch cho câu hỏi “Điểm môn Toán của tôi là bao nhiêu?”:

```json
{"tasks":[{"tool":"database","question":"Điểm môn Toán của người dùng hiện tại là bao nhiêu?"}]}
```

## Huấn luyện và kiểm thử

Mở notebook trong Google Colab, chọn GPU và chỉnh đường dẫn dữ liệu, checkpoint và thư mục xuất theo vị trí trên Google Drive. Mô hình được tinh chỉnh bằng QLoRA 4-bit trong hai epoch, với hàm mất mát chỉ tính trên phần trả lời. Checkpoint được chọn ở bước 386. Bản dùng cho suy luận được xuất ở định dạng GGUF Q4_K_M.

Để kiểm thử trên Windows, khởi động Ollama và dùng `ollama list` để kiểm tra tên mô hình gốc và mô hình đã tinh chỉnh trên máy. Trong môi trường Python của dự án, cài bộ mã hóa dùng để chấm cosine:

```powershell
python -m pip install sentence-transformers
```

Chạy từ thư mục chứa các tệp dữ liệu và mã kiểm thử. Dùng tên CSV mới cho mỗi lần chạy.

```powershell
python .\test_C1.py --data .\test_200.jsonl --out .\lan_moi_C1.csv
python .\test_C2.py --data .\test_200.jsonl --out .\lan_moi_C2.csv
python .\test_C3.py --data .\test_200.jsonl --out .\lan_moi_C3.csv
```

## Kết quả trên 200 câu hỏi

| Cấu hình | JSON đúng cấu trúc | Đúng chuỗi công cụ | Cosine trung bình ± độ lệch chuẩn |
| --- | ---: | ---: | ---: |
| C1 · Mô hình gốc, chỉ dẫn đầy đủ | 200/200 (100%) | 155/200 (77,5%) | 0,821891 ± 0,122463 |
| C2 · Mô hình gốc, chỉ dẫn rút gọn | 68/200 (34,0%) | 53/200 (26,5%) | 0,792964 ± 0,139631 |
| C3 · Mô hình tinh chỉnh, đầu vào tối giản | 200/200 (100%) | 153/200 (76,5%) | 0,815864 ± 0,117118 |

Một kế hoạch đúng chuỗi công cụ khi khớp cả số lượng, tên và thứ tự công cụ với tham chiếu. Điểm cosine chỉ được tính trên các mẫu đạt tiêu chí này. Với mỗi câu hỏi, bộ chấm lấy trung bình cosine của các cặp tác vụ cùng vị trí rồi tổng hợp theo câu hỏi. Số mẫu dùng để tính cosine lần lượt là 155, 53 và 153.

Trong phép đo thời gian riêng, Analyzer dùng mô hình tinh chỉnh mất trung bình 1,545 giây cho câu hỏi một bước và 4,062 giây cho câu hỏi hai bước. Các giá trị tương ứng của mô hình gốc với chỉ dẫn đầy đủ là 2,874 và 5,448 giây.
