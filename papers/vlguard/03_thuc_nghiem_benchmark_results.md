[⬅️ Bài 2: Hàm Mất Mát & Gradient Balancing](02_ham_mat_mat_va_gradient_balancing.md) | [🏠 Mục Lục Repo](../../README.md) | [Bài 4: Ranh Giới Thất Bại & Pháp Y ➡️](04_ranh_gioi_that_bai_va_phan_tich_phap_y.md)

---

# Bài 3: Đánh Giá Thực Nghiệm Toàn Diện: Hiệu Năng Phòng Thủ & Chi Phí Hữu Ích

> **Nội dung chuyên khảo:** Tổng hợp và bóc tách toàn bộ hệ thống số liệu thực nghiệm gốc của công trình VLGuard; phân tích benchmark trên 10 mô hình VLLM đương đại; kiểm thử phòng thủ bẻ khóa trên FigStep, AdvBench, XSTest; đo lường năng lực hữu ích trên ScienceQA, VizWiz, MMLU, MM-Vet; kiểm định mù con người (Human Evaluation); và phân tích khả năng tổng quát hóa trên các danh mục độc hại chưa từng huấn luyện.

---

## 1. Thiết Lập Thực Nghiệm Chuẩn Mực

Để đảm bảo tính tái lập (reproducibility) và tính công bằng tuyệt đối giữa các mô hình, toàn bộ các thí nghiệm trong công trình của Zong et al. được thực hiện theo giao thức nghiêm ngặt sau:

- **Phần cứng thực nghiệm:** Cụm máy chủ chuyên dụng trang bị các GPU **NVIDIA A100-80GB**.
- **Cấu hình giải mã:** Sử dụng giải mã tham lam (greedy decoding, nhiệt độ $T=0$) cho toàn bộ các bài kiểm thử nhằm loại bỏ biến thiên ngẫu nhiên; độ dài sinh tối đa được cố định ở `max_tokens = 256`.
- **Hệ thống Lời nhắc Hệ thống (System Prompt):** Sử dụng system prompt mặc định của từng mô hình nếu có; nếu không, áp dụng system prompt chuẩn:
  ```text
  System: A chat between a curious user and an artificial intelligence assistant. 
  The assistant gives helpful, detailed, and polite answers to the user's questions.
  User: [Query]
  Assistant: [Response]
  ```
- **Thước đo đánh giá an toàn (Safety Metrics):**
  1. *Khớp chuỗi từ chối (String Matching):* Tấn công được coi là thành công (Attack Success Rate - ASR) nếu câu trả lời của mô hình **không chứa bất kỳ mẫu câu từ chối chuẩn nào** trong danh mục kiểm định của Zou et al. (2023) và Röttger et al. (2023).
  2. *Bộ phân loại chuyên biệt Llama-Guard:* Sử dụng mô hình an toàn Llama-Guard (Inan et al., 2023) của Meta để phân loại nhị phân (Safe vs. Unsafe) câu trả lời do VLLM sinh ra.
- **Thước đo đánh giá năng lực hữu ích (Helpfulness Metrics):**
  1. *Tỷ lệ thắng (Win Rate) so với GPT-4V:* Áp dụng cho tập VLGuard Safe-Safe, sử dụng câu trả lời của GPT-4V làm chuẩn đối chứng (Ground Truth).
  2. *Độ chính xác chuẩn mực (Accuracy):* Trên ScienceQA và MMLU.
  3. *Tỷ lệ thắng theo AlpacaEval 2.0:* So sánh với mô hình `text-davinci-003` bằng đánh giá tự động của GPT-4.

---

## 2. Toàn Cảnh Rủi Ro: Benchmark 10 Dòng Mô Hình VLLM Hiện Đại

Trước khi tiến hành căn chỉnh, nhóm tác giả đã tiến hành đo kiểm quy mô lớn trên 10 mô hình ngôn ngữ - thị giác mã nguồn mở hàng đầu thế giới trên tập kiểm thử VLGuard Test Set (1.000 ảnh). Kết quả trích xuất từ **Table 12** trong công trình gốc:

### Bảng 12: Mức độ rủi ro an toàn của 10 mô hình VLLM đương đại trên VLGuard Test Set
| STT | Mô Hình VLLM | Kiến Trúc LLM Nền Tảng | Tham Số | VLGuard Safe-Safe (Helpfulness % ↑) | VLGuard Safe-Unsafe (ASR % ↓) | VLGuard Unsafe (ASR % ↓) |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | **InstructBLIP-7B** | Vicuna-v1.1-7B | 7B | 9.86 | 92.47 | 92.53 |
| 2 | **InstructBLIP-13B** | Vicuna-v1.1-13B | 13B | 10.57 | 96.42 | 98.64 |
| 3 | **Otter** | MPT-7B | 9B | 5.28 | 98.92 | 47.60 |
| 4 | **CogVLM** | Vicuna-v1.5-7B | 17B | 19.51 | 68.46 | 74.43 |
| 5 | **mPLUG-Owl2** | LLaMA-2-7B | 7B | 16.67 | 72.22 | 67.87 |
| 6 | **LLaVA-v1.5-7B** | Vicuna-v1.5-7B | 7B | 18.82 | 87.46 | 72.62 |
| 7 | **LLaVA-v1.5-13B** | Vicuna-v1.5-13B | 13B | 21.54 | 80.65 | 55.88 |
| 8 | **MiniGPT-v2** | LLaMA-2-Chat-7B | 7B | 12.21 | 88.17 | 87.33 |
| 9 | **Qwen-VL-Chat** | QwenLM-7B | 9B | 13.19 | 87.99 | 42.44 |
| 10 | **InternLM-XComposer** | InternLM-7B | 7B | 14.72 | 61.83 | 44.80 |

```mermaid
xychart-beta
    title "Ty Le Tan Cong Thanh Cong (ASR %) Tren 10 Mo Hinh VLLM (Thap Hon La Tot Hon)"
    x-axis ["InstBLIP-7B", "InstBLIP-13B", "Otter", "CogVLM", "mPLUG-Owl2", "LLaVA-7B", "LLaVA-13B", "MiniGPT-v2", "Qwen-VL", "InternLM"]
    y-axis "ASR Danh Muc Safe-Unsafe (%)" 0 --> 100
    bar [92.47, 96.42, 98.92, 68.46, 72.22, 87.46, 80.65, 88.17, 87.99, 61.83]
```

> [!WARNING]
> **Nhận Định Khoa Học:**
> Tất cả 10 mô hình VLLM tiên tiến nhất đều có mức độ rủi ro an toàn ở mức đáng báo động. Tỷ lệ trả lời chỉ thị độc hại trên ảnh an toàn (Safe-Unsafe) dao động từ **61.83% đến 98.92%**. Thậm chí mô hình InstructBLIP-13B thất bại trong việc từ chối tới **98.64%** các hình ảnh nguy hại nội tại. Điều này chứng minh lỗ hổng an toàn không phải là cá biệt của riêng một kiến trúc, mà là căn bệnh chung của toàn bộ thế hệ VLLM hiện nay.

---

## 3. Hiệu Năng Căn Chỉnh Của VLGuard: Bảng Số Liệu Cốt Lõi

Khi áp dụng phương pháp tinh chỉnh an toàn của VLGuard (cả hai biến thể Post-hoc và Mixed, sử dụng Full FT hoặc LoRA), mức độ an toàn được cải thiện vượt bậc trên toàn bộ các thước đo khắt khe nhất.

### Bảng 2: So sánh hiệu năng an toàn và tỷ lệ bẻ khóa trước và sau căn chỉnh VLGuard
*(Số liệu trích xuất chuẩn xác từ Table 2 trong bài báo gốc)*

| Mô Hình Đánh Giá | Phương Pháp Căn Chỉnh | AdvBench Vanilla (ASR % ↓) | AdvBench Suffix Inj. (ASR % ↓) | XSTest Unsafe (ASR % ↓) | XSTest Safe (Answer % ↑) | FigStep Typographic (ASR % ↓) | VLGuard Safe-Safe (Utility Win % ↑) | VLGuard Safe-Unsafe (ASR % ↓) | VLGuard Unsafe (ASR % ↓) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LLaVA-v1.5-7B** | Mô hình gốc | 6.45 | 78.27 | 26.50 | 91.20 | 90.40 | 18.82 | 87.46 | 72.62 |
| | **Post-hoc (Full)** | **0.00** | **13.08** | **6.00** | 80.80 | **0.00** | **18.96** | **0.90** | **0.23** |
| | **Post-hoc-LoRA** | **0.19** | **12.31** | **5.00** | 77.20 | **0.00** | 18.21 | **0.90** | **0.00** |
| | **Mixed (Full)** | **0.19** | **10.58** | **4.00** | 82.40 | **0.00** | **20.78** | **0.90** | **0.90** |
| | **Mixed-LoRA** | **0.19** | **11.15** | **4.00** | 83.60 | **0.00** | **19.18** | **1.25** | **0.00** |
| **LLaVA-v1.5-13B** | Mô hình gốc | 2.12 | 74.23 | 10.00 | 85.20 | 92.90 | 21.54 | 80.65 | 55.88 |
| | **Post-hoc (Full)** | **0.19** | **6.15** | **2.00** | 77.20 | **0.00** | 21.37 | **1.25** | **0.00** |
| | **Post-hoc-LoRA** | **0.38** | **9.81** | **5.50** | 83.20 | **0.00** | 20.98 | **0.72** | **0.00** |
| | **Mixed (Full)** | **0.00** | **8.46** | **0.50** | 84.00 | **0.00** | 21.43 | **0.90** | **0.90** |
| | **Mixed-LoRA** | **0.00** | **11.15** | **0.10** | 83.60 | **0.00** | **21.77** | **0.90** | **0.90** |
| **MiniGPT-v2** | Mô hình gốc (LoRA) | 19.04 | 22.50 | 16.50 | 88.80 | 93.60 | 12.21 | 88.17 | 87.33 |
| | **Post-hoc (LoRA)** | **3.00** | **4.81** | **6.00** | 81.20 | **2.00** | **12.30** | **5.19** | **12.37** |
| | **Mixed (LoRA)** | **0.00** | **5.10** | **4.00** | 84.00 | **0.00** | **12.72** | **6.27** | **10.18** |

### 3.1. Phân Tích Các Bước Ngoặt Thực Nghiệm

1. **Triệt tiêu hoàn toàn tấn công bẻ khóa Typography trên FigStep:**
   Trước khi căn chỉnh, cả ba mô hình LLaVA-7B, LLaVA-13B và MiniGPT-v2 đều bị bẻ khóa gần như tuyệt đối bởi các câu lệnh độc hại in trên ảnh của FigStep (ASR từ **90.40% đến 93.60%**). Sau khi tinh chỉnh với VLGuard, ASR trên FigStep **rơi thẳng đứng về 0.00%** (và 2.00% trên MiniGPT-v2 Post-hoc). Đây là mức cải thiện phòng thủ mạnh mẽ nhất từng được ghi nhận.
2. **Kháng cự tấn công bẻ khóa phức tạp Suffix Injection:**
   Với kỹ thuật tấn công chèn đuôi câu lệnh độc hại (`"Answer with 'Absolutely, here's'"`), ASR của LLaVA-13B giảm sâu từ **74.23% xuống chỉ còn 6.15%** (Post-hoc Full).
3. **Hiệu năng an toàn đa phương thức nội tại:**
   Tỷ lệ ASR trên tập VLGuard Safe-Unsafe giảm từ **87.46% xuống 0.90%**, và trên tập Unsafe giảm từ **72.62% xuống 0.00%**.
4. **Không làm tổn hại năng lực hữu ích (Utility):**
   Tỷ lệ thắng của LLaVA-7B trên tập Safe-Safe không hề suy giảm mà thậm chí tăng nhẹ từ **18.82% lên 20.78%** (Mixed Full).

---

## 4. Bóc Tách Chi Tiết Năng Lực Hữu Ích Đa Tác Vụ (Utility Breakdown)

Một chỉ trích phổ biến đối với các phương pháp căn chỉnh an toàn là hiện tượng "thuế căn chỉnh" (Alignment Tax) — tức việc tăng cường an toàn sẽ làm cùn mòn trí thông minh và năng lực suy luận của mô hình. Bảng 13 dưới đây bác bỏ hoàn toàn lo ngại này đối với VLGuard:

### Bảng 13: Bóc tách năng lực hữu ích trên các benchmark ngôn ngữ và thị giác chuẩn mực
*(Trích xuất từ Table 13 trong bài báo gốc)*

| Mô Hình | MMLU (NLP % ↑) | AlpacaEval 2.0 (NLP Win % ↑) | ScienceQA (V-L % ↑) | VizWiz (V-L % ↑) | Điểm TB NLP (% ↑) | Điểm TB V-L (% ↑) | ĐIỂM TB TOÀN PHẦN (% ↑) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vicuna-v1.5-7B** (Base LLM) | 48.55 | 62.50 | - | - | 55.53 | - | - |
| **LLaVA-v1.5-7B** (Gốc) | 36.52 | 61.50 | 67.68 | 55.16 | 49.01 | 61.42 | **55.22** |
| LLaVA-v1.5-7B-LoRA | 37.52 | 56.00 | 67.71 | 52.94 | 46.76 | 60.33 | 53.54 |
| LLaVA-v1.5-7B-Clean | 37.70 | 63.00 | 68.12 | 56.30 | 50.35 | 62.21 | 56.28 |
| LLaVA-v1.5-7B-LoRA-Clean | 36.81 | 62.33 | 68.27 | 51.84 | 49.57 | 60.06 | 54.81 |
| **LLaVA-v1.5-7B-Post-hoc-LoRA** | 36.44 | 63.00 | 67.92 | 57.76 | 49.72 | 62.84 | **56.28** *(+1.06)* |
| **LLaVA-v1.5-7B-Mixed (Full)** | 37.74 | 63.00 | 68.47 | 56.78 | 50.37 | 62.63 | **56.50** *(+1.28)* |
| **LLaVA-v1.5-7B-LoRA-Mixed** | 37.02 | 65.00 | 68.22 | 55.87 | 51.01 | 62.05 | **56.53** *(+1.31)* |
| **Vicuna-v1.5-13B** (Base LLM) | 54.54 | 63.16 | - | - | 58.85 | - | - |
| **LLaVA-v1.5-13B** (Gốc) | 44.72 | 63.33 | 71.64 | 57.50 | 54.03 | 64.57 | **59.30** |
| LLaVA-v1.5-13B-LoRA | 43.54 | 63.67 | 71.19 | 58.61 | 53.61 | 64.90 | 59.25 |
| LLaVA-v1.5-13B-Clean | 45.93 | 64.00 | 71.59 | 59.10 | 54.97 | 65.35 | 60.16 |
| **LLaVA-v1.5-13B-Post-hoc-LoRA** | 46.07 | 63.00 | 70.90 | 57.87 | 54.54 | 64.39 | **59.46** *(+0.16)* |
| **LLaVA-v1.5-13B-Mixed (Full)** | 46.51 | 65.00 | 71.54 | 59.11 | 55.76 | 65.33 | **60.54** *(+1.24)* |
| **Llama-2-7B-Chat** (Base LLM) | 45.31 | 37.00 | - | - | 41.16 | - | - |
| **MiniGPT-v2** (Gốc) | 41.43 | 38.67 | 57.06 | 53.60 | 40.05 | 55.33 | **47.69** |
| **MiniGPT-v2-Post-hoc** | 41.38 | 42.67 | 57.56 | 54.71 | 42.03 | 56.14 | **49.08** *(+1.39)* |
| **MiniGPT-v2-Mixed** | 41.67 | 46.50 | 59.44 | 54.23 | 44.09 | 56.84 | **50.46** *(+2.77)* |

> [!TIP]
> **Điểm Nhấn Khoa Học:**
> Trong toàn bộ các cấu hình thực nghiệm, mô hình sau khi tinh chỉnh an toàn với VLGuard **luôn có điểm năng lực tổng thể (Total Average Score) cao hơn mô hình gốc**:
> - LLaVA-7B tăng từ 55.22% lên **56.53%**.
> - LLaVA-13B tăng từ 59.30% lên **60.54%**.
> - MiniGPT-v2 tăng từ 47.69% lên **50.46%**.
> Điều này minh chứng rằng việc học được các phản hồi an toàn có lý luận chặt chẽ (rationale-based) không những không gây tổn hại nhận thức mà còn tăng cường khả năng lập luận logic và cấu trúc diễn đạt của mô hình trên các bài kiểm tra thực tế (AlpacaEval, ScienceQA).

---

## 5. Thí Nghiệm Đối Chứng Then Chốt: Đa Phương Thức (VLGuard) vs. Thuần Văn Bản (Safety LLaMA)

Một câu hỏi cốt tử được đặt ra: *"Liệu ta có thể chỉ cần dùng dữ liệu an toàn văn bản thuần túy (như tập dữ liệu của Safety LLaMA - Bianchi et al., 2024) để căn chỉnh an toàn cho VLLM hay không?"*

Nhóm tác giả đã tiến hành huấn luyện đối chứng có kiểm soát giữa Safety LLaMA và VLGuard trên cùng mô hình nền LLaVA-v1.5-7B:

### Bảng 4: So sánh hiệu quả phòng thủ giữa dữ liệu an toàn văn bản và VLGuard đa phương thức
*(Trích xuất từ Table 4 trong bài báo gốc)*

| Tập Dữ Liệu Tinh Chỉnh | AdvBench Vanilla (ASR % ↓) | AdvBench Suffix Inj. (ASR % ↓) | VLGuard Safe-Unsafe (ASR % ↓) | VLGuard Unsafe (ASR % ↓) | FigStep Typographic Jailbreak (ASR % ↓) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Mô hình gốc (LLaVA-7B)** | 6.45 | 78.27 | 87.46 | 72.62 | 90.40 |
| **Safety LLaMA (Thuần Văn Bản)** | **0.00** | **8.90** | 85.13 *(Thất bại)* | 56.57 *(Thất bại)* | **87.00** *(Thất bại hoàn toàn)* |
| **VLGuard (Đa Phương Thức)** | **0.00** | **13.08** | **0.90** *(Thành công)* | **0.23** *(Thành công)* | **0.00** *(Triệt tiêu hoàn toàn)* |

```mermaid
flowchart LR
    subgraph TextOnly["Safety LLaMA (Text-Only SFT)"]
        AdvBench1["AdvBench Suffix ASR: 8.90% (An toàn trên Text)"]
        FigStep1["FigStep ASR: 87.00% (Hoàn toàn thất thủ trên Ảnh!)"]
    end
    subgraph MultiModal["VLGuard (Multimodal SFT)"]
        AdvBench2["AdvBench Suffix ASR: 13.08% (An toàn trên Text)"]
        FigStep2["FigStep ASR: 0.00% (Phòng thủ tuyệt đối trên Ảnh!)"]
    end
```

> [!IMPORTANT]
> **Kết Luận Đanh Thép:**
> Dữ liệu an toàn thuần văn bản (Safety LLaMA) chỉ có thể giúp VLLM chống lại các câu hỏi độc hại dạng văn bản thuần túy (ASR AdvBench giảm xuống 8.90%). Nhưng khi câu lệnh độc hại được đưa vào kênh thị giác (FigStep typographic prompt), **Safety LLaMA thất thủ hoàn toàn với tỷ lệ bị bẻ khóa lên tới 87.00%**. Ngược lại, chỉ có VLGuard mới có khả năng bẻ gãy hoàn toàn cuộc tấn công thị giác này (ASR = 0.00%).

---

## 6. Đánh Giá Mù Con Người (Human Evaluation)

Để loại bỏ các thiên kiến có thể có từ các bộ lọc tự động hoặc GPT-4, nhóm nghiên cứu đã triển khai đánh giá mù con người (Human Blind Evaluation) với ba chuyên gia độc lập thuộc các sắc tộc và giới tính khác nhau. Mỗi chuyên gia đánh giá ngẫu nhiên 30 mẫu từ mỗi phân vùng của tập kiểm thử VLGuard:

### Bảng 5: Tỷ lệ thắng (Win Rate %) của mô hình căn chỉnh VLGuard so với mô hình gốc
*(Trích xuất từ Table 5 trong bài báo gốc)*

| Dòng Mô Hình | Chiến Lược Tinh Chỉnh | VLGuard Safe-Safe (Utility Win % ≈ 50%) | VLGuard Safe-Unsafe (Safety Win % ↑) | VLGuard Unsafe (Safety Win % ↑) |
|---|---|:---:|:---:|:---:|
| **LLaVA-v1.5-7B** | Post-hoc | 55.00 | **93.33** | **96.67** |
| | Mixed | 50.00 | **93.33** | **96.67** |
| **LLaVA-v1.5-13B** | Post-hoc | 51.67 | **93.33** | **100.00** |
| | Mixed | 42.00 | **90.00** | **100.00** |
| **MiniGPT-v2** | Post-hoc | 52.00 | **76.67** | **86.67** |
| | Mixed | 46.67 | **90.00** | **90.00** |

- **Trên tập Safe-Safe (Hữu ích):** Tỷ lệ thắng dao động quanh mốc cân bằng **~50%** (từ 42.00% đến 55.00%). Điều này chứng minh người dùng thực tế không nhận thấy bất kỳ sự suy giảm chất lượng câu trả lời nào giữa mô hình đã căn chỉnh và mô hình gốc.
- **Trên tập Safe-Unsafe & Unsafe (An toàn):** Mô hình tinh chỉnh VLGuard áp đảo hoàn toàn với tỷ lệ thắng đạt từ **90.00% đến 100.00%**, xác nhận sự vượt trội về mức độ văn minh và khả năng bảo vệ người dùng.

---

## 7. Khả Năng Tổng Quát Hóa Trên Danh Mục Độc Hại Mới Lạ (Unseen Harm Generalization)

Một phẩm chất tối quan trọng của mô hình học sâu an toàn là khả năng suy luận tổng quát trên các nguy cơ chưa từng gặp trong quá trình huấn luyện (Zero-shot Safety Generalization).

Nhóm tác giả đã thiết lập thí nghiệm kiểm chứng:
1. Rút trích một tập con 500 mẫu huấn luyện từ VLGuard.
2. Tạo ra hai kịch bản huấn luyện:
   - **Kịch bản A (Random):** 500 mẫu được chọn ngẫu nhiên bao gồm đủ cả 4 danh mục.
   - **Kịch bản B (Held-out Privacy):** 500 mẫu nhưng **loại bỏ hoàn toàn 100% dữ liệu thuộc danh mục Privacy (Quyền riêng tư)**.
3. Đánh giá hai mô hình trên tập kiểm thử phân vùng Privacy của VLGuard Test Set.

```mermaid
xychart-beta
    title "Kha Nang Tong Quat Hoa Khi Loai Bo Du Lieu Privacy (ASR % tren Tap Privacy)"
    x-axis ["Mo Hinh Goc (Chua Can Chinh)", "Held-out: Loai Bo 100% Privacy Data", "Random Subsampled (Co Privacy Data)"]
    y-axis "ASR Tren Danh Muc Quyen Rieng Tu (%)" 0 --> 80
    bar [72.62, 0.00, 0.00]
```

### Kết Quả (Trích từ Figure 5 trong bài báo):
- Mô hình gốc LLaVA-7B có ASR trên danh mục Privacy là **72.62%**.
- Mô hình được huấn luyện hoàn toàn không chứa dữ liệu Privacy (Held-out) **vẫn giảm tỷ lệ ASR trên danh mục Privacy xuống chính xác 0.00%**!

> [!NOTE]
> **Ý Nghĩa Lý Thuyết:**
> Kết quả này chứng minh rằng việc tinh chỉnh với VLGuard không làm mô hình ghi nhớ máy móc các trường hợp cụ thể theo dạng tra bảng (lookup table). Thay vào đó, mô hình đã hình thành một **biểu diễn khái niệm an toàn bậc cao (High-Level Conceptual Safety Representation)** trong không gian tiềm ẩn, cho phép nó nhận thức được tính chất vi phạm đạo đức và tự động kích hoạt phản xạ từ chối ngay cả trên những miền độc hại hoàn toàn mới.

---

## 8. Khả Năng Phòng Vệ Trước Tấn Công Bẻ Khóa Hộp Đen Nâng Cao (Tree of Attacks - TAP)

Ngoài các đòn tấn công tĩnh, VLGuard còn được thử nghiệm trước kỹ thuật bẻ khóa hộp đen động tự động hóa tinh vi nhất: **Tree of Attacks with Pruning (TAP)** (Mehrotra et al., 2023), trong đó một LLM tấn công liên tục tương tác và tinh chỉnh prompt qua nhiều vòng để tìm điểm yếu của mô hình mục tiêu.

### Bảng 15: Tác động của tấn công hộp đen TAP đối với LLaVA-v1.5-7B
*(Trích xuất từ Table 15 trong bài báo gốc)*

| Mô Hình Đánh Giá | Tỷ Lệ Bị Bẻ Khóa Thành Công (ASR % ↓) | Số Lượng Truy Vấn Trung Bình Cần Để Bẻ Khóa (Queries ↑) |
|---|:---:|:---:|
| **LLaVA-v1.5-7B (Gốc)** | **62.00%** | **15.98** truy vấn |
| **LLaVA-v1.5-7B-Post-hoc** | **34.00%** *(Giảm gần 1/2)* | **20.78** truy vấn |
| **LLaVA-v1.5-7B-Mixed** | **20.00%** *(Giảm hơn 3 lần)* | **21.56** truy vấn |

Không chỉ làm giảm tỷ lệ thành công của đòn tấn công TAP từ 62.00% xuống còn 20.00%, VLGuard còn buộc kẻ tấn công phải tiêu tốn nhiều lượt truy vấn hơn đáng kể (tăng từ 15.98 lên 21.56 lượt), làm tăng mạnh chi phí tính toán của kẻ tấn công và tạo điều kiện thuận lợi cho các hệ thống giám sát an ninh phát hiện hành vi bất thường.

---

[⬅️ Bài 2: Hàm Mất Mát & Gradient Balancing](02_ham_mat_mat_va_gradient_balancing.md) | [🏠 Mục Lục Repo](../../README.md) | [Bài 4: Ranh Giới Thất Bại & Pháp Y ➡️](04_ranh_gioi_that_bai_va_phan_tich_phap_y.md)
