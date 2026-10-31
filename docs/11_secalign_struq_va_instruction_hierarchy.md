[⬅️ Chương trước: Gemini Adversarial Training](10_gemini_adversarial_training_google_deepmind.md) | [🏠 Mục Lục](../README.md) | [Chương tiếp: Circuit Breakers ➡️](12_circuit_breakers_representation_rerouting.md)

---

# Chương 11: Chuyên Đề Hội Nghị A\* — StruQ, SecAlign, ISE và Phân Cấp Chỉ Thị

> **Công trình trọng tâm:**
> - *StruQ: Defending Against Prompt Injection with Structured Queries* — USENIX Security 2025, arXiv:2402.06363.
> - *SecAlign: Defending Against Prompt Injection with Preference Optimization* — ACM CCS 2025, arXiv:2410.05451.
> - *Instructional Segment Embedding* — ICLR 2025, arXiv:2410.09102.
> - *Meta SecAlign* — arXiv:2507.02735 (preprint).
>
> **Mức xác minh:** venue đã xác minh trên trang hội nghị/ACM. Phần phương pháp được suy ra từ phần mô tả của bài báo trong các nguồn đọc được; **công thức DPO dưới đây là công thức DPO chuẩn** áp dụng cho dữ liệu của SecAlign, không phải trích nguyên văn từ bài. Con số ASR lấy từ nguồn thứ cấp, cần đối chiếu bài gốc.

---

## 1. Tại Sao Đưa Vào Chuyên Khảo VPI

Các phương pháp này đánh giá trên **văn bản**, nhưng chúng định nghĩa chuẩn mực mà mọi phòng vệ model-based cho VPI phải so sánh: tách *kênh lệnh* và *kênh dữ liệu* rồi huấn luyện mô hình tôn trọng ranh giới đó. Một VLM nhận ảnh chứa chữ chính là trường hợp kênh dữ liệu (pixel) mang chỉ thị. Hiểu StruQ/SecAlign giúp thấy rõ VLGuard (Chương 1) và A3T (Chương 4) khác ở điểm nào.

## 2. StruQ: Truy Vấn Có Cấu Trúc

Ý tưởng gồm hai phần:

1. **Front-end an toàn:** dùng các token phân cách dành riêng (reserved separator tokens) cho phần lệnh và phần dữ liệu, và lọc mọi chuỗi phân cách xuất hiện trong dữ liệu để kẻ tấn công không giả mạo ranh giới.
2. **Structured instruction tuning:** huấn luyện trên các ví dụ chèn lệnh mô phỏng sao cho mô hình chỉ làm theo lệnh ở phần lệnh và bỏ qua lệnh trong phần dữ liệu.

Kết quả theo nguồn: tấn công thủ công (không tối ưu) thành công dưới 2% trên Llama và Mistral ở bản arXiv v2; tấn công dựa trên tối ưu như GCG và TAP khó hơn. Một bài tóm tắt công khai của nhóm tác giả ghi StruQ có ASR tối đa 45% trước các tấn công tối ưu, và SecAlign giảm xuống 8% (nguồn: [AI Hub, 05/2025](https://aihub.org/2025/05/06/defending-against-prompt-injection-with-structured-queries-struq-and-preference-optimization-secalign/)).

## 3. SecAlign: Tối Ưu Hóa Ưu Tiên

### 3.1. Xây dựng dữ liệu

Từ một tập tinh chỉnh chỉ thị sạch (Cleaned Alpaca), mỗi mẫu được tạo thành một mẫu bị chèn lệnh bằng cách nối một chỉ thị ngẫu nhiên thứ hai vào phần dữ liệu. Với mỗi đầu vào bị chèn $x$, ta có hai phản hồi:

- $y_w$ (**chọn**, secure): phản hồi theo chỉ thị hợp lệ $U$.
- $y_l$ (**loại**, insecure): phản hồi theo chỉ thị chèn $I$.

Việc dựng dữ liệu là phép nối chuỗi, không cần nhân công.

### 3.2. Hàm mục tiêu

Dùng DPO (Direct Preference Optimization) với mô hình tham chiếu $\pi_{\text{ref}}$ và hệ số $\beta > 0$:

$$
\mathcal{L}_{\text{SecAlign}}(\theta) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

So với SFT của StruQ (chỉ tăng xác suất $y_w$), mục tiêu này **đẩy xuống xác suất $y_l$** một cách tường minh, tạo khoảng cách xác suất lớn hơn giữa phản hồi đúng và phản hồi bị chèn. Bài báo cho rằng nhờ vậy SecAlign bền hơn StruQ trước các tấn công tối ưu mạnh mà nó không thấy khi huấn luyện, và giảm ASR của tấn công mạnh nhất được thử xuống 8% trên Llama3-8B-Instruct mà không làm giảm tiện ích (theo tóm tắt).

### 3.3. Đánh giá độc lập

Một đánh giá bên ngoài (*A Critical Evaluation of Defenses against Prompt Injection Attacks*, arXiv:2505.18333) báo cáo GCG vẫn đạt 0.24 và 0.72 trên Llama-3-8B-Instruct-SecAlign ở hai thiết lập, cùng mức giảm tiện ích 0.17 và 0.11. Khác biệt với bài gốc có thể do thiết lập đánh giá; đây là lý do **luôn cần đọc cả phản biện độc lập**. Một nguồn khác lưu ý SecAlign vẫn hạn chế trước các đối thủ nhận biết kiến trúc và tấn công dựa trên học tăng cường.

## 4. ISE: Phân Cấp Chỉ Thị Ở Mức Kiến Trúc

Thay vì chỉ huấn luyện, ISE thêm một **embedding phân đoạn** học được cho mỗi token, gắn nhãn vai trò (mặc định bốn vai trò: system, user, data, output) và cộng vào embedding token, theo kiểu segment embedding của BERT. Tác giả báo cáo tăng độ chính xác bền vững trung bình tới 15.75% và 18.68% trên hai benchmark (Structured Query và Instruction Hierarchy), và cải thiện khả năng tuân thủ chỉ thị tới 4.1% trên AlpacaEval (theo trang tóm tắt ICLR).

## 5. Meta SecAlign và Instruction Hierarchy

- **Meta SecAlign (arXiv:2507.02735):** phiên bản open-weight (8B, 70B) áp dụng công thức SecAlign cho mô hình nền tảng Llama; tác giả báo cáo mô hình 70B đạt đánh đổi bảo mật–tiện ích mới cho mô hình mã nguồn mở. Đây là kết quả của tác giả, không phải bảo đảm miễn nhiễm.
- **The Instruction Hierarchy (OpenAI, arXiv:2404.13208):** nền tảng ý tưởng: chỉ thị từ nguồn đặc quyền cao (system) thắng chỉ thị từ nguồn thấp (dữ liệu công cụ). SecAlign và ISE đều là cách hiện thực khác nhau của ý tưởng này.

## 6. So Sánh Với Các Chương Trước

| Tiêu chí | StruQ | SecAlign | ISE | VLGuard (Ch. 1) | WARD A3T (Ch. 4) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Cách huấn luyện | SFT có cấu trúc | DPO | Học embedding vai trò + huấn luyện | SFT trộn dữ liệu an toàn | RL đồng tiến hóa với attacker |
| Phương thức đầu vào | Văn bản | Văn bản | Văn bản | Ảnh + văn bản | Ảnh chụp màn hình web |
| Đổi kiến trúc | Không (thêm token) | Không | **Có** (thêm embedding) | Không | Không (mô hình guard riêng) |
| Truy cập cần | Trọng số | Trọng số | Trọng số + kiến trúc | Trọng số | Chỉ cần chạy guard song song |

**Khoảng trống nghiên cứu:** chưa thấy bản mở rộng SecAlign cho kênh ảnh, nơi "ranh giới kênh lệnh/dữ liệu" không thể được đánh dấu bằng token phân cách vì chỉ thị nằm trong pixel. Đây là đề tài luận văn tiềm năng.

## Câu Hỏi Ôn Tập

1. Vì sao DPO có thể bền hơn SFT khi đối mặt tấn công tối ưu? Giải thích bằng số hạng $\log \pi_\theta(y_l \mid x)$ trong $\mathcal{L}_{\text{SecAlign}}$.
2. Token phân cách của StruQ không áp dụng được cho VPI từ ảnh vì sao? Đề xuất một cách đánh dấu ranh giới cho ảnh.
3. Nêu lý do tại sao phải đọc cả đánh giá độc lập trước khi tin vào một ASR do tác giả báo cáo.

## Tài Liệu Tham Khảo

- Chen, S., Piet, J., Sitawarin, C., Wagner, D. *StruQ*. USENIX Security 2025. arXiv:2402.06363. https://www.usenix.org/conference/usenixsecurity25/presentation/chen-sizhe
- Chen, S., Zharmagambetov, A., Mahloujifar, S., Chaudhuri, K., Wagner, D., Guo, C. *SecAlign*. ACM CCS 2025. arXiv:2410.05451. https://doi.org/10.1145/3719027.3744836
- Wu, T. và cộng sự. *Instructional Segment Embedding*. ICLR 2025. arXiv:2410.09102. https://iclr.cc/virtual/2025/poster/28101
- *Meta SecAlign*. arXiv:2507.02735, 2025.
- Wallace, E. và cộng sự (OpenAI). *The Instruction Hierarchy*. arXiv:2404.13208, 2024.
- *A Critical Evaluation of Defenses against Prompt Injection Attacks*. arXiv:2505.18333, 2025.

---

[⬅️ Chương trước: Gemini Adversarial Training](10_gemini_adversarial_training_google_deepmind.md) | [🏠 Mục Lục](../README.md) | [Chương tiếp: Circuit Breakers ➡️](12_circuit_breakers_representation_rerouting.md)
