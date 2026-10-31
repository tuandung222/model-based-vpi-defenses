[⬅️ Chương trước: Thư Mục Tài Liệu Mở Rộng](09_tai_lieu_mo_rong_top_tier.md) | [🏠 Mục Lục](../README.md) | [Chương tiếp: SecAlign, StruQ & Instruction Hierarchy ➡️](11_secalign_struq_va_instruction_hierarchy.md)

---

# Chương 10: Chuyên Đề Big Tech — Tinh Chỉnh Đối Kháng Gemini Chống Chèn Lệnh Gián Tiếp (Google DeepMind)

> **Công trình:** *Lessons from Defending Gemini Against Indirect Prompt Injections* (Google DeepMind, 05/2025), arXiv:2505.14534. Bài viết kèm theo trên blog Google DeepMind: [Advancing Gemini's security safeguards](https://deepmind.google/blog/advancing-geminis-security-safeguards/).
>
> **Mức xác minh:** Chương này dựa trên phần tóm tắt, trang blog và các bản tóm tắt thứ cấp. Môi trường biên soạn không truy cập được toàn văn arXiv, nên **các bảng số liệu theo từng kiểu tấn công chưa được đối chiếu với bài gốc**. Những con số dưới đây là con số xuất hiện trong nguồn đọc được và được ghi chú nguồn.

---

## 1. Vì Sao Chương Này Đáng Học

Đây là tài liệu hiếm hoi từ một phòng nghiên cứu Big Tech mô tả **một phòng vệ model-based đã được triển khai trên mô hình sản phẩm** (Gemini 2.5), thay vì một nguyên mẫu học thuật. Có ba điều đáng học:

1. Phòng vệ model-based chính là **tinh chỉnh đối kháng** (adversarial fine-tuning), cùng họ với VLGuard (Chương 1) và A3T của WARD (Chương 4), nhưng với quy mô và quy trình red-teaming của một nhà phát triển mô hình.
2. Bài viết đưa ra **bằng chứng thực nghiệm rằng phòng vệ tĩnh suy giảm trước tấn công thích ứng**, điều mà các bài báo học thuật thường không có điều kiện kiểm chứng.
3. Kết luận của bài viết là **phòng thủ nhiều tầng**, khớp với kết luận ở [Chương 8](08_so_sanh_thuc_nghiem_va_ranh_gioi_that_bai.md).

## 2. Bài Toán và Mô Hình Đe Dọa

**Chèn lệnh gián tiếp** (Indirect Prompt Injection, IPI): kẻ tấn công không điều khiển hội thoại với mô hình, nhưng chèn chỉ thị vào dữ liệu mà mô hình đọc trong khi thực thi tác vụ (email, lịch, trang web, tài liệu). Mô hình có quyền gọi công cụ nên có thể bị ép thực hiện hành động ngoài ý định của người dùng, ví dụ gửi thông tin nhạy cảm ra ngoài.

Ký hiệu: $U$ là chỉ thị hợp lệ của người dùng, $D$ là dữ liệu do bên thứ ba kiểm soát chứa chỉ thị chèn $I$. Mô hình $\pi_\theta$ nhận $(U, D \oplus I)$ và mục tiêu của kẻ tấn công là $\pi_\theta$ thực thi $I$ thay vì $U$.

Bài viết đo bằng **tỷ lệ tấn công thành công** (Attack Success Rate, ASR) trên các kịch bản có công cụ (ví dụ email và lịch), dưới cả tấn công không thích ứng lẫn thích ứng.

## 3. Hai Thành Phần Chính của Phòng Vệ

### 3.1. Tinh Chỉnh Đối Kháng (Model-Based)

Theo trang blog và tóm tắt, nhóm huấn luyện Gemini 2.5 trên dữ liệu chứa các cuộc tấn công chèn lệnh được sinh bằng red-teaming tự động, để mô hình học bỏ qua chỉ thị chèn và tiếp tục theo $U$. Các điểm chính được nguồn thứ cấp nêu:

- Mô hình sau tinh chỉnh cho thấy giảm ASR trung bình khoảng **47%** trước các tấn công thích ứng so với Gemini 2.0, **kể cả trên một kịch bản lịch không có trong dữ liệu huấn luyện** (nguồn: bản tóm tắt trên [arXiv](https://arxiv.org/pdf/2505.14534) và [SecurityWeek](https://www.securityweek.com/google-deepmind-unveils-defense-against-indirect-prompt-injection-attacks/)).
- Các tác giả lập luận rằng tinh chỉnh đối kháng **không nhất thiết làm suy giảm năng lực chung** nếu có nỗ lực thiết kế dữ liệu cẩn thận; họ phản bác quan niệm rằng huấn luyện đối kháng luôn làm hỏng khả năng tuân thủ chỉ thị.

### 3.2. Phòng Vệ Bổ Trợ Ngoài Mô Hình

Gemini 2.0 dùng thêm các phòng vệ bên ngoài, trong đó "Warning defense" (nhắc mô hình cảnh giác với chỉ thị trong dữ liệu) được mô tả là một trong những phòng vệ bổ trợ hiệu quả. Điểm quan trọng: **kết hợp Warning với tinh chỉnh đối kháng cho tỷ lệ phòng vệ cao hơn dùng riêng lẻ từng cái** trước tấn công thích ứng.

## 4. Red-Teaming Thích Ứng: Phần Khó Nhất

Bài viết nhấn mạnh rằng đánh giá bằng tấn công cố định (không thích ứng) đánh giá cao phòng vệ. Họ vận hành một khung red-teaming tự động chạy liên tục chống lại các phiên bản Gemini cũ, hiện tại và tương lai, dùng nhiều thuật toán tấn công (trong đó có tìm kiếm cây TAP và beam search được nhắc tên ở nguồn thứ cấp).

Các con số xuất hiện trong nguồn thứ cấp (chưa đối chiếu bảng gốc, đọc như minh họa mức độ chứ không dùng để trích dẫn):

| Kịch bản / Tấn công | Gemini 2.0 | Gemini 2.5 (tinh chỉnh đối kháng) | Nguồn / Ghi chú |
|:---|:---:|:---:|:---|
| Email, TAP | 99.8% | 53.6% | Tóm tắt thứ cấp |
| Beam Search | khoảng 75% | khoảng 4% | Tóm tắt thứ cấp |
| Lịch, TAP, có Warning | 10.8% (2.0 + Warning) | 6.2% (2.5 + Warning) | Tóm tắt thứ cấp |

Nhận xét sư phạm: ngay cả sau huấn luyện đối kháng, TAP vẫn đạt hơn một nửa trong kịch bản email (53.6%). **Phòng vệ làm giảm ASR đáng kể nhưng không đưa nó về 0.**

## 5. Kết Luận Của Bài Viết (Theo Tóm Tắt)

- Không có một cơ chế phòng vệ đơn lẻ hiệu quả trước mọi tấn công; việc liệt kê toàn bộ không gian tấn công là không khả thi, nên không thể tuyên bố mô hình "thật sự bền vững".
- Các phòng vệ nền tảng như Spotlighting hay tự phản tư (self-reflection) mất tác dụng đáng kể trước tấn công thích ứng.
- Khuyến nghị phòng thủ phân tầng gồm mô hình, ứng dụng và hạ tầng phục vụ.

## 6. Liên Hệ và Hạn Chế

| Khía cạnh | Gemini adversarial training | Liên hệ trong chuyên khảo |
|:---|:---|:---|
| Cơ chế | Tinh chỉnh trên tấn công mô phỏng | VLGuard (Ch. 1): SFT dữ liệu an toàn; WARD A3T (Ch. 4): đồng tiến hóa với attacker |
| Quyền truy cập | Chỉ nhà phát triển mô hình | Không áp dụng cho người dùng API đóng (Tử huyệt 5, Ch. 8) |
| Đa phương thức | Bài viết tập trung IPI qua văn bản/công cụ; **không phải bài VPI từ ảnh** | Khoảng trống: chưa có báo cáo công khai tương đương cho VPI |
| Đánh giá | Tấn công thích ứng liên tục | Bài học cho mọi chương: đo ASR trên tập tĩnh là chưa đủ |

**Hạn chế khi dùng chương này:** các con số đến từ nguồn thứ cấp; cần đọc bảng gốc trước khi trích dẫn trong luận văn.

## Câu Hỏi Ôn Tập

1. Vì sao đánh giá bằng tấn công cố định có thể đánh giá cao hiệu quả phòng vệ? Hãy đề xuất một quy trình đánh giá thích ứng cho một phòng vệ VLM.
2. So sánh tinh chỉnh đối kháng của Gemini với A3T của WARD: điểm giống và khác về nguồn tấn công huấn luyện.
3. Nếu tinh chỉnh đối kháng chỉ giảm ASR khoảng một nửa, vì sao vẫn cần các lớp phòng vệ khác?

## Tài Liệu Tham Khảo

- Shi, C. và cộng sự (Google DeepMind). *Lessons from Defending Gemini Against Indirect Prompt Injections*. arXiv:2505.14534, 2025. https://arxiv.org/abs/2505.14534
- Google DeepMind. *Advancing Gemini's security safeguards*, 20/05/2025. https://deepmind.google/blog/advancing-geminis-security-safeguards/
- Hines, K. và cộng sự (Microsoft). *Defending Against Indirect Prompt Injection Attacks With Spotlighting*. arXiv:2403.14720, 2024.

---

[⬅️ Chương trước: Thư Mục Tài Liệu Mở Rộng](09_tai_lieu_mo_rong_top_tier.md) | [🏠 Mục Lục](../README.md) | [Chương tiếp: SecAlign, StruQ & Instruction Hierarchy ➡️](11_secalign_struq_va_instruction_hierarchy.md)
