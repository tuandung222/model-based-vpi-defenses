[⬅️ Chương trước: SecAlign, StruQ & Instruction Hierarchy](11_secalign_struq_va_instruction_hierarchy.md) | [🏠 Mục Lục](../README.md) | [Chương tiếp: WebAgentGuard, SnapGuard & WARD ➡️](13_webagentguard_snapguard_va_so_sanh_ward.md)

---

# Chương 12: Chuyên Đề Hội Nghị A\* — Circuit Breakers và Định Tuyến Lại Biểu Diễn (NeurIPS 2024)

> **Công trình:** Zou và cộng sự, *Improving Alignment and Robustness with Circuit Breakers*, **NeurIPS 2024**, arXiv:2406.04313. Mã nguồn và checkpoint công khai của Gray Swan AI.
>
> **Mức xác minh:** venue đã xác minh trên trang NeurIPS. Dạng hàm tổn thất bên dưới lấy từ tóm tắt thứ cấp ($\mathcal{L}_s$ dạng ReLU của cosine) và **cần đối chiếu với bài gốc**; hệ số, tầng áp dụng và bảng số liệu đa phương thức chưa được đọc trực tiếp.

---

## 1. Ý Tưởng

Phần lớn phòng vệ tác động lên *đầu vào* (lọc) hoặc *đầu ra* (từ chối). Circuit Breakers can thiệp vào **quá trình sinh**: khi biểu diễn ẩn cho thấy mô hình đang đi vào vùng sinh nội dung có hại, biểu diễn đó bị "định tuyến lại" (Representation Rerouting, RR) sang vùng vô nghĩa để việc sinh bị ngắt. Điểm khác biệt quan trọng với huấn luyện từ chối: phương pháp nhắm vào *biểu diễn gây hại*, không phụ thuộc vào kiểu tấn công cụ thể.

So với ARGUS ([Chương 3](03_argus_activation_steering.md)): ARGUS cộng vector tại lúc suy luận và không đổi trọng số; Circuit Breakers đổi trọng số (qua LoRA) để biểu diễn tự được đẩy đi.

## 2. Hàm Mục Tiêu (Dạng Tổng Quát)

Gọi $\text{rep}_\theta(x)$ là biểu diễn ẩn tại các tầng được chọn của mô hình đang huấn luyện và $\text{rep}_{\text{orig}}(x)$ là của mô hình gốc đóng băng. Có hai tập: tập **circuit breaker** $\mathcal{D}_s$ (đầu vào dẫn tới nội dung có hại) và tập **retain** $\mathcal{D}_r$ (nội dung lành tính, cần giữ nguyên hành vi).

Tổn thất định tuyến lại đẩy biểu diễn có hại về phía trực giao với biểu diễn gốc:

$$
\mathcal{L}_s = \mathbb{E}_{x \in \mathcal{D}_s} \left[ \text{ReLU}\left( \cos\left( \text{rep}_\theta(x), \text{rep}_{\text{orig}}(x) \right) \right) \right]
$$

Tổn thất giữ lại duy trì biểu diễn trên dữ liệu lành tính:

$$
\mathcal{L}_r = \mathbb{E}_{x \in \mathcal{D}_r} \left[ \left\| \text{rep}_\theta(x) - \text{rep}_{\text{orig}}(x) \right\|_2 \right]
$$

Tổng tổn thất là tổ hợp có trọng số của hai thành phần trên, với trọng số của $\mathcal{L}_s$ giảm dần và của $\mathcal{L}_r$ tăng dần trong quá trình huấn luyện (chi tiết lịch trọng số cần đối chiếu với bài gốc). Bài báo dùng Low-Rank Representation Adaptation (LoRRA) để huấn luyện ổn định, và nêu rõ dạng tổn thất định tuyến lại có thể thay đổi.

## 3. Thí Nghiệm Đa Phương Thức

Nhóm tác giả áp dụng RR cho **LLaVA-NeXT-Mistral-7B**, trộn tập circuit breaker tổng hợp đa phương thức với tập retain LLaVA-Instruct. Theo tóm tắt, mô hình sau RR **bền hơn đáng kể trước tấn công PGD trên ảnh** (kể cả khi bản gốc có thêm safety prompt) và **giữ năng lực** trên các benchmark đa phương thức như MMMU và LLaVA-Wild. Một bản tóm tắt thứ cấp nêu mức giảm khoảng 84% tỷ lệ tuân thủ ở tấn công ảnh với năng lực giữ trong 0.5%; **con số này chưa được kiểm chứng** với bài gốc nên không nên trích dẫn.

## 4. Giới Hạn

- Nguồn thứ cấp lưu ý RR dễ bị tổn thương trước kiểu tấn công ép token (token-forcing) mới, và chính các tác giả coi circuit breakers là lớp bảo vệ cuối cùng chứ không thay thế được căn chỉnh nền.
- Dữ liệu circuit breaker phải đại diện cho "vùng có hại". Với VPI, "có hại" không nằm ở chủ đề mà ở *việc làm theo chỉ thị sai nguồn*; chưa có bằng chứng công khai rằng RR bao quát hijack tác vụ.
- Cần quyền truy cập trọng số (white-box).

## 5. Liên Hệ Với Chuyên Khảo

| Phương pháp | Can thiệp | Thời điểm | Đổi trọng số | Phạm vi đã chứng minh |
|:---|:---|:---:|:---:|:---|
| Circuit Breakers | Biểu diễn ẩn nhiều tầng | Huấn luyện | Có (LoRA) | Jailbreak văn bản và PGD ảnh |
| ARGUS (Ch. 3) | Vector hướng trong không gian con | Suy luận | Không | Chèn lệnh gián tiếp ảnh/video/audio |
| SafePTR (Ch. 2) | Cắt tỉa token tại tầng nhạy cảm | Suy luận | Không | Jailbreak ảnh-văn bản |
| Dyslexify (Ch. 9) | Loại bỏ attention head mang chữ | Suy luận | Không (ablate) | Tấn công typographic trên CLIP |

## Câu Hỏi Ôn Tập

1. Vì sao $\mathcal{L}_s$ dùng ReLU của cosine thay vì cực tiểu hóa cosine thẳng? (Gợi ý: điều gì xảy ra khi cosine đã âm.)
2. Nếu tập circuit breaker không chứa ví dụ VPI, ta kỳ vọng RR bảo vệ đến đâu? Thiết kế một thí nghiệm kiểm tra.
3. So sánh ưu nhược điểm của can thiệp lúc huấn luyện (RR) và lúc suy luận (ARGUS).

## Tài Liệu Tham Khảo

- Zou, A. và cộng sự. *Improving Alignment and Robustness with Circuit Breakers*. NeurIPS 2024. arXiv:2406.04313. https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract.html
- Hufe, L. và cộng sự. *Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP*. ICLR 2026. arXiv:2508.20570.

---

[⬅️ Chương trước: SecAlign, StruQ & Instruction Hierarchy](11_secalign_struq_va_instruction_hierarchy.md) | [🏠 Mục Lục](../README.md) | [Chương tiếp: WebAgentGuard, SnapGuard & WARD ➡️](13_webagentguard_snapguard_va_so_sanh_ward.md)
