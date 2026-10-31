[⬅️ Chương trước: Circuit Breakers](12_circuit_breakers_representation_rerouting.md) | [🏠 Mục Lục](../README.md)

---

# Chương 13: Chuyên Đề Guard Cho Web Agent — WebAgentGuard, SnapGuard và So Sánh Với WARD

> **Công trình:**
> - *WebAgentGuard: A Reasoning-Driven Guard Model for Detecting Prompt Injection Attacks in Web Agents*, arXiv:2604.12284 (14/04/2026); có trang ICML 2026.
> - *SnapGuard: Lightweight Prompt Injection Detection for Screenshot-Based Web Agents*, arXiv:2604.25562 (28/04/2026); preprint.
> - *WARD* (arXiv:2605.15030, ICML 2026) đã trình bày ở [Chương 4](04_ward_web_agent_robust_defense.md).
> - *VPI-Bench*, arXiv:2506.02456, ICLR 2026: benchmark VPI cho Computer-Use và Browser-Use Agents.
>
> **Mức xác minh:** thông tin lấy từ tóm tắt và các trang liệt kê. **Môi trường biên soạn không đọc được toàn văn**, nên **không có số liệu F1/ASR của WebAgentGuard và SnapGuard** trong chương này. Những gì được nêu là thiết kế đã được mô tả công khai.

---

## 1. Bối Cảnh: Guard Chạy Song Song Với Agent

Bài toán chung: một web agent đọc ảnh chụp màn hình (kèm HTML đơn giản hóa) và hành động; kẻ tấn công nhúng chỉ thị vào trang (popup, banner, nội dung do người dùng sinh). Giải pháp ba bài đều đi theo hướng **guard độc lập chạy cạnh agent**, tách việc kiểm tra an toàn khỏi việc thực thi tác vụ, nhằm không làm tăng độ trễ trên đường chính và không phải sửa agent (tương thích hộp đen).

## 2. WebAgentGuard

- **Đầu vào/đầu ra:** theo mô tả trong danh sách mô hình cộng đồng, guard nhận chỉ thị người dùng, ảnh chụp màn hình trang và HTML đơn giản hóa, rồi suy luận và trả về nhãn dương (có chèn lệnh) hoặc âm (sạch).
- **Dữ liệu:** bộ dữ liệu đa phương thức **hoàn toàn tổng hợp**, sinh bằng GPT-5, gồm 164 chủ đề và 230 phong cách thiết kế giao diện.
- **Huấn luyện:** SFT nặng về suy luận, sau đó học tăng cường (RL). Backbone Qwen3-VL (4B và 8B) theo danh sách cộng đồng.
- **Tuyên bố của tác giả:** vượt các baseline mạnh mà giữ nguyên tiện ích và không thêm độ trễ; được đánh giá trên các benchmark như VPI và EIA.

## 3. SnapGuard

Bộ phát hiện **nhẹ** cho agent dựa trên ảnh chụp màn hình. Theo mô tả, SnapGuard coi phát hiện chèn lệnh là bài toán phân tích **biểu diễn đa phương thức** để tránh hai điểm yếu: phụ thuộc cấu trúc văn bản và chi phí của guard là VLM đầy đủ. Mẫu bài báo ghi ACM MM '26 nhưng phần xác nhận bản quyền chưa điền, nên **chưa thể coi là đã được nhận đăng**.

## 4. So Sánh Có Cơ Sở Với WARD

Bài WARD (Chương 4) tự phê bình WebAgentGuard ở hai điểm, được ghi nhận trong tóm tắt và trong kết quả ở Chương 4:

| Tiêu chí | WebAgentGuard | WARD (Ch. 4) |
|:---|:---|:---|
| Dữ liệu huấn luyện | Hoàn toàn tổng hợp (GPT-5) | WARD-Base khoảng 177K mẫu từ 719 nguồn (709 URL + 10 nền tảng), WARD-PIG cho tấn công nhắm vào guard |
| Mô hình hóa tấn công | Nhãn nhị phân | Cấu trúc hóa kênh tiêm và mục tiêu tấn công |
| Huấn luyện đối kháng | SFT + RL | A3T: đồng tiến hóa attacker–guard có bộ nhớ |
| Kích thước | 4B, 8B | 0.8B, 2B |
| Số liệu so sánh trong Ch. 4 | WebAgentGuard-4B: recall Popup 37.55%; 8B: 76.14% | WARD-0.8B/2B: 99% đến 100% trên các bộ OOD |

Lưu ý quan trọng: con số WARD vs WebAgentGuard trong bảng trên **do chính nhóm WARD đo** trên benchmark của họ. Với một cặp bài cạnh tranh, kết luận thận trọng là: "WARD tự báo cáo hiệu quả hơn", không phải "đã được kiểm chứng độc lập".

## 5. Vì Sao Cần VPI-Bench

VPI-Bench (ICLR 2026) cung cấp 306 ca kiểm thử trên 5 nền tảng (Amazon, Booking, BBC, Messenger, Email) cho cả Computer-Use Agent và Browser-Use Agent. Theo bài báo, agent bị lừa tới 51% (CUA) và 100% (BUA) trên một số nền tảng; prompt hệ thống kiểu "hãy cảnh giác" chỉ cải thiện hạn chế. Bộ này là thước đo trung lập để so sánh các guard ở trên, vì nó không do nhóm phòng vệ nào xây dựng.

## Câu Hỏi Ôn Tập

1. Vì sao chạy guard song song làm giảm độ trễ cảm nhận, và trong tình huống nào guard song song vẫn gây chậm hành động?
2. Dữ liệu hoàn toàn tổng hợp (WebAgentGuard) có rủi ro gì về phân phối so với dữ liệu thu từ URL thật (WARD)? Hãy nêu cách kiểm tra.
3. Thiết kế một thí nghiệm công bằng so sánh WebAgentGuard và WARD trên VPI-Bench.

## Tài Liệu Tham Khảo

- *WebAgentGuard*. arXiv:2604.12284, 2026. Trang ICML 2026: https://icml.cc/virtual/2026/67830
- *SnapGuard*. arXiv:2604.25562, 2026.
- Cao, T. và cộng sự. *WARD: Adversarially Robust Defense of Web Agents Against Prompt Injections*. ICML 2026. arXiv:2605.15030. https://icml.cc/virtual/2026/67778
- *VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents*. ICLR 2026. arXiv:2506.02456.

---

[⬅️ Chương trước: Circuit Breakers](12_circuit_breakers_representation_rerouting.md) | [🏠 Mục Lục](../README.md)
