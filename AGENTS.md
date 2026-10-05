# Hướng Dẫn & Chỉ Thị Bắt Buộc Dành Cho AI Agents (AGENTS.md)

> **Thông Tin Dự Án / Project Metadata:**
> - **Tác giả / Học viên:** Võ Phạm Tuấn Dũng (MSHV: 2570015) — `@tuandung222`
> - **Cán bộ Hướng dẫn:** TS. Lê Xuân Bách
> - **Chuyên ngành:** Thạc sĩ Khoa học Dữ liệu & Trí tuệ Nhân tạo Ứng dụng (CDNC AI - Khóa 261)
> - **Đơn vị:** Khoa Khoa học & Kỹ thuật Máy tính, Trường ĐH Bách Khoa, ĐHQG-HCM (HCMUT)
> - **Kho lưu trữ:** `tuandung222/model-based-vpi-defenses`

---

## 1. Quy Chuẩn Toán Học KaTeX (Bắt Buộc Tuân Thủ Tuyệt Đối)

### 1.1. CẤM TUYỆT ĐỐI DÙNG `\tag{...}`
- **Nguyên nhân kỹ thuật:** Lệnh `\tag{...}` là macro LaTeX chỉ chạy được khi KaTeX engine ở chế độ Display Mode nghiêm ngặt (`displayMode: true`). Khi xem file Markdown trên môi trường local (VS Code Markdown Preview, Markdown All in One, Obsidian, Notion), các trình phân tích Markdown thường parse khối công thức dưới dạng inline hoặc span, dẫn đến lỗi crash:
  ```text
  KaTeX parse error: \tag works only in display equations
  ```
- **Quy tắc thay thế bắt buộc:**
  - **KHÔNG BAO GIỜ** viết: `$$ E = mc^2 \tag{1} $$` hoặc `$ E = mc^2 \tag{1} $`
  - **LUÔN VIẾT**:
    ```latex
    $$
    E = mc^2 \qquad (1)
    $$
    ```
    Toán tử `\qquad (1)` (hoặc `\quad (1)`) hoạt động hoàn hảo 100% trên cả Display Mode lẫn Inline Mode, hiển thị đồng nhất trên GitHub, VSCode, Notion, Jupyter và các trình duyệt web.

### 1.2. Khối Công Thức Hiển Thị (Display Math Blocks) Phải Tách Dòng Riêng
- Luôn đặt dấu `$$` mở và `$$` đóng trên các dòng riêng biệt để trình phân tích CommonMark/GFM nhận diện chính xác là khối Math Block độc lập:
  ```markdown
  <!-- ĐÚNG -->
  $$
  \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{refusal}} + \lambda \cdot \mathcal{L}_{\text{utility}}
  $$

  <!-- SAI (Dễ bị parser ép về inline) -->
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{refusal}} + \lambda \cdot \mathcal{L}_{\text{utility}}$$
  ```

---

## 2. Quy Chuẩn Biểu Đồ Mermaid

1. **Không dùng khoảng nửa mở trong nhãn node:**
   - ❌ SAI: `subgraph S["Cửa sổ tầng [0, n)"]` hoặc `Node["Tầng [n, n+Delta_n)"]` (cặp `[` và `)` không cân bằng làm hỏng bộ phân tích cú pháp).
   - ✅ ĐÚNG: `subgraph S["Cửa sổ tầng (0 đến n)"]` hoặc `Node["Tầng (n đến n+Delta_n)"]`.
2. **Bọc nhãn chứa ký tự đặc biệt:**
   - Luôn bọc nhãn bằng cặp ngoặc kép an toàn: `NodeId["Nội dung nhãn"]`.

---

## 3. Quy Chuẩn Liên Kết & Kiểm Thẩm Tự Động

1. **Không bao giờ dùng link cục bộ `file:///`**: Toàn bộ liên kết phải là link tương đối nội bộ (`../`, `./`) hoặc URL công khai (`https://arxiv.org/...`).
2. **Chạy linter trước khi commit**:
   ```bash
   python3 scripts/check_markdown_katex.py
   # Hoặc tự động sửa nếu có phát sinh \tag:
   python3 scripts/check_markdown_katex.py --fix
   ```
   Repository đã được tích hợp Git Pre-commit Hook tự động tại `.git/hooks/pre-commit`, mọi commit vi phạm sẽ bị chặn tự động.
