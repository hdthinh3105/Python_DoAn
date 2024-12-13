import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import easyocr


class LicensePlateDetector:
    def __init__(self, root):
        # Khởi tạo EasyOCR reader với GPU
        self.reader = easyocr.Reader(['en'], gpu=True)  # Sử dụng GPU ở đây

        self.root = root
        self.root.title("Nhận diện biển số xe")
        self.root.geometry("800x800")
        self.root.configure(bg="#2e3d49")

        self.frame = tk.Frame(root, bg="#2e3d49", padx=20, pady=20)
        self.frame.pack(fill="both", expand=True)

        self.video_source = None
        self.vid = None
        self.is_video = False

        self.setup_ui()

    def setup_ui(self):
        # Tiêu đề
        title_label = tk.Label(
            self.frame,
            text="Nhận diện biển số xe",
            font=("Helvetica", 24, "bold"),
            fg="white",
            bg="#2e3d49"
        )
        title_label.pack(pady=10)

        # Nhãn hiển thị ảnh
        self.label = tk.Label(self.frame, bg="#2e3d49")
        self.label.pack(pady=10)

        # Khung nút
        button_frame = tk.Frame(self.frame, bg="#2e3d49")
        button_frame.pack(pady=10)

        # Nút chọn ảnh/video
        self.select_button = tk.Button(
            button_frame,
            text="Chọn Ảnh/Video",
            font=("Helvetica", 16),
            fg="white",
            bg="#388E3C",
            relief="flat",
            width=20,
            height=2,
            command=self.open_file
        )
        self.select_button.pack(side=tk.LEFT, padx=10)

        # Nút thoát
        exit_button = tk.Button(
            button_frame,
            text="Thoát",
            font=("Helvetica", 14),
            fg="white",
            bg="#FF5722",
            relief="flat",
            width=20,
            height=2,
            command=self.root.quit
        )
        exit_button.pack(side=tk.LEFT, padx=10)

        # Nhãn hiển thị biển số
        self.plate_text_label = tk.Label(
            self.frame,
            text="Biển số: ",
            font=("Helvetica", 16),
            fg="white",
            bg="#2e3d49"
        )
        self.plate_text_label.pack(pady=10)

        # Nhãn hiển thị loại xe
        self.vehicle_type_label = tk.Label(
            self.frame,
            text="Loại xe: ",
            font=("Helvetica", 16),
            fg="white",
            bg="#2e3d49"
        )
        self.vehicle_type_label.pack(pady=10)

    def open_file(self):
        # Đặt lại trạng thái video cũ
        if self.vid:
            self.vid.release()
            self.vid = None

        # Mở hộp thoại chọn file
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh hoặc video",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"),
                      ("Video files", "*.mp4 *.avi *.mov")]
        )
        if not file_path:
            return

        # Xác định loại file
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):  # Kiểm tra video
            self.is_video = True
            self.process_video(file_path)
        else:  # Kiểm tra ảnh
            self.is_video = False
            self.process_image(file_path)

    def process_image(self, file_path):
        # Đọc ảnh
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Lỗi", "Không thể mở ảnh.")
            return

        # Nhận diện biển số
        plate_image, plate_text, vehicle_type = self.detect_license_plate(image)

        # Hiển thị kết quả
        self.display_result(plate_image, plate_text, vehicle_type)

    def process_video(self, file_path):
        # Mở video
        self.vid = cv2.VideoCapture(file_path)
        if not self.vid.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở video.")
            return

        # Bắt đầu xử lý video
        self.update_video()

    def update_video(self):
        # Đọc frame từ video
        ret, frame = self.vid.read()
        if ret:
            # Nhận diện biển số trong frame
            plate_image, plate_text, vehicle_type = self.detect_license_plate(frame)

            # Hiển thị kết quả
            self.display_result(plate_image, plate_text, vehicle_type)

            # Lên lịch frame tiếp theo
            self.root.after(30, self.update_video)
        else:
            # Kết thúc video
            self.vid.release()
            self.vid = None

    def detect_license_plate(self, image):
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Áp dụng bộ lọc để cải thiện ảnh
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 170, 200)

        # Tìm contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]  # Dò nhiều contours hơn

        detected_regions = []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Tìm contours hình chữ nhật
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                # Lọc vùng có kích thước phù hợp với biển số
                aspect_ratio = w / h
                if 1 <= aspect_ratio <= 6:  # Đảm bảo tìm được biển số cả khi lệch
                    plate_region = image[y:y + h, x:x + w]
                    detected_regions.append((plate_region, (x, y, w, h)))

        plate_text = ""
        best_region = None
        vehicle_type = "Không nhận diện được"

        # Dò tìm biển số trong từng vùng phát hiện
        for region, (x, y, w, h) in detected_regions:
            try:
                # Sử dụng EasyOCR để nhận diện text
                results = self.reader.readtext(region)
                if results:
                    # Nối các phần của biển số lại thành một chuỗi duy nhất
                    plate_text_parts = [result[1] for result in results]
                    plate_text = ' '.join(plate_text_parts).strip()

                    # Lưu vùng có độ tin cậy cao nhất
                    candidate_confidence = max(results, key=lambda x: x[2])[2]
                    if candidate_confidence > 0.5:
                        best_region = (x, y, w, h)

            except Exception as e:
                print(f"OCR Error: {e}")

        # Vẽ hình chữ nhật quanh vùng có biển số tốt nhất
        if best_region:
            x, y, w, h = best_region
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Xác định loại xe dựa vào kích thước biển số
            if w > 100 and h > 30:
                vehicle_type = "Ô tô"
            else:
                vehicle_type = "Xe máy"

        return image, plate_text, vehicle_type

    def display_result(self, image, plate_text, vehicle_type):
        # Chuyển ảnh từ BGR sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Điều chỉnh kích thước ảnh
        img = Image.fromarray(image)
        img = img.resize((640, 480))
        img_tk = ImageTk.PhotoImage(image=img)

        # Cập nhật nhãn ảnh
        self.label.img_tk = img_tk
        self.label.config(image=img_tk)

        # Cập nhật text biển số
        self.plate_text_label.config(text=f"Biển số: {plate_text}")

        # Cập nhật loại xe
        self.vehicle_type_label.config(text=f"Loại xe: {vehicle_type}")


def main():
    root = tk.Tk()
    app = LicensePlateDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
