import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import easyocr
from PIL import Image, ImageTk

class LicensePlateDetector:
    def __init__(self, root):
        self.reader = easyocr.Reader(['en'], gpu=True)

        self.root = root
        self.root.title("Nhận diện biển số xe")
        self.root.geometry("800x800")
        self.root.configure(bg="#2e3d49")

        self.frame = tk.Frame(root, bg="#2e3d49", padx=20, pady=20)
        self.frame.pack(fill="both", expand=True)
        self.video_source = None
        self.vid = None
        self.is_video = False
        self.is_paused = False  # Thêm biến để kiểm tra video có đang tạm dừng hay không
        self.enhance_count = 0  # Thêm biến theo dõi số lần tăng độ phân giải
        self.setup_ui()
        self.max_plate_confidence = 0  # Khởi tạo độ tin cậy cao nhất

    def setup_ui(self):
        title_label = tk.Label(
            self.frame,
            text="Nhận diện biển số xe",
            font=("Helvetica", 24, "bold"),
            fg="white",
            bg="#2e3d49"
        )
        title_label.pack(pady=10)

        self.label = tk.Label(self.frame, bg="#2e3d49")
        self.label.pack(pady=10)

        button_frame = tk.Frame(self.frame, bg="#2e3d49")
        button_frame.pack(pady=10)

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

        control_frame = tk.Frame(self.frame, bg="#2e3d49")
        control_frame.pack(pady=10)

        self.pause_button = tk.Button(
            control_frame,
            text="Dừng Video",
            font=("Helvetica", 14),
            fg="white",
            bg="#FFEB3B",
            relief="flat",
            width=20,
            height=2,
            command=self.pause_video
        )
        self.pause_button.pack(side=tk.LEFT, padx=10)

        self.resume_button = tk.Button(
            control_frame,
            text="Tiếp Tục",
            font=("Helvetica", 14),
            fg="white",
            bg="#03A9F4",
            relief="flat",
            width=20,
            height=2,
            command=self.resume_video
        )
        self.resume_button.pack(side=tk.LEFT, padx=10)

        self.plate_text_label = tk.Label(
            self.frame,
            text="Biển số: ",
            font=("Helvetica", 16),
            fg="white",
            bg="#2e3d49"
        )
        self.plate_text_label.pack(pady=10)

        self.vehicle_type_label = tk.Label(
            self.frame,
            text="Loại xe: ",
            font=("Helvetica", 16),
            fg="white",
            bg="#2e3d49"
        )
        self.vehicle_type_label.pack(pady=10)

    def open_file(self):
        if self.vid:
            self.vid.release()
            self.vid = None

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh hoặc video",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"),
                       ("Video files", "*.mp4 *.avi *.mov")]
        )
        if not file_path:
            return

        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):  # Nếu là video
            self.is_video = True
            self.process_video(file_path)
        else:  # Nếu là ảnh
            self.is_video = False
            self.process_image(file_path)

    def process_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Lỗi", "Không thể mở ảnh.")
            return

        plate_image, plate_text, vehicle_type, plate_confidence, vehicle_confidence = self.detect_license_plate(image)
        self.display_result(plate_image, plate_text, vehicle_type, plate_confidence, vehicle_confidence)

    def process_video(self, file_path):
        self.vid = cv2.VideoCapture(file_path)
        if not self.vid.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở video.")
            return
        self.update_video()

    def update_video(self):
        if not self.is_paused:  # Kiểm tra nếu video không tạm dừng
            ret, frame = self.vid.read()
            if ret:
                plate_image, plate_text, vehicle_type, plate_confidence, vehicle_confidence = self.detect_license_plate(
                    frame)

                # Kiểm tra nếu phát hiện biển số với độ tin cậy cao hơn trước đó
                if plate_text and plate_confidence > self.max_plate_confidence:  # So sánh với độ tin cậy cao nhất hiện tại
                    self.max_plate_confidence = plate_confidence  # Cập nhật độ tin cậy cao nhất
                    messagebox.showinfo("Phát hiện biển số",
                                        f"Biển số: {plate_text}\nLoại xe: {vehicle_type}\nĐộ tin cậy: {plate_confidence:.2f}")
                    # Bạn có thể thêm logic lưu hình ảnh hoặc thực hiện hành động khác tại đây

                self.display_result(plate_image, plate_text, vehicle_type, plate_confidence, vehicle_confidence)
                self.root.after(30, self.update_video)
            else:
                self.vid.release()
                self.vid = None

    def detect_license_plate(self, image):
        # Tiền xử lý ảnh
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.display_intermediate_image(gray, "Gray Image")

        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        self.display_intermediate_image(gray, "Filtered Image")

        edges = cv2.Canny(gray, 170, 200)
        self.display_intermediate_image(edges, "Edges")

        # Phát hiện các đường viền (contour)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        detected_regions = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:  # Nếu có 4 cạnh, có thể là hình chữ nhật
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                if 1 <= aspect_ratio <= 6 and w > 50 and h > 15:  # Điều kiện lọc biển số
                    plate_region = image[y:y + h, x:x + w]
                    detected_regions.append((plate_region, (x, y, w, h)))

        self.display_intermediate_image(image, "Detected Region")

        # OCR nhận diện biển số
        plate_text = ""
        best_region = None
        vehicle_type = "Không nhận diện được"
        plate_confidence = 0  # Độ tin cậy của biển số
        vehicle_confidence = 0  # Độ tin cậy của loại xe

        for region, (x, y, w, h) in detected_regions:
            try:
                results = self.reader.readtext(region)
                if results:
                    plate_text_parts = [result[1] for result in results]
                    plate_text = ' '.join(plate_text_parts).strip().upper()  # Chuyển chữ cái thành chữ hoa
                    plate_confidence = max(results, key=lambda x: x[2])[2]  # Độ tin cậy của OCR

                    if plate_confidence > 0.5:
                        # Lưu lại vùng có độ tin cậy cao nhất
                        best_region = (x, y, w, h)

                        # Sử dụng tỷ lệ chiều rộng/chiều cao để phân loại xe
                        if w / h > 2:  # Nếu tỷ lệ chiều ngang/chiều cao lớn hơn 2, là ô tô
                            vehicle_type = "Ô tô"
                            vehicle_confidence = 1.0  # Độ tin cậy của ô tô
                        else:  # Nếu tỷ lệ nhỏ hơn hoặc bằng 2, là xe máy
                            vehicle_type = "Xe máy"
                            vehicle_confidence = 1.0  # Độ tin cậy của xe máy

                        break  # Dừng khi nhận diện được biển số
            except Exception as e:
                print(f"OCR Error: {e}")

        # Xóa tất cả các khung không phải là của biển số có độ tin cậy cao nhất
        if best_region:
            x, y, w, h = best_region
            # Vẽ khung của biển số có độ tin cậy cao nhất
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        self.display_intermediate_image(image, "Final Result")

        return image, plate_text, vehicle_type, plate_confidence, vehicle_confidence

    def display_intermediate_image(self, image, title):
        """Hiển thị ảnh ở mỗi bước xử lý trong cửa sổ riêng."""
        # Chuyển đổi ảnh sang RGB nếu cần
        if len(image.shape) == 2:  # Nếu là ảnh grayscale
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize ảnh để hiển thị
        height, width = display_image.shape[:2]
        max_size = 400
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))

        img_pil = Image.fromarray(display_image)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Tạo cửa sổ mới nếu chưa có
        window_name = f"intermediate_window_{title}"
        if not hasattr(self, window_name):
            window = tk.Toplevel(self.root)
            window.title(title)
            label = tk.Label(window)
            label.pack()
            setattr(self, window_name, window)
            setattr(self, f"{window_name}_label", label)

        # Cập nhật ảnh
        label = getattr(self, f"{window_name}_label")
        label.img_tk = img_tk
        label.config(image=img_tk)

    def display_result(self, image, plate_text, vehicle_type, plate_confidence, vehicle_confidence):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        img = img.resize((640, 480))
        img_tk = ImageTk.PhotoImage(image=img)

        self.label.img_tk = img_tk
        self.label.config(image=img_tk)

        self.plate_text_label.config(text=f"Biển số: {plate_text.upper()} (Độ tin cậy: {plate_confidence:.2f})")
        self.vehicle_type_label.config(text=f"Loại xe: {vehicle_type.upper()}")

    def pause_video(self):
        self.is_paused = True  # Tạm dừng video khi nhấn nút

    def resume_video(self):
        self.is_paused = False  # Tiếp tục video khi nhấn nút
        self.update_video()


def main():
    root = tk.Tk()
    app = LicensePlateDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
