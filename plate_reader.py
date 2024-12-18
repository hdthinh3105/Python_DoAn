import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import easyocr
from sklearn.svm import SVC


class LicensePlateDetector:
    def __init__(self, root):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.svm = self.train_svm()

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

    def train_svm(self):
        # Dữ liệu giả lập huấn luyện
        X = np.array([
            [150, 50, 3.0, 7500],  # Biển số ô tô
            [120, 40, 3.0, 4800],  # Biển số xe máy
            [100, 30, 3.33, 3000],  # Xe máy
            [200, 60, 3.33, 12000],  # Ô tô
            [300, 100, 3.0, 30000],  # Ô tô lớn
            [80, 25, 3.2, 2000],  # Xe máy
            [250, 75, 3.3, 18750],  # Ô tô
            [180, 50, 3.6, 9000],  # Xe máy phân khối lớn
            [320, 90, 3.5, 28800],  # Ô tô SUV
        ])
        y = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1])  # 1 = Ô tô, 0 = Xe máy

        svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        svm.fit(X, y)
        return svm

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

        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            self.is_video = True
            self.process_video(file_path)
        else:
            self.is_video = False
            self.process_image(file_path)

    def process_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Lỗi", "Không thể mở ảnh.")
            return

        plate_image, plate_text, vehicle_type = self.detect_license_plate(image)
        self.display_result(plate_image, plate_text, vehicle_type)

    def process_video(self, file_path):
        self.vid = cv2.VideoCapture(file_path)
        if not self.vid.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở video.")
            return
        self.update_video()

    def update_video(self):
        ret, frame = self.vid.read()
        if ret:
            plate_image, plate_text, vehicle_type = self.detect_license_plate(frame)
            self.display_result(plate_image, plate_text, vehicle_type)
            self.root.after(30, self.update_video)
        else:
            self.vid.release()
            self.vid = None

    def detect_license_plate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 170, 200)

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

        plate_text = ""
        best_region = None
        vehicle_type = "Không nhận diện được"

        for region, (x, y, w, h) in detected_regions:
            try:
                # Đọc văn bản biển số từ vùng phát hiện
                results = self.reader.readtext(region)
                if results:
                    plate_text_parts = [result[1] for result in results]
                    plate_text = ' '.join(plate_text_parts).strip()
                    candidate_confidence = max(results, key=lambda x: x[2])[2]
                    if candidate_confidence > 0.5:
                        best_region = (x, y, w, h)

                        features = np.array([w, h, w / h, w * h]).reshape(1, -1)
                        prediction = self.svm.predict(features)
                        vehicle_type = "Ô tô" if prediction == 1 else "Xe máy"

            except Exception as e:
                print(f"OCR Error: {e}")

        if best_region:
            x, y, w, h = best_region
            # Vẽ khung biển số và tính toán các điểm đỉnh
            top_left = (x, y)
            top_right = (x + w, y)
            bottom_left = (x, y + h)
            bottom_right = (x + w, y + h)

            # Vẽ các điểm đỉnh và khung
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(image, top_left, 5, (0, 0, 255), -1)
            cv2.circle(image, top_right, 5, (0, 0, 255), -1)
            cv2.circle(image, bottom_left, 5, (0, 0, 255), -1)
            cv2.circle(image, bottom_right, 5, (0, 0, 255), -1)

        return image, plate_text, vehicle_type

    def display_result(self, image, plate_text, vehicle_type):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        img = img.resize((640, 480))
        img_tk = ImageTk.PhotoImage(image=img)

        self.label.img_tk = img_tk
        self.label.config(image=img_tk)

        self.plate_text_label.config(text=f"Biển số: {plate_text}")
        self.vehicle_type_label.config(text=f"Loại xe: {vehicle_type}")


def main():
    root = tk.Tk()
    app = LicensePlateDetector(root)
    root.mainloop()


if __name__ == "__main__":
    main()
