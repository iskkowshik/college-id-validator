# 🔍 Project Title — College ID Validator

An AI-based microservice for verifying the authenticity of college ID cards using image and text validation techniques. It supports offline deployment via Docker and uses OCR (Tesseract + OpenCV), fuzzy matching, and ResNet-based template matching.

---

## 📌 Features

- 🔤 Text extraction using custom OCR pipeline  
- 🧠 Template image matching using ResNet50  
- 🎯 Validation of key fields: name, roll number, college  
- 🧾 Fuzzy logic for institution name matching  
- 🐳 Fully Dockerized backend for offline usage  
- 📂 `.npy` embedding saving & inference  
- 🔐 Secure and lightweight API server  

---

## 🛠️ Tech Stack

- **Languages**: Python  
- **Libraries**: OpenCV, Tesseract-OCR, PyTorch (ResNet50), FastAPI, FuzzyWuzzy / RapidFuzz  
- **Containerization**: Docker  

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/iskkowshik/college-id-validator.git
cd college-id-validator/backend

# Build Docker image
docker build -t college-id-validator .

# Run container
docker run -p 8000:8000 college-id-validator
```

---

## 📁 Folder Structure

```
college-id-validator/
├── backend/
│   ├── main.py
│   ├── image_classifier.py
│   ├── ocr_validator.py
│   ├── id_resnet_model.pth
│   └── ...
│
├── mernapp/
│   ├── client/
│   │   ├── public/
│   │   ├── src/
│   │   │   ├── App.js
│   │   │   └── components/
│   ├── server/ (optional)
│   └── package.json
│
├── screenshots/
│   ├── Screenshot1.png
│   ├── Screenshot2.png
│   └── ...
│
└── README.md
```

---

## 📸 Demo

_Add screenshots here of sample image input, extracted OCR text, and validation result._

---

## ✅ Example Input

**Image**: JPEG/PNG scanned ID card  
**Output**: JSON response with validity status and reason

```json
{
  "is_valid": true,
  "missing_fields": [],
  "matched_college": "Vasavi College of Engineering"
}
```

---

## 🧪 Test Locally (Without Docker)

```bash
cd backend
python main.py
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

MIT License

---

## 🧑‍💻 Maintainer

**Saikowshik Immanneni**

- 📧 Email: [kowshiksaikowshik696@gmail.com](mailto:kowshiksaikowshik696@gmail.com)  
- 💻 GitHub: [@iskkowshik](https://github.com/iskkowshik)  
- 🔗 LinkedIn: [Immanneni Sai Kowshik](https://www.linkedin.com/in/kowshik-saikowshik-063619264)  
- 🌐 Portfolio: [portfolio-ukv3.vercel.app](https://portfolio-ukv3.vercel.app)  
