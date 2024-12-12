# AI-NLP-Lab
 A collection of advanced NLP projects, including a medical QA chatbot, text classification models, named entity recognition (NER), and deployment techniques like pruning, quantization, knowledge distillation, and ONNX conversion.

## Repository Structure

```
AI-NLP-Lab/
├── chatbot/               # Medical chatbot using GPT-2 with LoRA, SFT, and DPO
├── classification/
│   ├── encoder/           # Encoder-based model for book rating classification
│   └── decoder/           # Decoder-based model for sentiment analysis
├── NER/
│   ├── NER         # Medical entity recognition with MedicalZS
│   └── MultilingualNER    # Multilingual NER experiments
├── Deployment/
│    ├── Encoder
│    │   ├── Pruning       # Model pruning techniques
│    │   ├── Quantization   # Model quantization examples
│    │   ├── KnowledgeDistillation      # Distillation-based model │compression
│    │   └── ONNXConversion     # Exporting models to ONNX and running │them
└── README.md               # Project overview and instructions
```

## Key Features

- **Medical QA Bot:** Chatbot for medical Q&A using GPT-2 with LoRA, SFT, and DPO.
- **Text Classification:** Goodreads rating prediction and Twitter sentiment classification.
- **NER Models:** Medical and multilingual named entity recognition.
- **Deployment Techniques:** Pruning, quantization, knowledge distillation, and ONNX conversion.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MedNLP-Hub.git
   cd MedNLP-Hub
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the specific project folder.
2. Follow the instructions provided in the project's README.md.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or support, feel free to contact me via [LinkedIn](https://www.linkedin.com/in/aquib-ali-khan-668a4a192/) or open an issue in this repository.
