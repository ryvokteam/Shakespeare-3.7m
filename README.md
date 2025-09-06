# 🎭 Shakespeare AI Chat

Experience AI-generated Shakespearean text with our advanced LSTM neural network. This project provides both web and terminal interfaces for generating authentic Shakespeare-style text using a 3.7M parameter character-level language model.

## 🚀 Quick Start

### Web Interface (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run web interface
streamlit run app.py
```
Access at: http://localhost:8501

### Terminal Chat
```bash
# Run terminal interface
python chat_terminal.py
```

### Hugging Face Spaces
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/)

Visit our Hugging Face Space to try the demo instantly without installation.

## 🎯 Features

- **🎭 Authentic Shakespearean Text**: Trained on the complete works of William Shakespeare
- **⚡ Real-time Generation**: Generate text instantly with GPU acceleration support
- **🎛️ Interactive Controls**: Adjust creativity, text length, and temperature
- **💻 Multiple Interfaces**: Web browser and terminal chat modes
- **📱 Mobile Friendly**: Responsive design for all devices
- **📊 Performance Metrics**: Real-time generation statistics

## 🧠 Model Details

- **Architecture**: Character-level LSTM
- **Parameters**: 3.7 million
- **Training Data**: Complete Shakespeare corpus
- **Vocabulary**: 57 unique characters
- **Model File**: `english_char_rnn.pth`

## 🎮 Usage Examples

### Web Interface
1. **Character Prompts**: Start with `ROMEO:`, `HAMLET:`, or `JULIET:`
2. **Famous Quotes**: Try "To be or not to be" or "But soft"
3. **Creative Themes**: Use "Love is", "Death comes", "Time will"

### Terminal Commands
```
Available commands:
- generate <prompt> [length] [temperature]
- help
- quit

Example: generate "ROMEO: My love" 200 1.0
```

## 🔧 Installation

### System Requirements
- Python 3.7+
- PyTorch 1.9+
- 4GB RAM minimum
- CUDA-compatible GPU (optional, for acceleration)

### Step-by-Step Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/shakespeare-ai-chat.git
cd shakespeare-ai-chat
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify Model**
Ensure `english_char_rnn.pth` is in the project directory.

4. **Run Application**
```bash
# Web interface
streamlit run app.py

# Terminal interface
python chat_terminal.py
```

## 🎛️ Configuration

### Web Interface Settings
- **Text Length**: 50-1000 characters
- **Temperature**: 0.1-2.0 (creativity control)
- **Prompt Templates**: Pre-built Shakespearean prompts

### Advanced Options
- GPU acceleration (automatically detected)
- Custom model loading
- Performance monitoring

## 📊 Performance

| Interface | Generation Speed | Memory Usage |
|-----------|------------------|--------------|
| Web (CPU) | ~50 chars/sec | ~500MB |
| Web (GPU) | ~200 chars/sec | ~1GB |
| Terminal | ~100 chars/sec | ~300MB |

## 🐳 Docker Deployment

### Using Docker
```bash
# Build image
docker build -t shakespeare-ai .

# Run container
docker run -p 8501:8501 shakespeare-ai
```

### Hugging Face Spaces
This project is configured for easy deployment to Hugging Face Spaces:
1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect your GitHub repository
4. Deploy automatically

## 🧪 Development

### Project Structure
```
shakespeare-ai-chat/
├── app.py              # Main web interface
├── chat_terminal.py    # Terminal chat interface
├── web_interface.py    # Alternative web interface
├── model.py           # LSTM model definition
├── english_char_rnn.pth # Trained model
├── requirements.txt   # Dependencies
├── Dockerfile        # Container configuration
├── .streamlit/       # Streamlit config
└── README.md         # This file
```

### Adding New Features
1. **Custom Models**: Place new `.pth` files in root directory
2. **Interface Modifications**: Edit `app.py` for web changes
3. **Terminal Enhancements**: Modify `chat_terminal.py`

## 🐛 Troubleshooting

### Common Issues

**Model Not Found**
```bash
# Ensure model file exists
ls -la english_char_rnn.pth
```

**Memory Issues**
```bash
# Reduce batch size or text length
# Use CPU mode if GPU memory is limited
```

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Optimization
- Use GPU acceleration when available
- Adjust temperature for faster generation
- Reduce text length for quicker responses

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Shakespeare Corpus**: Complete works of William Shakespeare
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Hugging Face**: Model hosting and deployment

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/shakespeare-ai-chat/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/shakespeare-ai-chat/discussions)
- **Email**: your.email@example.com

---

<div align="center">

**⭐ Star this repository if you find it useful!**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/shakespeare-ai-chat?style=social)](https://github.com/yourusername/shakespeare-ai-chat)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/shakespeare-ai-chat?style=social)](https://github.com/yourusername/shakespeare-ai-chat/fork)

</div>