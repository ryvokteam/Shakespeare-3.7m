import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import time

# Configure Streamlit page for Hugging Face deployment
st.set_page_config(
    page_title="Shakespeare AI Chat",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))

class ShakespeareGenerator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.vocab = {}
        self.load_model()
    
    def load_model(self):
        """Load the Shakespeare model"""
        if not os.path.exists(self.model_path):
            st.error(f"Model file not found: {self.model_path}")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            vocab_size = checkpoint.get('vocab_size', 57)
            embed_dim = checkpoint.get('embedding_dim', 256)
            hidden_dim = checkpoint.get('hidden_dim', 512)
            num_layers = checkpoint.get('num_layers', 2)
            
            self.model = ShakespeareLSTM(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.vocab = {
                'char_to_idx': checkpoint['char_to_idx'],
                'idx_to_char': checkpoint['idx_to_char'],
                'chars': list(checkpoint['char_to_idx'].keys())
            }
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def generate_text(self, prompt, length=200, temperature=1.0):
        """Generate Shakespeare-style text"""
        if not self.model:
            return "Model not loaded"
        
        char_to_idx = self.vocab['char_to_idx']
        idx_to_char = self.vocab['idx_to_char']
        
        try:
            with torch.no_grad():
                hidden = self.model.init_hidden(1, self.device)
                
                prompt_indices = [char_to_idx.get(char, 0) for char in prompt]
                
                if prompt_indices:
                    prompt_tensor = torch.tensor([prompt_indices], device=self.device)
                    _, hidden = self.model(prompt_tensor, hidden)
                
                generated_text = prompt
                current_char = prompt_indices[-1] if prompt_indices else np.random.randint(len(idx_to_char))
                
                progress_bar = st.progress(0)
                
                for i in range(length):
                    input_tensor = torch.tensor([[current_char]], device=self.device)
                    output, hidden = self.model(input_tensor, hidden)
                    
                    output = output.squeeze() / temperature
                    probabilities = torch.softmax(output, dim=-1)
                    current_char = torch.multinomial(probabilities, 1).item()
                    
                    generated_text += idx_to_char[current_char]
                    progress_bar.progress((i + 1) / length)
                
                progress_bar.empty()
                return generated_text
                
        except Exception as e:
            return f"Error generating text: {e}"

# Initialize the generator
@st.cache_resource
def load_generator():
    generator = ShakespeareGenerator("english_char_rnn.pth")
    return generator

# Main UI
def main():
    st.title("🎭 Shakespeare AI Chat")
    st.markdown("**Experience AI-generated Shakespearean text with our 3.7M parameter LSTM model**")
    
    # Load model
    with st.spinner("Loading Shakespeare model..."):
        generator = load_generator()
    
    if not generator.model:
        st.error("Failed to load model. Please check if english_char_rnn.pth exists.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Generation Settings")
    
    # Model info
    st.sidebar.success("✅ Model loaded: 3.7M parameters")
    st.sidebar.info("📊 Vocabulary: 57 characters")
    
    # Settings
    length = st.sidebar.slider(
        "Text Length",
        min_value=50,
        max_value=1000,
        value=200,
        step=50
    )
    
    temperature = st.sidebar.slider(
        "Creativity (Temperature)",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Lower values = more predictable, Higher values = more creative"
    )
    
    # Main interface
    st.header("💬 Generate Shakespearean Text")
    
    # Preset prompts
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            value="To be or not to be",
            height=100,
            placeholder="Start with character names like 'ROMEO:' or 'HAMLET:'"
        )
    
    with col2:
        st.write("")
        st.write("")
        preset = st.selectbox(
            "Quick prompts:",
            ["", "To be or not to be", "ROMEO:", "JULIET:", "HAMLET:", "My lord"]
        )
        if preset:
            prompt = preset
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎭 Generate Shakespeare Text", use_container_width=True, type="primary"):
            if prompt.strip():
                with st.spinner("🎭 Creating Shakespearean masterpiece..."):
                    start_time = time.time()
                    generated_text = generator.generate_text(prompt, length, temperature)
                    end_time = time.time()
                    
                    # Display results
                    st.header("📜 Generated Text")
                    
                    # Stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Characters", len(generated_text))
                    with col2:
                        st.metric("Time", f"{end_time - start_time:.2f}s")
                    with col3:
                        st.metric("Speed", f"{len(generated_text)/(end_time - start_time):.1f} char/s")
                    
                    # Text display
                    st.text_area(
                        "Generated text:",
                        value=generated_text,
                        height=300,
                        key="generated_text"
                    )
                    
                    # Download
                    st.download_button(
                        label="📥 Download Text",
                        data=generated_text,
                        file_name=f"shakespeare_{int(time.time())}.txt",
                        mime="text/plain"
                    )
            else:
                st.warning("Please enter a prompt to generate text.")
    
    # Tips and info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.markdown("- Use character names as prompts")
    st.sidebar.markdown("- Try different temperatures")
    st.sidebar.markdown("- Start with famous Shakespeare lines")
    
    # System info
    if torch.cuda.is_available():
        st.sidebar.success(f"🚀 GPU: {torch.cuda.get_device_name()}")
    else:
        st.sidebar.info("💻 Running on CPU")
    
    # About section
    with st.sidebar.expander("ℹ️ About"):
        st.write("""
        **Shakespeare AI Chat**
        
        This application uses a character-level LSTM neural network trained on the complete works of William Shakespeare.
        
        **Features:**
        - Interactive web interface
        - Real-time text generation
        - Adjustable creativity parameters
        - GPU acceleration support
        
        **Model Details:**
        - Architecture: LSTM
        - Parameters: 3.7 million
        - Training data: Complete Shakespeare corpus
        - Vocabulary: 57 unique characters
        """)

if __name__ == "__main__":
    main()