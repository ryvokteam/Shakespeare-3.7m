#!/usr/bin/env python3
"""
Shakespeare Text Generator Terminal Chat
Single model interface using 3.7M parameter english_char_rnn.pth
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import cmd
import sys
try:
    import readline
except ImportError:
    # Windows compatibility
    try:
        import pyreadline3 as readline
    except ImportError:
        readline = None

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

class ShakespeareChat(cmd.Cmd):
    """Interactive terminal chat for Shakespeare text generation using 3.7M parameter model"""
    
    prompt = '🎭 Shakespeare> '
    intro = """
🎭 Welcome to Shakespeare Text Generator Terminal!
Using 3.7M parameter English Char RNN model

Available commands:
  generate <prompt>     - Generate text from prompt
  settings              - Show current settings
  save <filename>       - Save last generation
  clear                 - Clear screen
  help                  - Show this help
  exit                  - Exit chat

Examples:
  generate To be or not to be
  generate ROMEO:
  save my_shakespeare.txt
"""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab = {}
        self.settings = {
            'length': 200,
            'temperature': 1.0,
            'model_name': 'English Char RNN (3.7M params)'
        }
        self.last_generation = ""
        self.load_model()
    
    def load_model(self):
        """Load the 3.7M parameter model"""
        model_path = "english_char_rnn.pth"
        
        if not os.path.exists(model_path):
            print(f"❌ Model file {model_path} not found!")
            sys.exit(1)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
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
            
            print(f"✅ Loaded {self.settings['model_name']}")
            print(f"📊 Vocabulary size: {len(self.vocab['chars'])} characters")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def generate_text(self, prompt: str, length: int = None, temperature: float = None):
        """Generate text from prompt"""
        if not self.model:
            return "No model loaded"
        
        length = length or self.settings['length']
        temperature = temperature or self.settings['temperature']
        
        char_to_idx = self.vocab['char_to_idx']
        idx_to_char = self.vocab['idx_to_char']
        
        try:
            with torch.no_grad():
                # Initialize hidden state
                hidden = self.model.init_hidden(1, self.device)
                
                # Convert prompt to indices
                prompt_indices = [char_to_idx.get(char, 0) for char in prompt]
                
                # Process prompt
                if prompt_indices:
                    prompt_tensor = torch.tensor([prompt_indices], device=self.device)
                    _, hidden = self.model(prompt_tensor, hidden)
                
                # Generate text
                generated = list(prompt_indices) if prompt_indices else []
                current_char = prompt_indices[-1] if prompt_indices else np.random.randint(len(idx_to_char))
                
                for _ in range(length):
                    input_tensor = torch.tensor([[current_char]], device=self.device)
                    output, hidden = self.model(input_tensor, hidden)
                    
                    # Apply temperature
                    output = output.squeeze() / temperature
                    probabilities = torch.softmax(output, dim=-1)
                    
                    # Sample next character
                    current_char = torch.multinomial(probabilities, 1).item()
                    generated.append(current_char)
                
                # Convert back to text
                generated_text = prompt
                for idx in generated[len(prompt_indices):]:
                    generated_text += idx_to_char[idx]
                
                self.last_generation = generated_text
                return generated_text
                
        except Exception as e:
            return f"Error generating text: {e}"
    
    def do_generate(self, arg):
        """Generate text: generate <prompt> [length] [temperature]"""
        if not arg.strip():
            print("❌ Usage: generate <prompt> [length] [temperature]")
            return
        
        parts = arg.split()
        prompt = " ".join(parts)  # Take all text as prompt
        length = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else self.settings['length']
        temperature = float(parts[-1]) if len(parts) > 2 else self.settings['temperature']
        
        print(f"🎭 Generating with {self.settings['model_name']}...")
        
        generated = self.generate_text(prompt, length, temperature)
        
        print("\n" + "="*60)
        print(generated)
        print("="*60 + "\n")
    
    def do_settings(self, arg):
        """Show current settings"""
        print("⚙️ Current settings:")
        print(f"  Model: {self.settings['model_name']}")
        print(f"  Length: {self.settings['length']}")
        print(f"  Temperature: {self.settings['temperature']}")
        print(f"  Device: {self.device}")
    
    def do_save(self, arg):
        """Save last generation: save <filename>"""
        if not arg.strip():
            print("❌ Usage: save <filename>")
            return
        
        filename = arg.strip()
        if not self.last_generation:
            print("❌ No text to save")
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.last_generation)
            print(f"✅ Saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving: {e}")
    
    def do_clear(self, arg):
        """Clear screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def do_length(self, arg):
        """Set generation length: length <number>"""
        if arg.strip().isdigit():
            self.settings['length'] = int(arg)
            print(f"✅ Length set to {self.settings['length']}")
        else:
            print(f"❌ Current length: {self.settings['length']}")
    
    def do_temp(self, arg):
        """Set temperature: temp <number>"""
        try:
            temp = float(arg)
            if 0.1 <= temp <= 2.0:
                self.settings['temperature'] = temp
                print(f"✅ Temperature set to {self.settings['temperature']}")
            else:
                print("❌ Temperature must be between 0.1 and 2.0")
        except ValueError:
            print(f"❌ Current temperature: {self.settings['temperature']}")
    
    def do_exit(self, arg):
        """Exit chat"""
        print("👋 Goodbye!")
        return True
    
    def do_quit(self, arg):
        """Exit chat"""
        return self.do_exit(arg)
    
    def default(self, arg):
        """Handle unknown commands"""
        if arg.strip():
            # Treat as generate command
            self.do_generate(arg)

def main():
    print("🎭 Shakespeare Text Generator Chat Terminal")
    print("=" * 50)
    
    try:
        chat = ShakespeareChat()
        chat.cmdloop()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == '__main__':
    main()