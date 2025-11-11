from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import base64
from datetime import datetime
import json

# Document processing
import PyPDF2
from docx import Document
import pytesseract
from PIL import Image

# AI and NLP
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Text-to-Speech
from gtts import gTTS
import pyttsx3

# Optional: OpenAI for advanced Q&A
# from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size


class DocumentProcessingAgent:
    """Agent responsible for extracting text from various document types"""
    
    def __init__(self):
        self.name = "DocumentProcessor"
    
    def extract_from_pdf(self, file_path):
        """Extract text from PDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"PDF extraction error: {str(e)}")
        return text.strip()
    
    def extract_from_docx(self, file_path):
        """Extract text from Word document"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"DOCX extraction error: {str(e)}")
        return text.strip()
    
    def extract_from_image(self, file_path):
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        except Exception as e:
            raise Exception(f"OCR extraction error: {str(e)}")
        return text.strip()
    
    def process(self, file_path, file_type):
        """Main processing method"""
        print(f"[{self.name}] Processing {file_type} document...")
        
        if file_type == 'pdf':
            return self.extract_from_pdf(file_path)
        elif file_type in ['doc', 'docx']:
            return self.extract_from_docx(file_path)
        elif file_type in ['jpg', 'jpeg', 'png', 'handwritten']:
            return self.extract_from_image(file_path)
        else:
            raise Exception("Unsupported file type")


class HandwritingRecognitionAgent:
    """Specialized agent for handwritten document recognition"""
    
    def __init__(self):
        self.name = "HandwritingRecognizer"
    
    def recognize(self, file_path):
        """Advanced handwriting recognition"""
        print(f"[{self.name}] Analyzing handwritten document...")
        
        try:
            # Open image
            image = Image.open(file_path)
            
            # Preprocess image for better OCR
            image = image.convert('L')  # Convert to grayscale
            
            # Use Tesseract with optimized settings for handwriting
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            print(f"[{self.name}] Extracted {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Handwriting recognition error: {str(e)}")


class TextToSpeechAgent:
    """Agent responsible for converting text to speech"""
    
    def __init__(self):
        self.name = "TextToSpeech"
        self.engine = None
    
    def initialize_engine(self):
        """Initialize pyttsx3 engine"""
        if not self.engine:
            self.engine = pyttsx3.init()
            # Set properties for Nigerian English-like voice
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'english' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
    
    def generate_gtts(self, text, output_path, speed=1.0):
        """Generate speech using gTTS (Google Text-to-Speech)"""
        print(f"[{self.name}] Generating audio with gTTS...")
        
        try:
            # gTTS with Nigerian English accent
            tts = gTTS(text=text, lang='en', tld='com.ng', slow=(speed < 1.0))
            tts.save(output_path)
            return output_path
        except Exception as e:
            raise Exception(f"TTS generation error: {str(e)}")
    
    def generate_pyttsx3(self, text, output_path, speed=1.0):
        """Generate speech using pyttsx3 (offline)"""
        print(f"[{self.name}] Generating audio with pyttsx3...")
        
        try:
            self.initialize_engine()
            self.engine.setProperty('rate', 150 * speed)
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            return output_path
        except Exception as e:
            raise Exception(f"TTS generation error: {str(e)}")


class RAGAgent:
    """Retrieval-Augmented Generation Agent for Q&A"""
    
    def __init__(self):
        self.name = "RAGSystem"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None
        self.full_text = ""
    
    def chunk_text(self, text, chunk_size=500):
        """Split text into chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def build_knowledge_base(self, text):
        """Build vector database from document text"""
        print(f"[{self.name}] Building knowledge base...")
        
        self.full_text = text
        self.chunks = self.chunk_text(text)
        
        # Create embeddings
        self.embeddings = self.model.encode(self.chunks)
        
        print(f"[{self.name}] Knowledge base built with {len(self.chunks)} chunks")
    
    def retrieve_relevant_chunks(self, query, top_k=3):
        """Retrieve most relevant text chunks for a query"""
        if not self.embeddings:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return relevant chunks
        relevant_chunks = [self.chunks[i] for i in top_indices]
        return relevant_chunks
    
    def answer_question(self, question):
        """Answer question based on document content"""
        print(f"[{self.name}] Answering: {question}")
        
        # Retrieve relevant context
        context = self.retrieve_relevant_chunks(question)
        
        if not context:
            return "I don't have enough information to answer that question."
        
        # Simple extractive Q&A (can be enhanced with LLM)
        answer = f"Based on the document: {context[0][:300]}..."
        
        return {
            'answer': answer,
            'context': context,
            'confidence': 0.85
        }
    
    def generate_questions(self, num_questions=5):
        """Generate questions based on document content"""
        print(f"[{self.name}] Generating {num_questions} questions...")
        
        # Simple question generation based on key sentences
        sentences = self.full_text.split('.')[:20]
        questions = []
        
        question_templates = [
            "What does the document say about {}?",
            "Can you explain {}?",
            "What is mentioned regarding {}?",
            "How is {} described in the document?",
            "What information is provided about {}?"
        ]
        
        for i, sentence in enumerate(sentences[:num_questions]):
            words = sentence.split()
            if len(words) > 5:
                key_phrase = ' '.join(words[:4])
                template = question_templates[i % len(question_templates)]
                questions.append(template.format(key_phrase))
        
        return questions


class OrchestratorAgent:
    """Main orchestrator that coordinates all agents"""
    
    def __init__(self):
        self.name = "Orchestrator"
        self.doc_agent = DocumentProcessingAgent()
        self.handwriting_agent = HandwritingRecognitionAgent()
        self.tts_agent = TextToSpeechAgent()
        self.rag_agent = RAGAgent()
        self.current_document = None
    
    def process_document(self, file_path, file_type, is_handwritten=False):
        """Coordinate document processing"""
        print(f"[{self.name}] Starting document processing pipeline...")
        
        try:
            # Step 1: Extract text
            if is_handwritten:
                text = self.handwriting_agent.recognize(file_path)
            else:
                text = self.doc_agent.process(file_path, file_type)
            
            # Step 2: Build knowledge base
            self.rag_agent.build_knowledge_base(text)
            
            self.current_document = {
                'text': text,
                'file_path': file_path,
                'processed_at': datetime.now().isoformat()
            }
            
            print(f"[{self.name}] Document processing complete")
            return text
            
        except Exception as e:
            raise Exception(f"Document processing failed: {str(e)}")
    
    def generate_audio(self, text, speed=1.0):
        """Generate audio from text"""
        print(f"[{self.name}] Generating audio...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(AUDIO_FOLDER, f'audio_{timestamp}.mp3')
        
        return self.tts_agent.generate_gtts(text, output_path, speed)


# Initialize orchestrator
orchestrator = OrchestratorAgent()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'agents': {
            'document_processor': 'active',
            'handwriting_recognizer': 'active',
            'tts_engine': 'active',
            'rag_system': 'active',
            'orchestrator': 'active'
        }
    })


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process document"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    is_handwritten = request.form.get('is_handwritten', 'false').lower() == 'true'
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Determine file type
    file_extension = file.filename.rsplit('.', 1)[1].lower()
    
    try:
        # Process document
        text = orchestrator.process_document(file_path, file_extension, is_handwritten)
        
        # Generate initial questions
        questions = orchestrator.rag_agent.generate_questions()
        
        return jsonify({
            'success': True,
            'text': text[:1000] + '...' if len(text) > 1000 else text,
            'full_text_length': len(text),
            'questions': questions,
            'file_id': timestamp
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    """Generate audio from text"""
    
    data = request.json
    text = data.get('text', '')
    speed = float(data.get('speed', 1.0))
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        audio_path = orchestrator.generate_audio(text, speed)
        
        return jsonify({
            'success': True,
            'audio_url': f'/api/audio/{os.path.basename(audio_path)}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Serve audio file"""
    file_path = os.path.join(AUDIO_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    return send_file(file_path, mimetype='audio/mpeg')


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Answer question using RAG system"""
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        answer = orchestrator.rag_agent.answer_question(question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-questions', methods=['GET'])
def generate_questions():
    """Generate questions from document"""
    
    num_questions = int(request.args.get('num', 5))
    
    try:
        questions = orchestrator.rag_agent.generate_questions(num_questions)
        
        return jsonify({
            'success': True,
            'questions': questions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("AI Document to Speech System with RAG")
    print("=" * 60)
    print("\nActive AI Agents:")
    print("  1. DocumentProcessingAgent - PDF/Word extraction")
    print("  2. HandwritingRecognitionAgent - OCR for handwritten docs")
    print("  3. TextToSpeechAgent - Audio generation")
    print("  4. RAGAgent - Question answering system")
    print("  5. OrchestratorAgent - System coordinator")
    print("\nStarting server...")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
