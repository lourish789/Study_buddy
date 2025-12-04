from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from datetime import datetime
import re
from typing import List, Dict
import tempfile
from collections import Counter, defaultdict
import math
import random

import PyPDF2
from docx import Document
from gtts import gTTS

app = Flask(__name__)

# FIXED CORS CONFIGURATION
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-Session-ID"],
        "expose_headers": ["X-Session-ID"]
    }
})

UPLOAD_FOLDER = tempfile.gettempdir()
AUDIO_FOLDER = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Session storage
sessions = {}

class DocumentProcessor:
    def extract_from_pdf(self, path: str) -> str:
        text = ""
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        if not text.strip():
            raise Exception("No text extracted from PDF")
        return self.clean_text(text)
    
    def extract_from_docx(self, path: str) -> str:
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if not text.strip():
            raise Exception("No text extracted from DOCX")
        return self.clean_text(text)
    
    def extract_from_txt(self, path: str) -> str:
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    text = f.read()
                if text.strip():
                    return self.clean_text(text)
            except:
                continue
        raise Exception("Could not read text file")
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def process(self, path: str, file_type: str) -> str:
        if file_type == 'pdf':
            return self.extract_from_pdf(path)
        elif file_type in ['doc', 'docx']:
            return self.extract_from_docx(path)
        else:
            return self.extract_from_txt(path)

class QuestionGenerator:
    def __init__(self):
        self.stop_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','as','is','was','are','were','be','been','have','has','had','do','does','did','this','that','it','its'}
    
    def get_sentences(self, text: str) -> List[str]:
        sents = re.split(r'(?<=[.!?])\s+', text)
        valid = []
        for s in sents:
            s = s.strip()
            if 20 <= len(s) <= 500 and len(s.split()) >= 4:
                valid.append(s)
        return valid[:100]
    
    def score_sentences(self, sents: List[str]) -> List[tuple]:
        tokens = []
        for s in sents:
            words = [w.lower() for w in re.findall(r'\b\w+\b', s) if w.lower() not in self.stop_words and len(w) > 2]
            tokens.append(words)
        
        df = defaultdict(int)
        for doc in tokens:
            for w in set(doc):
                df[w] += 1
        
        n = len(tokens)
        idf = {w: math.log((n+1)/(f+1))+1 for w,f in df.items()}
        
        scored = []
        for i, doc in enumerate(tokens):
            if not doc:
                scored.append((i, 0.0))
                continue
            tf = Counter(doc)
            score = sum(tf[w]/len(doc) * idf.get(w,0) for w in doc)
            score /= math.sqrt(len(doc))
            if re.search(r'\d', sents[i]):
                score *= 1.2
            scored.append((i, score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def generate_mcq(self, text: str, num: int = 5) -> List[Dict]:
        sents = self.get_sentences(text)
        if not sents:
            return []
        
        scored = self.score_sentences(sents)
        num = min(num, len(scored))
        questions = []
        
        templates = [
            "According to the text, {}?",
            "What does the text state about {}?",
            "Which of the following is true about {}?",
            "The text indicates that {}:",
            "Which statement best describes {}?"
        ]
        
        for i in range(num):
            idx, _ = scored[i]
            sent = sents[idx]
            
            words = sent.split()
            key = ' '.join(words[2:6]) if len(words) > 6 else ' '.join(words[:4])
            
            q_text = templates[i % len(templates)].format(key.lower())
            correct = sent[:150] if len(sent) <= 150 else sent[:147]+"..."
            
            distractors = [
                "This is not mentioned in the text",
                "The opposite of what is stated",
                "A related but incorrect statement"
            ]
            
            if idx > 0 and idx < len(sents)-1:
                distractors[1] = sents[idx-1][:150] if len(sents[idx-1]) <= 150 else sents[idx-1][:147]+"..."
            
            opts = [correct] + distractors[:3]
            random.shuffle(opts)
            correct_idx = opts.index(correct)
            
            questions.append({
                'id': f'mcq_{i+1}',
                'type': 'multiple_choice',
                'question': q_text,
                'options': opts,
                'correct_answer': correct_idx,
                'explanation': f"From: {sent[:200]}"
            })
        
        return questions
    
    def generate_theory(self, text: str, num: int = 4) -> List[Dict]:
        sents = self.get_sentences(text)
        if not sents:
            return []
        
        scored = self.score_sentences(sents)
        num = min(num, len(scored))
        questions = []
        
        templates = [
            "Explain what the text says about {}.",
            "Define {} as described in the text.",
            "Discuss the significance of {}.",
            "Describe {}."
        ]
        
        for i in range(num):
            idx, _ = scored[i]
            sent = sents[idx]
            words = sent.split()
            key = ' '.join(words[:5]) if len(words) > 5 else ' '.join(words[:3])
            
            q_text = templates[i % len(templates)].format(key.lower())
            
            context = []
            for j in range(max(0,idx-1), min(len(sents),idx+2)):
                context.append(sents[j])
            expected = ' '.join(context)[:400]
            
            kws = [w for w in key.lower().split() if w not in self.stop_words and len(w) > 3][:5]
            
            questions.append({
                'id': f'theory_{i+1}',
                'type': 'theory',
                'question': q_text,
                'expected_answer': expected,
                'keywords': kws,
                'points': 10
            })
        
        return questions
    
    def generate_essay(self, text: str, num: int = 3) -> List[Dict]:
        sents = self.get_sentences(text)
        if len(sents) < 5:
            return []
        
        scored = self.score_sentences(sents)
        num = min(num, len(scored)//2)
        questions = []
        
        templates = [
            "Discuss {} in detail.",
            "Write about {} based on the text.",
            "Analyze {}."
        ]
        
        for i in range(num):
            idx, _ = scored[i*2]
            sent = sents[idx]
            words = sent.split()
            key = ' '.join(words[:6])
            
            q_text = templates[i % len(templates)].format(key.lower())
            
            context = ' '.join(sents[max(0,idx-2):min(len(sents),idx+3)])
            
            questions.append({
                'id': f'essay_{i+1}',
                'type': 'essay',
                'question': q_text,
                'context': context,
                'min_words': 150
            })
        
        return questions

class AnswerEvaluator:
    def __init__(self):
        self.stop_words = {'the','a','an','and','or','but','in','on','at','to','for'}
    
    def tokenize(self, text: str) -> set:
        words = re.findall(r'\b\w+\b', text.lower())
        return set(w for w in words if w not in self.stop_words and len(w) > 2)
    
    def similarity(self, t1: str, t2: str) -> float:
        s1, s2 = self.tokenize(t1), self.tokenize(t2)
        if not s1 or not s2:
            return 0.0
        inter = s1 & s2
        union = s1 | s2
        return len(inter) / len(union)
    
    def eval_mcq(self, q: Dict, ans: int) -> Dict:
        try:
            ans = int(ans)
        except:
            return {'correct': False, 'feedback': 'Invalid answer', 'score': 0}
        
        correct = ans == q['correct_answer']
        return {
            'correct': correct,
            'user_answer': ans,
            'correct_answer': q['correct_answer'],
            'feedback': "✓ Correct!" if correct else f"✗ Wrong. Correct: option {q['correct_answer']+1}",
            'score': 100 if correct else 0
        }
    
    def eval_theory(self, q: Dict, ans: str) -> Dict:
        if not ans or not ans.strip():
            return {'score': 0, 'points_earned': 0, 'total_points': q.get('points',10), 'feedback': 'No answer'}
        
        sim = self.similarity(ans, q['expected_answer'])
        kws = q.get('keywords', [])
        kw_found = sum(1 for k in kws if k in ans.lower())
        kw_score = kw_found / len(kws) if kws else 0.5
        
        wc = len(ans.split())
        len_score = min(wc/50, 1.0)
        
        overall = sim*0.5 + kw_score*0.3 + len_score*0.2
        pts = int(overall * q.get('points',10))
        
        if overall >= 0.85:
            fb = "✓ Excellent!"
        elif overall >= 0.7:
            fb = "✓ Good answer"
        elif overall >= 0.5:
            fb = "○ Fair answer"
        else:
            fb = "✗ Needs improvement"
        
        return {
            'score': round(overall*100, 2),
            'points_earned': pts,
            'total_points': q.get('points',10),
            'feedback': fb,
            'expected_answer': q['expected_answer'][:300]
        }
    
    def eval_essay(self, q: Dict, ans: str) -> Dict:
        if not ans or not ans.strip():
            return {'score': 0, 'feedback': 'No answer'}
        
        wc = len(ans.split())
        min_w = q.get('min_words', 150)
        
        rel = self.similarity(ans, q.get('context',''))
        len_sc = min(wc/min_w, 1.0) if wc < min_w else 1.0
        
        overall = rel*0.6 + len_sc*0.4
        
        fb = []
        if wc < min_w*0.5:
            fb.append(f"✗ Too short (need {min_w}+ words)")
        elif wc < min_w:
            fb.append("○ Could be longer")
        else:
            fb.append("✓ Good length")
        
        if rel >= 0.5:
            fb.append("✓ Relevant")
        else:
            fb.append("○ Stay focused")
        
        return {
            'score': round(overall*100, 2),
            'word_count': wc,
            'feedback': ' '.join(fb)
        }

class RAGSystem:
    def __init__(self):
        self.chunks = []
        self.text = ""
    
    def build(self, text: str):
        self.text = text
        sents = re.split(r'(?<=[.!?])\s+', text)
        chunks, curr, size = [], [], 0
        
        for s in sents:
            wc = len(s.split())
            if size + wc > 300 and curr:
                chunks.append(' '.join(curr))
                curr, size = [s], wc
            else:
                curr.append(s)
                size += wc
        
        if curr:
            chunks.append(' '.join(curr))
        
        self.chunks = chunks
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        scores = []
        
        for chunk in self.chunks:
            c_words = set(re.findall(r'\b\w+\b', chunk.lower()))
            score = len(q_words & c_words)
            scores.append(score)
        
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top if scores[i] > 0]
    
    def answer(self, q: str) -> Dict:
        chunks = self.retrieve(q, 3)
        if not chunks:
            return {'answer': 'No relevant info found', 'confidence': 0.0}
        
        ans = f"Based on the document: {chunks[0][:300]}..."
        return {'answer': ans, 'confidence': 0.7, 'sources': [{'text': c[:150]} for c in chunks]}
    
    def summary(self) -> str:
        if not self.chunks:
            return ""
        n = min(5, len(self.chunks))
        step = len(self.chunks) // n if n > 0 else 1
        parts = [self.chunks[i*step] for i in range(n) if i*step < len(self.chunks)]
        return ' '.join(parts)[:800]

class Orchestrator:
    def __init__(self, session_id: str):
        self.id = session_id
        self.doc_proc = DocumentProcessor()
        self.qgen = QuestionGenerator()
        self.evaluator = AnswerEvaluator()
        self.rag = RAGSystem()
        self.doc = None
        self.questions = {}
    
    def process_doc(self, path: str, ftype: str) -> Dict:
        text = self.doc_proc.process(path, ftype)
        if len(text) < 100:
            raise Exception("Text too short")
        
        self.rag.build(text)
        self.doc = {
            'text': text,
            'word_count': len(text.split()),
            'summary': self.rag.summary(),
            'processed_at': datetime.now().isoformat()
        }
        return self.doc
    
    def gen_audio(self, text: str, speed: float = 1.0) -> str:
        if len(text) > 5000:
            text = text[:5000]
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(AUDIO_FOLDER, f'audio_{ts}.mp3')
        tts = gTTS(text=text, lang='en', tld='com.ng', slow=(speed<0.8))
        tts.save(path)
        return path
    
    def gen_questions(self, qtype: str = 'all', num: int = 5) -> Dict:
        text = self.doc['text']
        qs = {}
        
        if qtype in ['all','mcq']:
            qs['mcq'] = self.qgen.generate_mcq(text, num)
        if qtype in ['all','theory']:
            qs['theory'] = self.qgen.generate_theory(text, num)
        if qtype in ['all','essay']:
            qs['essay'] = self.qgen.generate_essay(text, max(3, num//2))
        
        self.questions = qs
        return qs
    
    def eval_answer(self, qid: str, qtype: str, ans) -> Dict:
        q = None
        for qs in self.questions.values():
            for item in qs:
                if item['id'] == qid:
                    q = item
                    break
        
        if not q:
            return {'error': 'Question not found'}
        
        if qtype == 'multiple_choice':
            return self.evaluator.eval_mcq(q, ans)
        elif qtype == 'theory':
            return self.evaluator.eval_theory(q, ans)
        elif qtype == 'essay':
            return self.evaluator.eval_essay(q, ans)
        
        return {'error': 'Unknown type'}

def get_orchestrator(session_id: str = None) -> Orchestrator:
    if not session_id:
        session_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    
    if session_id not in sessions:
        sessions[session_id] = Orchestrator(session_id)
    
    return sessions[session_id]

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'version': '2.1'})

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No filename'}), 400
    
    session_id = request.headers.get('X-Session-ID') or datetime.now().strftime('%Y%m%d%H%M%S%f')
    orch = get_orchestrator(session_id)
    
    ext = file.filename.rsplit('.',1)[1].lower() if '.' in file.filename else 'txt'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"{ts}_{file.filename}"
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    
    try:
        file.save(fpath)
        doc = orch.process_doc(fpath, ext)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'text_preview': doc['text'][:500]+'...',
            'word_count': doc['word_count'],
            'summary': doc['summary']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except:
                pass

@app.route('/api/generate-audio', methods=['POST'])
def gen_audio():
    data = request.json
    text = data.get('text', '')
    speed = float(data.get('speed', 1.0))
    use_sum = data.get('use_summary', False)
    
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    orch = get_orchestrator(session_id)
    
    if use_sum and orch.doc:
        text = orch.doc.get('summary', text)
    
    if not text:
        return jsonify({'error': 'No text'}), 400
    
    try:
        path = orch.gen_audio(text, speed)
        with open(path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode()
        os.remove(path)
        
        return jsonify({'success': True, 'audio_data': audio_data, 'format': 'mp3'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-questions', methods=['POST'])
def gen_questions():
    data = request.json
    qtype = data.get('type', 'all')
    num = int(data.get('num_questions', 5))
    
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    orch = get_orchestrator(session_id)
    if not orch.doc:
        return jsonify({'error': 'No document'}), 400
    
    try:
        qs = orch.gen_questions(qtype, num)
        return jsonify({'success': True, 'questions': qs, 'total': sum(len(v) for v in qs.values())})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate-answer', methods=['POST'])
def eval_ans():
    data = request.json
    qid = data.get('question_id')
    qtype = data.get('question_type')
    ans = data.get('answer')
    
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    if not all([qid, qtype, ans is not None]):
        return jsonify({'error': 'Missing fields'}), 400
    
    orch = get_orchestrator(session_id)
    
    try:
        result = orch.eval_answer(qid, qtype, ans)
        return jsonify({'success': True, 'evaluation': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    q = data.get('question', '')
    
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    if not q:
        return jsonify({'error': 'No question'}), 400
    
    orch = get_orchestrator(session_id)
    if not orch.doc:
        return jsonify({'error': 'No document'}), 400
    
    try:
        ans = orch.rag.answer(q)
        return jsonify({'success': True, 'question': q, **ans})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-summary', methods=['GET'])
def get_sum():
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    orch = get_orchestrator(session_id)
    if not orch.doc:
        return jsonify({'error': 'No document'}), 400
    
    return jsonify({'success': True, 'summary': orch.doc['summary'], 'word_count': orch.doc['word_count']})

@app.route('/api/batch-evaluate', methods=['POST'])
def batch_eval():
    data = request.json
    answers = data.get('answers', [])
    
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        return jsonify({'error': 'No session'}), 400
    
    if not answers:
        return jsonify({'error': 'No answers'}), 400
    
    orch = get_orchestrator(session_id)
    
    try:
        results = []
        total, max_score = 0, 0
        
        for a in answers:
            ev = orch.eval_answer(a['question_id'], a['question_type'], a['answer'])
            results.append({'question_id': a['question_id'], 'evaluation': ev})
            
            if 'score' in ev:
                total += ev['score']
                max_score += 100
            elif 'correct' in ev:
                total += 100 if ev['correct'] else 0
                max_score += 100
        
        pct = (total/max_score*100) if max_score > 0 else 0
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_questions': len(answers),
                'total_score': round(total, 2),
                'max_score': max_score,
                'percentage': round(pct, 2)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'AI Study Assistant', 'version': '2.1', 'status': 'active'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Enable debug mode to see errors
    app.run(debug=True, host='0.0.0.0', port=port)
