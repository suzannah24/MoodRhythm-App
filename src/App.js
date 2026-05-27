import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// --- DEPLOYMENT CONFIGURATION ---
// Replace the URL inside the quotes below with your actual live Render Web Service URL once you deploy it!
// Example: "https://moodrhythm-backend.onrender.com"
const BACKEND_API_URL = process.env.NODE_ENV === 'production' 
  ? "https://moodrhythm-backend.onrender.com" 
  : "http://localhost:8000";

const PLAYLIST_IDS = {
  happy: '4bexyPcwfp7h0kNvPOM1LD', 
  chill: '37i9dQZF1EVHGWrwldPRtj', 
  sad: '37i9dQZF1EIg85EO6f7KwU'    
};

const MOOD_TO_PLAYLIST = {
  happy: 'happy',
  surprise: 'happy',
  neutral: 'chill',
  sad: 'sad',
  angry: 'sad',
  fear: 'sad',
  disgust: 'sad'
};

const INSTRUMENTAL_MUSIC = {
  happy: process.env.PUBLIC_URL + '/music/happy.mp3',
  sad: process.env.PUBLIC_URL + '/music/sad.mp3',
  chill: process.env.PUBLIC_URL + '/music/chill.mp3',
  hype: process.env.PUBLIC_URL + '/music/happy.mp3'
};

function App() {
  const [step, setStep] = useState(1);
  const [detectedMood, setDetectedMood] = useState('neutral');
  const [playlistCategory, setPlaylistCategory] = useState('chill');
  const [confidence, setConfidence] = useState(0);
  const [playerType, setPlayerType] = useState('spotify'); 
  const [textInput, setTextInput] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [countdown, setCountdown] = useState(null);
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [debugData, setDebugData] = useState({ s_w: 0, b_h: 0, b_i: 0 });
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [showCorrectionMenu, setShowCorrectionMenu] = useState(false);
  const [lastScanFeatures, setLastScanFeatures] = useState(null);
  
  const videoRef = useRef(null);
  const dataBuffer = useRef([]); 

  useEffect(() => {
    async function initAI() {
      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO"
      });
      setFaceLandmarker(landmarker);
    }
    initAI();
  }, []);

  const goHome = () => {
    stopCamera();
    setStep(1);
    setCountdown(null);
    setTextInput("");
    setFeedbackSubmitted(false);
    setShowCorrectionMenu(false);
    dataBuffer.current = [];
    setLastScanFeatures(null);
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
  };

  const calculateDistances = () => {
    if (faceLandmarker && videoRef.current && videoRef.current.readyState >= 2) {
      const result = faceLandmarker.detectForVideo(videoRef.current, Date.now());
      if (result.faceLandmarks && result.faceLandmarks.length > 0) {
        const pts = result.faceLandmarks[0];
        const getDist = (p1, p2) => Math.hypot(pts[p1].x - pts[p2].x, pts[p1].y - pts[p2].y);

        const faceW = getDist(234, 454);
        if (faceW === 0) return;

        const features = {
          smile_w: getDist(61, 291) / faceW,
          smile_h: getDist(13, 14) / faceW,
          brow_h: Math.abs(pts[52].y - pts[33].y) / faceW,
          eye_h: Math.abs(pts[159].y - pts[145].y) / faceW,
          brow_i: getDist(55, 285) / faceW,
          mouth_drop: getDist(61, 164) / faceW,
          brow_v_left: getDist(107, 159) / faceW,
          brow_v_right: getDist(336, 386) / faceW,
          lip_dist: getDist(13, 14) / faceW,
          jaw_dist: getDist(10, 152) / faceW
        };

        setDebugData({ 
            s_w: features.smile_w.toFixed(2), 
            b_h: features.brow_h.toFixed(2), 
            b_i: features.brow_i.toFixed(2) 
        });

        dataBuffer.current.push(features);
      }
    }
  };

  const startCamera = async () => {
    setStep(3);
    setCountdown(3);
    dataBuffer.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
      
      const captureInterval = setInterval(calculateDistances, 100);

      const timer = setInterval(() => {
        setCountdown(prev => {
          if (prev <= 1) {
            clearInterval(timer);
            clearInterval(captureInterval);
            setIsAnalyzing(true);
            setTimeout(() => sendToBackend(stream), 500);
            return null;
          }
          return prev - 1;
        });
      }, 1000);
    } catch (err) { alert("Camera access denied."); }
  };

  const sendToBackend = async (stream) => {
    const buf = dataBuffer.current;
    if (buf.length === 0) { setIsAnalyzing(false); setStep(2); return; }

    const keys = ['smile_w', 'smile_h', 'brow_h', 'eye_h', 'brow_i', 'mouth_drop', 'brow_v_left', 'brow_v_right', 'lip_dist', 'jaw_dist'];
    const averages = {};
    keys.forEach(key => {
        averages[key] = buf.reduce((a, b) => a + b[key], 0) / buf.length;
    });

    try {
      const res = await fetch(`${BACKEND_API_URL}/analyze-face`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(averages),
      });
      const data = await res.json();
      
      setDetectedMood(data.detected_mood);
      setPlaylistCategory(MOOD_TO_PLAYLIST[data.detected_mood] || 'chill');
      setConfidence(data.confidence);
      setLastScanFeatures(averages);
      setStep(5);
    } catch (e) { 
        console.error("Backend Error:", e);
        setStep(5); 
    } finally {
      setIsAnalyzing(false);
      if (stream) stream.getTracks().forEach(t => t.stop());
    }
  };

  const handleFeedback = async (isCorrect, manualMood = null) => {
    const finalMood = isCorrect ? detectedMood : manualMood;
    setFeedbackSubmitted(true);
    setShowCorrectionMenu(false);
    
    if (lastScanFeatures && finalMood) {
      try {
        await fetch(`${BACKEND_API_URL}/save-feedback?confirmed_mood=${finalMood}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(lastScanFeatures),
        });
        console.log(`Saved features successfully under verified label: ${finalMood}`);
      } catch (e) {
        console.error("Failed to log targeted feedback", e);
      }
    }
  };

  const handleTextAnalysis = async () => {
    setIsAnalyzing(true);
    try {
      const res = await fetch(`${BACKEND_API_URL}/analyze-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textInput }),
      });
      const data = await res.json();
      setDetectedMood(data.detected_mood);
      setPlaylistCategory(MOOD_TO_PLAYLIST[data.detected_mood] || 'chill');
      setStep(5);
    } catch (e) { setDetectedMood('chill'); setStep(5); }
    setIsAnalyzing(false);
  };

  return (
    <div className="App">
      <div className="brand-logo" onClick={goHome}>
        <div className="brand-dot"></div>MoodRhythm
      </div>
      <div className="glow-bg"></div>

      {step === 1 && (
        <div className="container">
          <h1 className="main-title">How's your vibe?</h1>
          <p className="subtitle">AI-Powered Mood Detection</p>
          <button className="btn-primary" onClick={() => setStep(2)}>PLAY ME MUSIC</button>
        </div>
      )}

      {step === 2 && (
        <div className="container">
          <h2 className="step-title">Select Method</h2>
          <div className="card-container">
            <div className="glass-card" onClick={startCamera}>
              <div className="card-icon">📷</div>
              <h3>SCAN FACE</h3>
            </div>
            <div className="glass-card" onClick={() => setStep(4)}>
              <div className="card-icon">⌨️</div>
              <h3>WRITE MOOD</h3>
            </div>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="container">
          <div className="scan-box">
            <video ref={videoRef} autoPlay playsInline className="video-feed" />
            {countdown && <div className="countdown-overlay">{countdown}</div>}
            <div className="scan-line"></div>
            <div className="debug-label">
              SVM: SW:{debugData.s_w} | BH:{debugData.b_h} | BI:{debugData.b_i}
            </div>
          </div>
          <h2 className="status-text">{isAnalyzing ? "SVM AI Analyzing..." : "Hold Still..."}</h2>
        </div>
      )}

      {step === 4 && (
        <div className="container">
          <div className="glass-card full-width">
            <h2 className="step-title">Describe your mood</h2>
            <textarea 
              className="mood-textarea" 
              placeholder="I'm feeling..." 
              value={textInput} 
              onChange={(e) => setTextInput(e.target.value)}
            />
            <div className="btn-row">
              <button onClick={() => setStep(2)} className="back-btn">Back</button>
              <button className="btn-primary" onClick={handleTextAnalysis}>ANALYZE VIBE</button>
            </div>
          </div>
        </div>
      )}

      {step === 5 && (
        <div className="container">
          <div className="glass-card full-width">
            <h2 className="mood-title">{detectedMood} Vibe</h2>
            <p className="confidence-text">AI Confidence: {(confidence * 100).toFixed(1)}%</p>
            
            <div className="toggle-group">
              <button onClick={() => setPlayerType('spotify')} className={playerType === 'spotify' ? 'toggle-btn active' : 'toggle-btn'}>Spotify</button>
              <button onClick={() => setPlayerType('free')} className={playerType === 'free' ? 'toggle-btn active' : 'toggle-btn'}>Free Mode</button>
            </div>

            <div className="player-container">
              {playerType === 'spotify' ? (
                <iframe 
                  title="Spotify Mood Playlist"
                  src={`https://open.spotify.com/embed/playlist/${PLAYLIST_IDS[playlistCategory]}?utm_source=generator&theme=0`} 
                  width="100%" height="352" frameBorder="0" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"
                ></iframe>
              ) : (
                <div className="local-player">
                  <p>Playing {playlistCategory} Instrumental...</p>
                  <audio controls autoPlay src={INSTRUMENTAL_MUSIC[playlistCategory]} style={{ width: '100%' }} />
                </div>
              )}
            </div>

            <div className="feedback-container">
              {!feedbackSubmitted ? (
                <>
                  {!showCorrectionMenu ? (
                    <>
                      <p className="feedback-text">Does this match your {detectedMood} mood?</p>
                      <div className="feedback-btn-group">
                        <button className="feedback-btn feedback-btn-yes" onClick={() => handleFeedback(true)}>👍 Yes</button>
                        <button className="feedback-btn feedback-btn-no" onClick={() => setShowCorrectionMenu(true)}>👎 No</button>
                      </div>
                    </>
                  ) : (
                    <>
                      <p className="feedback-text">What was your actual mood?</p>
                      <div className="feedback-btn-group correction-menu" style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', justifyContent: 'center' }}>
                        {['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust'].map((mood) => (
                          <button 
                            key={mood} 
                            className="toggle-btn" 
                            style={{ padding: '8px 15px', textTransform: 'uppercase', fontSize: '12px' }}
                            onClick={() => handleFeedback(false, mood)}
                          >
                            {mood}
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </>
              ) : (
                <p className="thank-you-message">Thank you! Your expression was saved to train the AI. ✨</p>
              )}
            </div>

            <button className="btn-primary mt-25" onClick={goHome}>NEW MOOD</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;