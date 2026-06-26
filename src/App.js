import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import Meyda from "meyda";
import { useMicrophone } from "./Microphone";

const LABELS = ["Ambulance", "Horn", "Train Horn", "Safe Background"];

export default function App() {
  const [model, setModel] = useState(null);
  const [logs, setLogs] = useState([]);
  const mic = useMicrophone();
  const analyzerRef = useRef(null);
  const mfccHistory = useRef([]); 

  // 1. Load the compiled TensorFlow.js engine
  useEffect(() => {
    tf.loadLayersModel("/tfjs_model/model.json")
      .then((m) => {
        setModel(m);
        console.log("Deep CNN Model loaded successfully!");
        
        // Warm up the GPU with a dummy tensor to prevent initial lag
        tf.tidy(() => m.predict(tf.zeros([1, 40, 174, 1])));
      })
      .catch((err) => console.error("Failed to load model. Check the public/tfjs_model path.", err));
  }, []);

  // 2. The GPU-Accelerated Prediction Loop
  const runPrediction = (frames) => {
    if (!model) return;

    const { classIdx, confidence } = tf.tidy(() => {
      // Convert raw array to 2D Tensor and transpose to match Python librosa [40, 32]
      let tensor2d = tf.tensor2d(frames).transpose(); 
      
      // Pad to exactly 174 frames to fit the CNN matrix
      const padAmount = 174 - tensor2d.shape[1];
      if (padAmount > 0) {
        tensor2d = tensor2d.pad([[0, 0], [0, padAmount]]);
      } else {
        tensor2d = tensor2d.slice([0, 0], [40, 174]);
      }

      // Mathematical Z-score standardization (Matches Colab training)
      const mean = tensor2d.mean();
      const variance = tensor2d.sub(mean).square().mean();
      const std = tf.sqrt(variance);
      const normalized = tensor2d.sub(mean).div(std.add(1e-7));

      // Reshape to 4D tensor format [Batch, Height, Width, Channels]
      const input = normalized.reshape([1, 40, 174, 1]);
      const pred = model.predict(input);
      
      return {
        classIdx: pred.argMax(1).dataSync()[0],
        confidence: pred.max().dataSync()[0]
      };
    });

    // 3. Hazard Alert Logic (Only alert if confidence > 85% and NOT the background noise)
    if (confidence > 0.85 && classIdx < 3) {
      // Haptic feedback logic for mobile devices
      const patterns = { 0: [300, 100, 300], 1: [200, 50, 200], 2: [500, 100, 500] };
      if ("vibrate" in navigator) navigator.vibrate(patterns[classIdx]);

      const newLog = { 
        label: LABELS[classIdx], 
        time: new Date().toLocaleTimeString(), 
        conf: Math.round(confidence * 100) 
      };
      
      // Update UI with the latest 5 alerts
      setLogs((prev) => [newLog, ...prev.slice(0, 4)]);
    }
  };

  // 4. Feature Extraction Loop
  useEffect(() => {
    if (mic.active && mic.source && mic.audioContext) {
      Meyda.numberOfMFCCCoefficients = 40;
      
      analyzerRef.current = Meyda.createMeydaAnalyzer({
        audioContext: mic.audioContext,
        source: mic.source,
        bufferSize: 512, 
        featureExtractors: ["mfcc"],
        callback: (features) => {
          if (features && features.mfcc) {
            mfccHistory.current.push(features.mfcc);
            
            // Dispatch prediction when we collect 32 chunks (~1 second of audio context)
            if (mfccHistory.current.length >= 32) {
              runPrediction(mfccHistory.current);
              mfccHistory.current = []; 
            }
          }
        }
      });
      analyzerRef.current.start();
    } else {
      if (analyzerRef.current) analyzerRef.current.stop();
      mfccHistory.current = [];
    }
    
    return () => {
      if (analyzerRef.current) analyzerRef.current.stop();
    };
  }, [mic.active, model]);

  // 5. User Interface
  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif", maxWidth: "450px", margin: "0 auto" }}>
      <h1 style={{ color: "#d32f2f", marginBottom: "5px" }}>HearSight Engine</h1>
      <p style={{ marginTop: "0", color: "#666", fontWeight: "bold" }}>Real-Time Acoustic Hazard Detection</p>
      
      <div style={{ padding: "10px", backgroundColor: "#e3f2fd", borderRadius: "8px", marginBottom: "20px" }}>
        <strong>System Status:</strong> {model ? "✅ Core Neural Network Online" : "⏳ Booting Core Sequence..."}
      </div>
      
      <button 
        onClick={mic.active ? mic.stop : mic.start}
        style={{
          width: "100%", padding: "16px", fontSize: "18px",
          backgroundColor: mic.active ? "#f44336" : "#4CAF50",
          color: "white", border: "none", borderRadius: "8px", cursor: "pointer",
          fontWeight: "bold", boxShadow: "0 4px 6px rgba(0,0,0,0.1)", transition: "0.2s"
        }}
      >
        {mic.active ? "Deactivate Scanner" : "Activate Scanner"}
      </button>

      <h3 style={{ marginTop: "30px", borderBottom: "2px solid #eee", paddingBottom: "10px" }}>Incident Log</h3>
      <div style={{ backgroundColor: "#fafafa", padding: "15px", borderRadius: "8px", minHeight: "250px", border: "1px solid #ddd" }}>
        {logs.length === 0 ? (
          <p style={{ color: "#888", textAlign: "center", marginTop: "35%" }}>
            Scanning environment for sirens and horns...
          </p>
        ) : null}
        
        {logs.map((log, i) => (
          <div key={i} style={{ borderBottom: i === logs.length - 1 ? "none" : "1px solid #eee", padding: "12px 0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "16px", fontWeight: "bold", color: "#d32f2f" }}>🚨 {log.label}</span>
              <span style={{ backgroundColor: "#ffeb3b", padding: "4px 8px", borderRadius: "12px", fontSize: "12px", fontWeight: "bold", color: "#333" }}>
                {log.conf}% Match
              </span>
            </div>
            <div style={{ fontSize: "12px", color: "#888", marginTop: "4px" }}>Timestamp: {log.time}</div>
          </div>
        ))}
      </div>
    </div>
  );
}